"""蒸留した学生(Qwen3-1.7B + LoRA)を chakuho の `core.query_backend` として差し込む。

`core.query_backend = student_backend(base, adapter)` と差し替えると、学生が
**本番と同じ判断経路**(トーナメント・__none__・coverage・noul・score の集約まで
core が行う)を通る。契約は ollama_backend / 本番 vLLM と同じ:
    (prompt, backend_url, model, *, timeout, images) -> (top_logprobs, prompt_tokens)

プロンプトの組み立ては distill.train.build_input_ids に委ねる —— **学習時と同じ関数**
なので、学習・評価・この実地試験でプロンプトがずれない。core.query_backend 側の
messages 構成(system + user + assistant "Label:" の prefill)とも一致する。

画像は扱えない(学生はテキストのみ)。images 付きで呼ばれたら例外にする —— 黙って
無視すると「画像を見た上での判定」と誤読されるため。
"""
from __future__ import annotations

import math
import threading
from typing import Any

_STATE: dict[str, Any] = {}
# core.evaluate は質問ごと・トーナメントのチャンクごとに ThreadPoolExecutor を張る。
# ロック無しだと**同じモデルを複数スレッドが同時に読み込む**(2026-09-21 実測: 進捗バーが
# 二重に出て、1.7B が何本も GPU に載りかけた)。読み込みは 1 度きりにする。
_LOCK = threading.Lock()


def _load(base: str, adapter: str | None):
    if _STATE.get("key") == (base, adapter):
        return _STATE["model"], _STATE["tok"]
    with _LOCK:
        if _STATE.get("key") == (base, adapter):  # ロック待ちの間に別スレッドが読み終えている
            return _STATE["model"], _STATE["tok"]
        return _load_locked(base, adapter)


def _load_locked(base: str, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    _STATE.update(key=(base, adapter), model=model, tok=tok, torch=torch)
    return model, tok


def student_backend(base: str, adapter: str | None = None, top_k: int | None = None):
    """core.query_backend と同じ契約の関数を返す。"""
    from chakuho import core
    from distill.train import build_input_ids

    def q(prompt: str, backend_url: str, model: str, *, timeout: float = 120.0,
          images: list[str] | None = None) -> tuple[dict[str, float], int]:
        if images:
            raise RuntimeError("学生はテキストのみ。画像付きの判定は扱えない(黙って無視しない)")
        m, tok = _load(base, adapter)
        torch = _STATE["torch"]
        k = top_k if top_k is not None else core.top_logprobs_limit()

        ids = build_input_ids(tok, core.SYSTEM_PROMPT, prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device=m.device)
        with torch.no_grad():
            logits = m(input_ids=input_ids, logits_to_keep=1).logits[0, -1, :]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        vals, idx = torch.topk(logprobs, min(k, logprobs.shape[-1]))
        top: dict[str, float] = {}
        for v, i in zip(vals.tolist(), idx.tolist()):
            t = tok.decode([i])
            # 別のトークン id が同じ表層へ落ちることがある。確率を足して 1 つにまとめる
            # (core.aggregate は token 文字列で照合するので、上書きすると質量が消える)
            top[t] = math.log(math.exp(top[t]) + math.exp(v)) if t in top else v
        return top, len(ids)

    return q
