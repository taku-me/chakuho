"""④ 学習: 学生(Qwen3-1.7B 等)へ 27B の教師分布を LoRA で蒸留する。

CUDA コンテナ内(torch/transformers/peft/datasets インストール済み)で実行する前提
(docs/distill-design.md「置き場」節)。このリポジトリの開発機には torch が無いため、
このモジュールは **import 時に重い依存を要求しない**: torch 系の import は try/except
で遅延させ、:func:`label_token_ids` のような純粋関数だけを torch 抜きでテストできるように
分離してある。実際の学習・評価を行う関数は呼び出し時に :func:`_require_torch` でガードする。

損失: ラベルトークン集合上に制限した softmax で KL(teacher ‖ student) を、プロンプト末尾
(生成直前)の 1 位置だけで計算する。多トークンラベル(yes/no)は最初のトークンだけを使う
(docs/distill-design.md ④ 学習節、検証 OK 条件 11)。``--lm-weight w > 0`` で、同じバッチに
プロンプト全体の標準 LM 損失を重み w で加算できる(条件 11 の比較用)。

使い方:
  python3 -m distill.train --data dataset.jsonl --model Qwen/Qwen3-1.7B --out out/ \\
      --epochs 4 --lr 1e-4 --lora-r 16 --batch 8 --lm-weight 0.0 \\
      --eval-cases cases.json --eval-cases cases2.json --eval-cases2 holdout.json \\
      --eval-every-epoch --seed 0

  # LoRA を融合してフルモデルを保存(mlx_lm.convert への入力にする)
  python3 -m distill.train ... --merge
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distill.eval import score_cases  # noqa: E402  stdlib のみ、常に import 可能

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    _TORCH_AVAILABLE = True
except ImportError:  # このリポジトリの開発機には torch が無い
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]
    DataLoader = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "torch/transformers/peft がこの環境に無い。CUDA コンテナ内で実行すること"
            "(docs/distill-design.md の「置き場」節、docker run nvcr.io/nvidia/pytorch:<tag> を参照)。"
        )


# ---------------------------------------------------------------------------
# torch 無しでテストできる純粋関数群
# ---------------------------------------------------------------------------


def label_token_ids(tokenizer: Any, labels: list[str]) -> list[int]:
    """各ラベルの最初のトークン id を返す。

    多トークンラベル(yes/no 等)は tokenizer がそのラベルを 2 個以上のトークンへ
    分割していても先頭だけを使う(検証 OK 条件の指示どおり)。ラベル間で id が
    衝突していたら(異なるラベルが同じ最初のトークンを共有する)、KL 計算の
    前提が崩れるので学習前に検出する。

    tokenizer は ``.encode(text, add_special_tokens=False) -> list[int]`` を持てば
    よい(実 HF tokenizer とダック型で揃えたテスト用の fake でも動く)。

    Raises:
        ValueError: いずれかのラベルがトークン化できない、またはラベル間で
            最初のトークン id が重複する場合。
    """
    ids: list[int] = []
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if not encoded:
            raise ValueError(f"label {label!r} tokenized to zero tokens")
        ids.append(encoded[0])
    if len(set(ids)) != len(ids):
        raise ValueError(f"label token ids are not distinct: {dict(zip(labels, ids))}")
    return ids


def load_dataset_jsonl(path: str, limit: int = 0) -> list[dict]:
    """distill/build_dataset.py が書いた 1 行 1 例の JSONL を読む。"""
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if limit:
        records = records[:limit]
    return records


def load_cases(*paths: str) -> list[dict]:
    from distill.eval import load_cases as _load

    return _load(*paths)


def kl_from_teacher_dict(teacher: dict[str, float], labels: list[str], log_q: list[float]) -> float:
    """教師分布(dict)と学生の log_softmax(labels 上、python list)から KL(teacher ‖ student) を計算する。

    torch 無しでテストできるように tensor ではなく list[float] を受け取る形にしてある。
    実際の学習では :func:`kl_teacher_student` が torch tensor 版を計算する。
    """
    import math

    total_p = sum(teacher.get(label, 0.0) for label in labels)
    if total_p <= 0:
        raise ValueError("teacher distribution has zero mass on the requested labels")
    kl = 0.0
    for label, lq in zip(labels, log_q):
        p = teacher.get(label, 0.0) / total_p
        if p <= 0:
            continue
        kl += p * (math.log(p) - lq)
    return kl


# ---------------------------------------------------------------------------
# torch が要る本体(このリポジトリの開発機では import 可能・実行不可)
# ---------------------------------------------------------------------------


def kl_teacher_student(teacher: dict[str, float], student_logits: "torch.Tensor", label_ids: list[int], labels: list[str]) -> "torch.Tensor":
    """label_ids に制限した softmax で KL(teacher ‖ student) を計算する(1 例分)。"""
    _require_torch()
    picked = student_logits[..., label_ids]
    log_q = F.log_softmax(picked.float(), dim=-1)
    p = torch.tensor([teacher.get(label, 0.0) for label in labels], dtype=torch.float32, device=picked.device)
    total = p.sum().clamp_min(1e-12)
    p = p / total
    log_p = torch.log(p.clamp_min(1e-12))
    return torch.sum(p * (log_p - log_q))


class DistillDataset(Dataset):  # type: ignore[misc]
    def __init__(self, records: list[dict]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


def build_input_ids(tokenizer: Any, system: str, prompt: str, *, enable_thinking: bool = False) -> list[int]:
    """chakuho core と同じ messages 形(system + user)をチャットテンプレートへ通す。"""
    # chakuho core と同じ形: 答えの合図 "Label:" は assistant 側の書き出し(prefill)。
    # 教師データの prompt(user 側)には合図が無いので、ここで assistant メッセージとして足し、
    # continue_final_message で "Label:" の直後の 1 トークンを学生に出させる。
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt},
                {"role": "assistant", "content": "Label:"}]
    # tokenize=True の戻り値は transformers の版で list / BatchEncoding と揺れる(5.x は dict)。
    # テキストにしてから自分でトークン化し、常に list[int] を返す。
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
        enable_thinking=enable_thinking,
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return list(ids)


def collate(batch: list[dict], tokenizer: Any) -> dict[str, Any]:
    _require_torch()
    encoded = [build_input_ids(tokenizer, r["system"], r["prompt"]) for r in batch]
    lengths = [len(e) for e in encoded]
    max_len = max(lengths)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    # 左パディング: 全行の「最後のプロンプト位置」が列 -1 に揃うので、logits_to_keep=1 で
    # 最終位置の logits だけ計算できる(全位置の logits は batch 8 × 1500 トークン × 語彙 15 万で
    # 数十 GB を食い、GB10 のユニファイドメモリを使い切って機体が凍結した実測 2026-09-20)。
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, ids in enumerate(encoded):
        input_ids[i, max_len - len(ids):] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, max_len - len(ids):] = 1
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids,
            "lengths": lengths, "records": batch}


def make_student_decider(model: Any, tokenizer: Any) -> Callable[[str, str, list[str]], dict[str, float]]:
    """distill.eval.score_cases 用の decide(system, prompt, labels) を作る(in-process、サーバ不要)。

    学生へ system+prompt をそのまま入力し、各ラベルの最初のトークンに載った
    (語彙全体の softmax における)生の確率を返す。学習損失(kl_teacher_student)と
    同じラベルトークンの取り方に揃えてあるので、学習中の評価と学習後の実配信で
    coverage の定義がずれない。
    """
    _require_torch()

    def decide(system: str, prompt: str, labels: list[str]) -> dict[str, float]:
        was_training = model.training
        model.eval()
        try:
            ids = build_input_ids(tokenizer, system, prompt)
            input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
            with torch.no_grad():
                logits = model(input_ids=input_ids, logits_to_keep=1).logits[0, -1, :]
                probs = torch.softmax(logits.float(), dim=-1)
            label_ids = label_token_ids(tokenizer, labels)
            return {label: probs[tid].item() for label, tid in zip(labels, label_ids)}
        finally:
            if was_training:
                model.train()

    return decide


def run_eval(model: Any, tokenizer: Any, cases_paths: list[str]) -> dict[str, Any]:
    decide = make_student_decider(model, tokenizer)
    return score_cases(load_cases(*cases_paths), decide)


def train_one_epoch(
    model: Any,
    tokenizer: Any,
    loader: Any,
    optimizer: Any,
    scheduler: Any,
    *,
    lm_weight: float,
    grad_accum: int,
) -> dict[str, float]:
    _require_torch()
    model.train()
    total_kl = 0.0
    n = 0
    optimizer.zero_grad()
    n_batches = 0
    for step, batch in enumerate(loader):
        n_batches += 1
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        position_ids = batch["position_ids"].to(model.device)
        if lm_weight > 0:  # LM 損失には全位置の logits が要る(メモリ大。条件 11 の比較用)
            out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        else:  # 最終位置だけ(左パディングなので列 -1 が全行のプロンプト末尾)
            out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, logits_to_keep=1)
        logits = out.logits

        batch_loss = torch.zeros((), device=model.device, dtype=torch.float32)
        for i, record in enumerate(batch["records"]):
            last_logits = logits[i, -1, :]
            label_ids = label_token_ids(tokenizer, record["labels"])
            kl = kl_teacher_student(record["teacher"], last_logits, label_ids, record["labels"])
            batch_loss = batch_loss + kl
            total_kl += float(kl.detach())
            n += 1
        batch_loss = batch_loss / len(batch["records"])

        if lm_weight > 0:
            lm_logits = logits[:, :-1, :]
            lm_targets = input_ids[:, 1:]
            lm_mask = attention_mask[:, 1:].reshape(-1).float()
            lm_loss_per_token = F.cross_entropy(
                lm_logits.reshape(-1, lm_logits.size(-1)), lm_targets.reshape(-1), reduction="none"
            )
            lm_loss = (lm_loss_per_token * lm_mask).sum() / lm_mask.sum().clamp_min(1)
            batch_loss = batch_loss + lm_weight * lm_loss

        (batch_loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    if n_batches % grad_accum != 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return {"kl_mean": total_kl / max(1, n), "n_examples": n}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="distill/build_dataset.py の出力 (jsonl)")
    ap.add_argument("--model", required=True, help="例: Qwen/Qwen3-1.7B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1, help="勾配累積のステップ数")
    ap.add_argument("--lm-weight", type=float, default=0.0, help="プロンプト全体の LM 損失の重み(条件 11)")
    ap.add_argument("--eval-cases", action="append", default=[], help="336 ベンチのケース JSON。複数回指定で連結")
    ap.add_argument("--eval-cases2", action="append", default=[], help="hold-out(実運用ログ由来)のケース JSON")
    ap.add_argument("--eval-every-epoch", action="store_true", help="毎 epoch 評価する(無指定なら最終 epoch のみ)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="デバッグ用: 学習例の上限(smoke test)")
    ap.add_argument("--no-grad-checkpoint", action="store_true", help="勾配チェックポイントを切る(既定は有効。メモリ優先)")
    ap.add_argument("--merge", action="store_true", help="学習後に LoRA を merge_and_unload しフルモデルを保存する")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    _require_torch()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_dataset_jsonl(args.data, limit=args.limit)
    if not records:
        raise ValueError(f"{args.data} に例が無い")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if not args.no_grad_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()  # LoRA + checkpointing で入力に勾配を通す(定石)

    loader = DataLoader(
        DistillDataset(records),
        batch_size=args.batch,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer),
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    steps_per_epoch = max(1, len(loader) // max(1, args.grad_accum))
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps // 20), num_training_steps=total_steps
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        stats = train_one_epoch(
            model, tokenizer, loader, optimizer, scheduler, lm_weight=args.lm_weight, grad_accum=args.grad_accum
        )
        epoch_dir = out_dir / f"epoch-{epoch}"
        model.save_pretrained(epoch_dir)

        record: dict[str, Any] = {
            "epoch": epoch,
            "epochs": args.epochs,
            "kl_mean": stats["kl_mean"],
            "n_examples": stats["n_examples"],
            "seconds": round(time.time() - t0, 1),
            "adapter_dir": str(epoch_dir),
            "gpu_max_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if torch.cuda.is_available() else None,
            "gpu_max_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 1) if torch.cuda.is_available() else None,
        }
        should_eval = args.eval_cases and (args.eval_every_epoch or epoch == args.epochs)
        if should_eval:
            record["eval"] = run_eval(model, tokenizer, args.eval_cases)
            if args.eval_cases2:
                record["eval_holdout"] = run_eval(model, tokenizer, args.eval_cases2)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    if args.merge:
        merged = model.merge_and_unload()
        merged_dir = out_dir / "merged"
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(json.dumps({"merged": str(merged_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
