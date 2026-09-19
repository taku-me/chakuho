"""Jev 互換のローカル判定エンドポイント。vLLM(qwen3.8-27b)の 1 トークン logprob で答える。

    python3 jev_shim.py --port 9750            # POST /v1/systemone を受ける

リクエスト/レスポンスは jev-mario が使う範囲の Jev API 形式:
  req : {"state": <json>, "questions": {name: {"type": choice|noul|score, "instructions": str|dict, "criteria": dict|list}}}
  resp: {"answers": {name: {"choice": str, "probabilities": {opt: p}} | {"noul": p} | {"score": s, "probabilities": {...}}},
         "usage": {"input_tokens": n}, "latency_ms": {...}}
質問は同じ state に対して並列に投げる(vLLM がバッチ化する)。
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VLLM = "http://localhost:8006/v1/chat/completions"
MODEL = "qwen3.8-27b"
SYSTEM = ("You are a decision engine inside a program. You never explain. "
          "You answer with exactly one label from the allowed list.")


def _labels(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


def _vllm_first_token(user: str) -> tuple[dict[str, float], int]:
    body = {"model": MODEL, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(VLLM, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    top = {e["token"]: e["logprob"] for e in d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}
    return top, d["usage"]["prompt_tokens"]


def _dist(top: dict[str, float], labels: list[str]) -> tuple[dict[str, float], float]:
    mass = {l: 0.0 for l in labels}
    for t, lp in top.items():
        key = t.strip().upper()
        if key in mass:
            mass[key] += math.exp(lp)
    cover = sum(mass.values())
    if cover == 0:
        return {l: 1.0 / len(labels) for l in labels}, 0.0
    return {l: v / cover for l, v in mass.items()}, cover


def _text(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return "\n".join(f"{k}: {_text(v)}" for k, v in x.items())
    return json.dumps(x, ensure_ascii=False)


def answer(state_text: str, name: str, q: dict) -> dict:
    kind = q["type"]
    instr = _text(q.get("instructions", ""))
    crit = q.get("criteria")
    if kind == "choice":
        opts = list(crit.keys()) if isinstance(crit, dict) else list(crit)
        labels = _labels(len(opts))
        menu = "\n".join(f"{l}: {o}" + (f" — {crit[o]}" if isinstance(crit, dict) else "") for l, o in zip(labels, opts))
        user = f"STATE:\n{state_text}\n\nINSTRUCTIONS:\n{instr}\n\nQUESTION ({name}): choose one option.\n{menu}\nAllowed labels: {', '.join(labels)}\nLabel:"
        top, ntok = _vllm_first_token(user)
        dist, cover = _dist(top, labels)
        probs = {o: dist[l] for l, o in zip(labels, opts)}
        best = max(probs, key=probs.get)
        return {"choice": best, "probabilities": probs, "coverage": cover, "input_tokens": ntok}
    if kind == "noul":
        c = crit if isinstance(crit, dict) else {}
        user = (f"STATE:\n{state_text}\n\nINSTRUCTIONS:\n{instr}\n\nQUESTION ({name}): yes or no.\n"
                f"yes: {c.get('true', 'yes')}\nno: {c.get('false', 'no')}\nAllowed labels: yes, no\nLabel:")
        top, ntok = _vllm_first_token(user)
        mass = {"yes": 0.0, "no": 0.0}
        for t, lp in top.items():
            k = t.strip().lower()
            if k in mass:
                mass[k] += math.exp(lp)
        cover = mass["yes"] + mass["no"]
        p = mass["yes"] / cover if cover else 0.5
        return {"noul": p, "coverage": cover, "input_tokens": ntok}
    if kind == "score":
        levels = list(crit) if isinstance(crit, list) else [str(i) for i in range(5)]
        labels = [str(i + 1) for i in range(len(levels))]
        menu = "\n".join(f"{l}: {lv}" for l, lv in zip(labels, levels))
        user = f"STATE:\n{state_text}\n\nINSTRUCTIONS:\n{instr}\n\nQUESTION ({name}): rate on this scale.\n{menu}\nAllowed labels: {', '.join(labels)}\nLabel:"
        top, ntok = _vllm_first_token(user)
        dist, cover = _dist(top, labels)
        n = len(labels)
        mean = sum(int(l) * p for l, p in dist.items())
        score = (mean - 1) / (n - 1) if n > 1 else 0.0  # 0..1 に正規化
        return {"score": score, "probabilities": {lv: dist[l] for l, lv in zip(labels, levels)}, "coverage": cover, "input_tokens": ntok}
    raise ValueError(f"unknown question type {kind}")


def systemone(req: dict) -> dict:
    state_text = _text(req["state"]) if not isinstance(req["state"], str) else req["state"]
    qs = req["questions"]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max(1, len(qs))) as ex:
        results = list(ex.map(lambda kv: answer(state_text, kv[0], kv[1]), qs.items()))
    answers = dict(zip(qs.keys(), results))
    tokens = sum(a.pop("input_tokens", 0) for a in answers.values())
    return {"answers": answers, "usage": {"input_tokens": tokens}, "model": MODEL,
            "latency_ms": round((time.perf_counter() - t0) * 1000)}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 静かに
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        try:
            out = systemone(req)
            code = 200
        except Exception as e:  # noqa: BLE001
            out, code = {"error": f"{type(e).__name__}: {e}"}, 500
        body = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        with open("jev_shim.log.jsonl", "a") as fh:
            fh.write(json.dumps({"t": time.time(), "questions": list(req.get("questions", {})), "resp": out}) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9750)
    a = ap.parse_args()
    print(f"jev shim on :{a.port} -> {VLLM} ({MODEL})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
