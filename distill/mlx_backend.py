"""⑥ 配布: 蒸留した学生を mac 上で常駐配信する最小 OpenAI 互換サーバ(mlx_lm 使用)。

chakuho/core.py の ``query_backend()`` が読む応答形状に厳密に合わせる:

    choices[0].logprobs.content[0].top_logprobs = [{"token": str, "logprob": float}, ...]
    usage.prompt_tokens = int

このサーバは 1 トークン判定専用: ``max_tokens`` は 1 以外を 400 で拒否する。
``top_logprobs`` は宣言済みラベル集合をサーバ側が知らないため、語彙全体からの
上位 N 個を返す(既定 64、リクエストの ``top_logprobs`` を尊重する)。
``chat_template_kwargs.enable_thinking`` を honor する(Qwen3 系、既定 False)。

mlx_lm はこの開発機には入っていない。**mlx への依存(モデルロード・フォワード)は
:class:`StudentModel` に閉じ込め、それ以外の純粋関数(build_messages / select_top_k /
log_softmax / build_response / validate_request)は mlx 抜きでテストできる。**
テストでは :class:`StudentModel` と同じ最小インターフェース(``model_path`` 属性、
``decide()``、``token_text()``)を持つ fake を ``make_handler`` に渡す。

CLI:
  python3 -m distill.mlx_backend --model <path> --host 0.0.0.0 --port 8006
"""

from __future__ import annotations

import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

try:
    import mlx_lm  # type: ignore

    _MLX_AVAILABLE = True
except ImportError:  # このリポジトリの開発機には mlx が無い(mac 側の配信機で使う)
    mlx_lm = None  # type: ignore[assignment]
    _MLX_AVAILABLE = False

DEFAULT_TOP_LOGPROBS = 64


# ---------------------------------------------------------------------------
# 純粋関数(mlx 抜きでテストできる)
# ---------------------------------------------------------------------------


def build_messages(body: dict[str, Any]) -> list[dict[str, str]]:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    for m in messages:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError("each message must have 'role' and 'content'")
    return messages


def validate_request(body: dict[str, Any]) -> tuple[list[dict[str, str]], int, bool]:
    """リクエスト本体を検証し、(messages, top_logprobs, enable_thinking) を返す。

    Raises:
        ValueError: ``messages`` が無い/不正、``max_tokens`` が 1 以外、
            ``top_logprobs`` が正の整数でない場合。
    """
    messages = build_messages(body)
    max_tokens = body.get("max_tokens", 1)
    if max_tokens != 1:
        raise ValueError(
            f"max_tokens must be 1 (got {max_tokens!r}); this backend only serves 1-token label decisions"
        )
    top_logprobs = body.get("top_logprobs", DEFAULT_TOP_LOGPROBS)
    try:
        top_logprobs = int(top_logprobs)
    except (TypeError, ValueError):
        raise ValueError(f"top_logprobs must be an integer (got {top_logprobs!r})") from None
    if top_logprobs < 1:
        raise ValueError(f"top_logprobs must be >= 1 (got {top_logprobs})")
    template_kwargs = body.get("chat_template_kwargs") or {}
    if not isinstance(template_kwargs, dict):
        raise ValueError("chat_template_kwargs must be an object")
    enable_thinking = bool(template_kwargs.get("enable_thinking", False))
    return messages, top_logprobs, enable_thinking


def select_top_k(logits: list[float], k: int) -> list[int]:
    """語彙サイズぶんの生 logits から上位 k 個のトークン id を降順で返す(math のみ、numpy 不要)。

    同点はトークン id 昇順で安定させる(sorted は安定ソートなのでキーに (-logit, id) を使う)。
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    order = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
    return order[: min(k, len(logits))]


def log_softmax_at(logits: list[float], indices: list[int]) -> dict[int, float]:
    """logits 全体に対する log_softmax を計算し、indices 分だけ返す(max を引いてオーバーフロー回避)。"""
    m = max(logits)
    denom = m + math.log(sum(math.exp(x - m) for x in logits))
    return {i: logits[i] - denom for i in indices}


def build_response(
    *,
    model_id: str,
    token_texts: dict[int, str],
    logits: list[float],
    top_k_ids: list[int],
    prompt_tokens: int,
) -> dict[str, Any]:
    """chakuho/core.py query_backend() が読む形へレスポンスを組み立てる。"""
    if not top_k_ids:
        raise ValueError("top_k_ids must be non-empty")
    logprobs = log_softmax_at(logits, top_k_ids)
    top_entries = [{"token": token_texts[i], "logprob": logprobs[i]} for i in top_k_ids]
    best_id = top_k_ids[0]
    return {
        "id": "chakuho-mlx-backend-0",
        "object": "chat.completion",
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": token_texts[best_id]},
                "finish_reason": "length",
                "logprobs": {
                    "content": [
                        {
                            "token": token_texts[best_id],
                            "logprob": logprobs[best_id],
                            "top_logprobs": top_entries,
                        }
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 1, "total_tokens": prompt_tokens + 1},
    }


class StudentLike(Protocol):
    """StudentModel と同じ最小インターフェース(テストでは fake に差し替える)。"""

    model_path: str

    def decide(self, messages: list[dict[str, str]], enable_thinking: bool) -> tuple[list[float], int]: ...

    def token_text(self, token_id: int) -> str: ...


# ---------------------------------------------------------------------------
# mlx への依存はここに閉じ込める
# ---------------------------------------------------------------------------


class StudentModel:
    """mlx_lm.load() したモデル/トークナイザを包み、1 トークン判定に必要な最小 API だけ出す。"""

    def __init__(self, model_path: str) -> None:
        if not _MLX_AVAILABLE:
            raise RuntimeError("mlx_lm is not installed; StudentModel only runs on the serving Mac")
        self.model_path = model_path
        self.model, self.tokenizer = mlx_lm.load(model_path)

    def decide(self, messages: list[dict[str, str]], enable_thinking: bool) -> tuple[list[float], int]:
        """(語彙全体の生 logits, prompt トークン数) を返す。"""
        import mlx.core as mx  # type: ignore

        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        input_ids = mx.array([prompt_ids])
        logits = self.model(input_ids)
        last = logits[0, -1, :]
        return [float(v) for v in last.tolist()], len(prompt_ids)

    def token_text(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])


def make_handler(student: StudentLike) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "chakuho-mlx-backend/0.1"

        def log_message(self, *_args: Any) -> None:  # アクセスログは出さない(chakuho/server.py と同じ)
            return

        def _send(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.split("?")[0] != "/v1/models":
                self._send(404, {"error": "not found"})
                return
            self._send(200, {"data": [{"id": student.model_path}]})

        def do_POST(self) -> None:
            if self.path.split("?")[0] != "/v1/chat/completions":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                messages, top_logprobs, enable_thinking = validate_request(body)
            except (ValueError, json.JSONDecodeError) as exc:
                self._send(400, {"error": str(exc)})
                return
            try:
                logits, prompt_tokens = student.decide(messages, enable_thinking)
                top_k_ids = select_top_k(logits, top_logprobs)
                token_texts = {i: student.token_text(i) for i in top_k_ids}
                response = build_response(
                    model_id=student.model_path,
                    token_texts=token_texts,
                    logits=logits,
                    top_k_ids=top_k_ids,
                    prompt_tokens=prompt_tokens,
                )
            except Exception as exc:  # noqa: BLE001  モデル呼び出し失敗を 500 として返す(サーバを落とさない)
                self._send(500, {"error": str(exc)})
                return
            self._send(200, response)

    return Handler


def serve(model_path: str, host: str = "0.0.0.0", port: int = 8006) -> ThreadingHTTPServer:
    """サーバを生成して返す(serve_forever は呼び出し側)。"""
    student = StudentModel(model_path)
    server = ThreadingHTTPServer((host, port), make_handler(student))
    server.daemon_threads = True
    return server


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="mlx 変換済みモデルのパス(models/current 等)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8006)
    args = ap.parse_args()
    server = serve(args.model, args.host, args.port)
    print(f"chakuho mlx backend listening on {args.host}:{args.port} model={args.model}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
