"""fake backend: OpenAI 互換の /models と /chat/completions を返す。実 vLLM は叩かない。"""

from __future__ import annotations

import json
import math
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from chakuho import core

_MENU_LINE = re.compile(r"^([A-Za-z]|yes|no): (.*?)(?: — .*)?$")


def parse_menu(prompt: str) -> dict[str, str]:
    """prompt の OPTIONS 節から {label: option} を取り出す。"""
    section = prompt.split("OPTIONS:\n", 1)[1].split("\n\nSTATE:", 1)[0]
    out: dict[str, str] = {}
    for line in section.splitlines():
        m = _MENU_LINE.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def logprobs_for(winner_label: str, labels: list[str], p_win: float = 0.9, noise: float = 0.0) -> dict[str, float]:
    """winner に p_win、残りへ均等に (1-p_win-noise)、noise 分はラベル外トークンへ。"""
    rest = [l for l in labels if l != winner_label]
    out = {winner_label: math.log(p_win)}
    if rest:
        share = (1.0 - p_win - noise) / len(rest)
        for l in rest:
            out[l] = math.log(max(share, 1e-12))
    if noise > 0:
        out["<junk>"] = math.log(noise)
    return out


class FakeBackend:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.prompts: list[str] = []
        self.models_ok = True
        # responder(prompt, menu) -> top_logprobs dict。既定は先頭ラベル勝ち
        self.responder = lambda prompt, menu: logprobs_for(next(iter(menu)), list(menu))
        self.chat_error: int | None = None  # 例: 500 を返させる
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                return

            def _send(self, code, payload):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.endswith("/models") and outer.models_ok:
                    self._send(200, {"data": [{"id": "fake-model"}]})
                else:
                    self._send(500, {"error": "models unavailable"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n))
                outer.requests.append(req)
                prompt = req["messages"][-1]["content"]
                outer.prompts.append(prompt)
                if outer.chat_error:
                    self._send(outer.chat_error, {"error": "boom"})
                    return
                menu = parse_menu(prompt)
                top = outer.responder(prompt, menu)
                self._send(200, {
                    "choices": [{"logprobs": {"content": [{"token": max(top, key=top.get), "top_logprobs": [
                        {"token": t, "logprob": lp} for t, lp in top.items()]}]}}],
                    "usage": {"prompt_tokens": len(prompt) // 4},
                })

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def backend(monkeypatch):
    fb = FakeBackend()
    core.clear_model_cache()
    monkeypatch.delenv("CHAKUHO_MODEL", raising=False)
    monkeypatch.delenv("CHAKUHO_MAX_INFLIGHT", raising=False)
    core.configure_inflight(16)
    yield fb
    fb.close()
    core.clear_model_cache()
