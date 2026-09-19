"""chakuho HTTP サーバ。Jev 互換の POST /v1/systemone と GET /health を提供する(stdlib のみ)。

判定ログは CHAKUHO_LOG_DIR/decisions-YYYYMMDD.jsonl に 1 リクエスト 1 行で追記し、
CHAKUHO_LOG_KEEP_DAYS より古い日付のファイルは起動時と日付切替時に削除する。
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from chakuho import core

DEFAULT_LOG_DIR = Path.home() / ".ato" / "chakuho"
DEFAULT_KEEP_DAYS = 90
_LOG_NAME_RE = re.compile(r"^decisions-(\d{8})\.jsonl$")


class DecisionLog:
    """日付別 JSONL への追記と、保持日数を超えたファイルの削除。"""

    def __init__(self, log_dir: Path, keep_days: int) -> None:
        self.log_dir = log_dir
        self.keep_days = keep_days
        self._lock = threading.Lock()
        self._current_day: str | None = None
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.cleanup()

    def _today(self) -> str:
        return dt.date.today().strftime("%Y%m%d")

    def cleanup(self, today: dt.date | None = None) -> list[Path]:
        """keep_days より古い decisions-*.jsonl を削除し、削除したパスを返す。"""
        base = today or dt.date.today()
        cutoff = base - dt.timedelta(days=self.keep_days)
        removed: list[Path] = []
        for path in self.log_dir.glob("decisions-*.jsonl"):
            match = _LOG_NAME_RE.match(path.name)
            if not match:
                continue
            try:
                day = dt.datetime.strptime(match.group(1), "%Y%m%d").date()
            except ValueError:
                continue
            if day < cutoff:
                try:
                    path.unlink()
                    removed.append(path)
                except OSError:
                    continue
        return removed

    def append(self, record: dict[str, Any]) -> None:
        day = self._today()
        with self._lock:
            if day != self._current_day:
                self._current_day = day
                self.cleanup()
            path = self.log_dir / f"decisions-{day}.jsonl"
            try:
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError:
                pass  # ログ書き込み失敗で判定を落とさない


def _config_from_env() -> dict[str, Any]:
    return {
        "backend_url": os.environ.get("CHAKUHO_BACKEND_URL", core.DEFAULT_BACKEND_URL),
        "fallback_backend_url": os.environ.get("CHAKUHO_FALLBACK_BACKEND_URL") or None,
        "log_dir": Path(os.environ.get("CHAKUHO_LOG_DIR", str(DEFAULT_LOG_DIR))),
        "keep_days": int(os.environ.get("CHAKUHO_LOG_KEEP_DAYS", str(DEFAULT_KEEP_DAYS))),
    }


def make_handler(backend_url: str, log: DecisionLog | None,
                 fallback_backend_url: str | None = None) -> type[BaseHTTPRequestHandler]:
    """主 backend が BackendError を返した時だけ fallback_backend_url で同じ判定をやり直す。
    応答には "backend": "primary" | "fallback" を付ける(読む側が縮退を区別できるように)。"""
    class Handler(BaseHTTPRequestHandler):
        server_version = "chakuho/0.1"

        def log_message(self, *_args: Any) -> None:  # 標準の 1 行アクセスログは出さない
            return

        def _send(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.split("?")[0] != "/health":
                self._send(404, {"error": "not found"})
                return
            payload: dict[str, Any] = {"backend": backend_url, "fallback_backend": fallback_backend_url}
            if fallback_backend_url:
                try:
                    payload["fallback_model"] = core.resolve_model(fallback_backend_url, timeout=10)
                except core.BackendError as exc:
                    payload["fallback_error"] = str(exc)
            try:
                payload["model"] = core.resolve_model(backend_url, timeout=10)
            except core.BackendError as exc:
                payload["error"] = str(exc)
                if "fallback_model" in payload:  # 主が死んでいても fallback で判定できるなら稼働扱い
                    self._send(200, {"ok": True, "degraded": True, **payload})
                    return
                self._send(503, {"ok": False, **payload})
                return
            self._send(200, {"ok": True, **payload})

        def do_POST(self) -> None:
            if self.path.split("?")[0] != "/v1/systemone":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(req, dict) or "state" not in req or not isinstance(req.get("questions"), dict):
                    raise ValueError("request must be an object with 'state' and 'questions' (object)")
            except (ValueError, json.JSONDecodeError) as exc:
                self._send(400, {"error": f"bad request: {exc}"})
                return
            started = time.time()
            try:
                out = core.evaluate(req["state"], req["questions"], backend_url=backend_url)
                out["backend"] = "primary"
                code = 200
            except ValueError as exc:
                out, code = {"error": f"bad request: {exc}"}, 400
            except core.BackendError as exc:
                out, code = {"error": f"backend unavailable: {exc}"}, 503
                if fallback_backend_url:
                    try:
                        out = core.evaluate(req["state"], req["questions"], backend_url=fallback_backend_url)
                        out["backend"] = "fallback"
                        out["primary_error"] = str(exc)
                        code = 200
                    except core.BackendError as exc2:
                        out = {"error": f"backend unavailable: {exc}; fallback unavailable: {exc2}"}
            if log is not None:  # 応答前に書く(読み手が応答直後にログを見ても揃っている)
                log.append({"t": started, "status": code, "request": req, "response": out})
            self._send(code, out)

    return Handler


def serve(host: str = "0.0.0.0", port: int = 9750, *, backend_url: str | None = None,
          fallback_backend_url: str | None = None,
          log_dir: Path | None = None, keep_days: int | None = None) -> ThreadingHTTPServer:
    """サーバを生成して返す(serve_forever は呼び出し側)。"""
    cfg = _config_from_env()
    backend = backend_url or cfg["backend_url"]
    fallback = fallback_backend_url or cfg["fallback_backend_url"]
    log = DecisionLog(log_dir or cfg["log_dir"], keep_days if keep_days is not None else cfg["keep_days"])
    core.configure_inflight()
    server = ThreadingHTTPServer((host, port), make_handler(backend, log, fallback))
    server.daemon_threads = True
    return server
