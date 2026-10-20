"""待ち受け既定(127.0.0.1)・LAN 公開時の警告・リクエスト本体の上限の試験。モデルは読み込まない。"""

from __future__ import annotations

import http.client
import inspect
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from chakuho import cli, netsafe, server
from distill import mlx_backend

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_are_loopback():
    assert netsafe.DEFAULT_HOST == "127.0.0.1"
    assert inspect.signature(server.serve).parameters["host"].default == "127.0.0.1"
    assert inspect.signature(mlx_backend.serve).parameters["host"].default == "127.0.0.1"


def test_resolve_host(monkeypatch):
    monkeypatch.delenv("CHAKUHO_HOST", raising=False)
    assert netsafe.resolve_host(None, "CHAKUHO_HOST") == "127.0.0.1"
    monkeypatch.setenv("CHAKUHO_HOST", "0.0.0.0")
    assert netsafe.resolve_host(None, "CHAKUHO_HOST") == "0.0.0.0"
    assert netsafe.resolve_host("127.0.0.2", "CHAKUHO_HOST") == "127.0.0.2"


@pytest.mark.parametrize("host,loop", [("127.0.0.1", True), ("localhost", True), ("::1", True),
                                       ("[::1]", True), ("0.0.0.0", False), ("::", False),
                                       ("192.0.2.1", False), ("example.invalid", False)])
def test_is_loopback(host, loop):
    assert netsafe.is_loopback(host) is loop


def test_warn_only_when_exposed(capsys):
    assert netsafe.warn_if_exposed("127.0.0.1", 1, "x") is False
    assert capsys.readouterr().err == ""
    assert netsafe.warn_if_exposed("0.0.0.0", 1, "x") is True
    assert "WARNING" in capsys.readouterr().err


def test_cli_serve_binds_loopback_by_default(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("CHAKUHO_HOST", raising=False)
    monkeypatch.setenv("CHAKUHO_LOG_DIR", str(tmp_path))
    seen = {}

    def fake_serve(host, port):
        seen["host"] = host
        raise SystemExit(0)

    monkeypatch.setattr(server, "serve", fake_serve)
    with pytest.raises(SystemExit):
        cli.main(["serve", "--port", "0"])
    assert seen["host"] == "127.0.0.1"
    assert "WARNING" not in capsys.readouterr().err

    with pytest.raises(SystemExit):
        cli.main(["serve", "--host", "0.0.0.0", "--port", "0"])
    assert seen["host"] == "0.0.0.0"
    assert "WARNING" in capsys.readouterr().err


def test_chakuho_serve_process_listens_on_loopback(tmp_path):
    """実プロセスで起動し、待ち受けアドレスが 127.0.0.1 であることを確かめる。"""
    env = {"CHAKUHO_LOG_DIR": str(tmp_path), "CHAKUHO_BACKEND_URL": "http://127.0.0.1:9/v1",
           "PATH": "/usr/bin:/bin"}
    proc = subprocess.Popen([sys.executable, "-c", "import sys; from chakuho.cli import main; sys.exit(main())",
                             "serve", "--port", "0"], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    try:
        line = proc.stdout.readline()
        assert "listening on 127.0.0.1:0" in line, line
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    assert "WARNING" not in proc.stderr.read()


class _Fake:
    model_path = "fake"

    def decide(self, messages, enable_thinking):
        return [0.0, 1.0], 3

    def token_text(self, token_id):
        return "AB"[token_id]


@pytest.fixture
def servers(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAKUHO_MAX_BODY_BYTES", "1024")
    chak = server.serve(port=0, backend_url="http://127.0.0.1:9/v1", log_dir=tmp_path)
    mlx = mlx_backend.serve("fake", port=0, student=_Fake())
    for s in (chak, mlx):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    yield {"chakuho": (chak, "/v1/systemone"), "mlx": (mlx, "/v1/chat/completions")}
    for s in (chak, mlx):
        s.shutdown()
        s.server_close()


def _raw_post(port: int, path: str, body: bytes, length: str | None) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.putrequest("POST", path)
    if length is not None:
        conn.putheader("Content-Length", length)
    conn.endheaders()
    if body:
        conn.send(body)
    resp = conn.getresponse()
    out = resp.status, json.loads(resp.read())
    conn.close()
    return out


@pytest.mark.parametrize("name", ["chakuho", "mlx"])
def test_server_binds_loopback_and_limits_body(servers, name):
    httpd, path = servers[name]
    host, port = httpd.server_address[:2]
    assert host == "127.0.0.1"
    code, data = _raw_post(port, path, b"x" * 2048, "2048")
    assert code == 413 and "too large" in data["error"]
    assert _raw_post(port, path, b"", "-1")[0] == 400
    assert _raw_post(port, path, b"", None)[0] == 400
    assert _raw_post(port, path, b"[]", "2")[0] == 400


def test_mlx_backend_ok_through_serve(servers):
    httpd, path = servers["mlx"]
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}).encode()
    code, data = _raw_post(httpd.server_address[1], path, body, str(len(body)))
    assert code == 200 and data["choices"][0]["message"]["content"] == "B"


def _post_json(port: int, path: str, payload) -> tuple[int, dict]:
    body = json.dumps(payload).encode()
    return _raw_post(port, path, body, str(len(body)))


def test_chakuho_rejects_too_many_and_malformed_questions(backend, tmp_path, monkeypatch):
    monkeypatch.delenv("CHAKUHO_MAX_BODY_BYTES", raising=False)
    httpd = server.serve(port=0, backend_url=backend.url, log_dir=tmp_path)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port, path = httpd.server_address[1], "/v1/systemone"
    try:
        many = {f"q{i}": {"type": "noul"} for i in range(server.DEFAULT_MAX_QUESTIONS + 1)}
        code, data = _post_json(port, path, {"state": 1, "questions": many})
        assert code == 400 and "too many questions" in data["error"]
        code, data = _post_json(port, path, {"state": 1, "questions": {"a": 5}})
        assert code == 400 and "must be an object" in data["error"]
        # criteria 欠落は以前 TypeError で接続ごと落ちていた
        code, data = _post_json(port, path, {"state": {"secret": "x" * 100}, "questions": {"a": {"type": "choice"}}})
        assert code == 400, data
        logged = (tmp_path / next(p.name for p in tmp_path.glob("decisions-*.jsonl"))).read_text()
        assert "xxxxxxxxxx" not in logged  # 400 は本文(state)をログに残さない
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_handlers_have_socket_timeout():
    assert server.make_handler("http://127.0.0.1:9/v1", None).timeout == server.SOCKET_TIMEOUT_S
    assert mlx_backend.make_handler(_Fake()).timeout == 30


def test_mlx_top_logprobs_capped():
    with pytest.raises(ValueError):
        mlx_backend.validate_request({"messages": [{"role": "user", "content": "x"}],
                                      "top_logprobs": mlx_backend.MAX_TOP_LOGPROBS + 1})


def test_core_pool_bounded():
    from chakuho import core
    assert core.MAX_POOL_WORKERS <= 32
