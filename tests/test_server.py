from __future__ import annotations

import datetime as dt
import json
import threading
from urllib import error as urllib_error
from urllib import request as urllib_request

import pytest

from chakuho import server
from chakuho.client import ChakuhoError, decide


@pytest.fixture
def chakuho_server(backend, tmp_path):
    httpd = server.serve("127.0.0.1", 0, backend_url=backend.url, log_dir=tmp_path / "log", keep_days=14)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, tmp_path / "log"
    httpd.shutdown()
    httpd.server_close()


def _post(url, payload):
    req = urllib_request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib_request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib_error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_systemone_jev_shape_and_log(chakuho_server):
    base, log_dir = chakuho_server
    code, out = _post(base + "/v1/systemone", {"state": {"x": 1}, "questions": {
        "t": {"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": "B"}},
        "d": {"type": "noul", "instructions": "i"},
        "r": {"type": "score", "instructions": "i", "criteria": ["lo", "hi"]}}})
    assert code == 200
    assert out["answers"]["t"]["choice"] == "a"
    assert "noul" in out["answers"]["d"] and "score" in out["answers"]["r"]
    assert out["usage"]["input_tokens"] > 0 and "latency_ms" in out and out["model"] == "fake-model"
    files = list(log_dir.glob("decisions-*.jsonl"))
    assert len(files) == 1
    rec = json.loads(files[0].read_text().splitlines()[-1])
    assert rec["status"] == 200 and rec["request"]["state"] == {"x": 1}


def test_bad_request_400(chakuho_server):
    base, _ = chakuho_server
    assert _post(base + "/v1/systemone", {"questions": {}})[0] == 400
    assert _post(base + "/v1/systemone", {"state": "s", "questions": {"q": {"type": "nope"}}})[0] == 400
    code, out = _post(base + "/v1/systemone", {"state": "s", "questions": {
        "q": {"type": "choice", "criteria": [str(i) for i in range(2705)]}}})
    assert code == 400 and "error" in out


def test_backend_down_503_and_health(chakuho_server, backend):
    base, _ = chakuho_server
    backend.chat_error = 500
    code, out = _post(base + "/v1/systemone", {"state": "s", "questions": {"q": {"type": "noul"}}})
    assert code == 503 and "error" in out
    backend.models_ok = False
    from chakuho import core
    core.clear_model_cache()
    try:
        urllib_request.urlopen(base + "/health", timeout=10)
        assert False, "expected 503"
    except urllib_error.HTTPError as e:
        assert e.code == 503
        assert json.loads(e.read())["ok"] is False


def test_health_reflects_backend_without_cache_clear(chakuho_server, backend):
    """モデル ID がキャッシュ済みでも、/health は backend を実際に叩いて生死を返す(裏の vLLM 停止中に ok:true を返した実害の再発防止)。"""
    base, _ = chakuho_server
    with urllib_request.urlopen(base + "/health", timeout=10) as r:
        assert json.loads(r.read())["ok"] is True  # ここでモデル ID がキャッシュされる
    backend.models_ok = False
    try:
        urllib_request.urlopen(base + "/health", timeout=10)
        assert False, "expected 503"
    except urllib_error.HTTPError as e:
        assert e.code == 503 and json.loads(e.read())["ok"] is False
    backend.models_ok = True
    with urllib_request.urlopen(base + "/health", timeout=10) as r:
        assert json.loads(r.read())["ok"] is True


def test_health_ok(chakuho_server):
    base, _ = chakuho_server
    with urllib_request.urlopen(base + "/health", timeout=10) as r:
        d = json.loads(r.read())
    assert d["ok"] is True and d["model"] == "fake-model"


def test_client_decide_and_error(chakuho_server):
    base, _ = chakuho_server
    out = decide("s", {"q": {"type": "noul", "instructions": "i"}}, url=base + "/v1/systemone")
    assert "noul" in out["answers"]["q"]
    with pytest.raises(ChakuhoError):
        decide("s", {"q": {"type": "noul"}}, url="http://127.0.0.1:9/v1/systemone", timeout=2)


def test_log_rotation_removes_old_files(tmp_path):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    old = log_dir / "decisions-20200101.jsonl"
    old.write_text("{}\n")
    today = dt.date.today().strftime("%Y%m%d")
    fresh = log_dir / f"decisions-{today}.jsonl"
    fresh.write_text("{}\n")
    log = server.DecisionLog(log_dir, keep_days=14)  # __init__ で cleanup
    assert not old.exists() and fresh.exists()
    log.append({"a": 1})
    assert len(fresh.read_text().splitlines()) == 2


def test_fallback_backend_used_when_primary_down(backend, tmp_path):
    """主 backend が不通でも fallback があれば 200 で答え、backend: fallback が付く。"""
    httpd = server.serve("127.0.0.1", 0, backend_url="http://127.0.0.1:9/v1",
                         fallback_backend_url=backend.url, log_dir=tmp_path / "log", keep_days=14)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        code, out = _post(base + "/v1/systemone", {"state": "s", "questions": {
            "t": {"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": "B"}}}})
        assert code == 200 and out["backend"] == "fallback" and "primary_error" in out
        assert out["answers"]["t"]["choice"] == "a"
        with urllib_request.urlopen(base + "/health", timeout=10) as r:
            health = json.loads(r.read())
        assert health["ok"] is True and health["degraded"] is True and health["fallback_model"] == "fake-model"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_primary_answer_is_marked_primary(chakuho_server):
    base, _ = chakuho_server
    code, out = _post(base + "/v1/systemone", {"state": "s", "questions": {
        "t": {"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": "B"}}}})
    assert code == 200 and out["backend"] == "primary"
