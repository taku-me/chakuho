from __future__ import annotations

import math

import pytest

from chakuho import core
from tests.conftest import logprobs_for, parse_menu


def test_labels_beyond_26_use_lowercase():
    labels = core.labels_for(30)
    assert labels[:26] == [chr(c) for c in range(65, 91)]
    assert labels[26:] == ["a", "b", "c", "d"]
    with pytest.raises(ValueError):
        core.labels_for(53)


def test_aggregate_normalizes_within_labels_and_reports_coverage():
    top = {"A": math.log(0.6), "B": math.log(0.2), " A": math.log(0.1), "<junk>": math.log(0.1)}
    dist, cover = core.aggregate(top, ["A", "B"])
    assert dist["A"] == pytest.approx(0.7 / 0.9)
    assert dist["B"] == pytest.approx(0.2 / 0.9)
    assert cover == pytest.approx(0.9)


def test_aggregate_zero_coverage_is_uniform():
    dist, cover = core.aggregate({"<junk>": 0.0}, ["A", "B", "C"])
    assert cover == 0.0
    assert all(v == pytest.approx(1 / 3) for v in dist.values())


def test_choice_single_round(backend):
    backend.responder = lambda prompt, menu: logprobs_for("B", list(menu), p_win=0.8, noise=0.05)
    out = core.choice("s", "pick", {"x": "first", "y": "second", "z": "third"}, backend_url=backend.url)
    assert out["choice"] == "y"
    assert out["stages"] == 1
    assert out["coverage"] == pytest.approx(0.95)
    assert out["probabilities"]["y"] == pytest.approx(0.8 / 0.95)
    assert "degraded" not in out
    # 説明文がメニューに載る
    assert "B: y — second" in backend.prompts[-1]


def test_choice_degraded_when_no_label_mass(backend):
    backend.responder = lambda prompt, menu: {"<junk>": 0.0}
    out = core.choice("s", "pick", ["x", "y"], backend_url=backend.url)
    assert out["degraded"] is True
    assert out["coverage"] == 0.0
    assert out["probabilities"] == {"x": 0.5, "y": 0.5}


def test_noul_p_yes(backend):
    backend.responder = lambda prompt, menu: {"Yes": math.log(0.7), "no": math.log(0.3)}
    out = core.noul("s", "is it?", {"true": "yes it is", "false": "nope"}, backend_url=backend.url)
    assert out["noul"] == pytest.approx(0.7)
    assert out["coverage"] == pytest.approx(1.0)


def test_score_normalized_expected_value(backend):
    # 3 段階: A=0.0, B=0.5, C=1.0 の期待値
    backend.responder = lambda prompt, menu: {"A": math.log(0.2), "B": math.log(0.3), "C": math.log(0.5)}
    out = core.score("s", "how bad", ["low", "mid", "high"], backend_url=backend.url)
    assert out["score"] == pytest.approx(0.3 * 0.5 + 0.5 * 1.0)
    assert out["probabilities"]["high"] == pytest.approx(0.5)


def test_evaluate_parallel_and_jev_shape(backend):
    backend.responder = lambda prompt, menu: logprobs_for(next(iter(menu)), list(menu))
    out = core.evaluate("state", {
        "a": {"type": "choice", "instructions": "i", "criteria": ["p", "q"]},
        "b": {"type": "noul", "instructions": "i"},
        "c": {"type": "score", "instructions": "i", "criteria": ["1", "2"]},
    }, backend_url=backend.url)
    assert set(out["answers"]) == {"a", "b", "c"}
    assert out["answers"]["a"]["choice"] == "p"
    assert "noul" in out["answers"]["b"]
    assert "score" in out["answers"]["c"]
    assert out["usage"]["input_tokens"] > 0
    assert out["model"] == "fake-model"
    assert isinstance(out["latency_ms"], int)
    assert len(backend.requests) == 3


def test_tournament_top_k_and_zero_for_eliminated(backend):
    options = [f"opt_{i:03d}" for i in range(120)]  # 3 chunks (52, 52, 16) -> k = 52 // 3 = 17
    # 各チャンクでは末尾の option(opt_051 / opt_103 / opt_119)を勝たせ、
    # 決勝(別チャンク出身の opt_051 と opt_103 が同居)では opt_103 を勝たせる
    def responder(prompt, menu):
        labels = list(menu)
        if "opt_103" in menu.values() and "opt_051" in menu.values():
            win = next(l for l, o in menu.items() if o == "opt_103")
        else:
            win = labels[-1]
        return logprobs_for(win, labels, p_win=0.6)
    backend.responder = responder
    out = core.choice("s", "pick", options, backend_url=backend.url)
    assert out["stages"] == 2
    assert out["choice"] == "opt_103"
    assert out["probabilities"]["opt_100"] == 0.0  # チャンク 2 の上位 17 に入らず予選落ち
    # 4 リクエスト = 3 チャンク + 決勝。決勝には各チャンク上位 17 個 = 51 (最後のチャンクは 16 個) -> 17+17+16=50
    assert len(backend.requests) == 4
    final_menu = parse_menu(backend.prompts[-1])
    assert len(final_menu) == 17 + 17 + 16
    assert set(final_menu.values()) <= set(options)
    # 予選落ちは 0.0、決勝進出者は分布を持つ
    assert out["probabilities"]["opt_020"] == 0.0  # チャンク 1 の上位 17(opt_051, opt_000..015)に入らず予選落ち
    assert sum(1 for v in out["probabilities"].values() if v > 0) == len(final_menu)


def test_tournament_none_option_in_every_chunk_and_final(backend):
    options = [f"opt_{i:03d}" for i in range(120)] + [core.NONE_OPTION]
    backend.responder = lambda prompt, menu: logprobs_for(list(menu)[0], list(menu))
    out = core.choice("s", "pick", options, backend_url=backend.url)
    assert out["stages"] == 2
    for prompt in backend.prompts:
        assert core.NONE_OPTION in parse_menu(prompt).values()
    assert core.NONE_OPTION in out["probabilities"]


def test_too_many_options_raises(backend):
    with pytest.raises(ValueError):
        core.choice("s", "pick", [str(i) for i in range(2705)], backend_url=backend.url)


def test_backend_down_raises_backend_error(backend):
    with pytest.raises(core.BackendError):
        core.choice("s", "pick", ["a", "b"], backend_url="http://127.0.0.1:9/v1", timeout=2)


def test_inflight_limit_one_still_completes(backend, monkeypatch):
    core.configure_inflight(1)
    out = core.evaluate("s", {f"q{i}": {"type": "noul", "instructions": "i"} for i in range(4)}, backend_url=backend.url)
    assert len(out["answers"]) == 4
    with pytest.raises(ValueError):
        core.configure_inflight(0)


def test_boolean_alias_matches_noul(backend):
    backend.responder = lambda prompt, menu: {"yes": math.log(0.6), "no": math.log(0.4)}
    out = core.evaluate("s", {"q": {"type": "boolean", "instructions": "i"}}, backend_url=backend.url)
    a = out["answers"]["q"]
    assert a["noul"] == pytest.approx(0.6) and a["probability"] == pytest.approx(0.6)


def test_top_logprobs_env_invalid_falls_back_to_default(monkeypatch, capsys):
    from chakuho import core
    monkeypatch.setattr(core, "_top_logprobs_warned", False)
    monkeypatch.setenv("CHAKUHO_TOP_LOGPROBS", "abc")
    assert core.top_logprobs_limit() == core.DEFAULT_TOP_LOGPROBS
    monkeypatch.setenv("CHAKUHO_TOP_LOGPROBS", "0")
    assert core.top_logprobs_limit() == core.DEFAULT_TOP_LOGPROBS
    monkeypatch.setenv("CHAKUHO_TOP_LOGPROBS", "11")
    assert core.top_logprobs_limit() == 11
    monkeypatch.delenv("CHAKUHO_TOP_LOGPROBS")
    assert core.top_logprobs_limit() == core.DEFAULT_TOP_LOGPROBS
    assert "不正" in capsys.readouterr().err


def test_images_in_state_become_image_parts(backend):
    """state.images の URL は user content の image_url パートになり、prompt テキストには混ざらない。"""
    from chakuho import core
    state = {"app": "x", "images": ["data:image/png;base64,AAAA", "https://example.invalid/b.png"]}
    r = core.choice(state, "pick", {"a": "A", "b": "B"}, backend_url=backend.url, model="fake-model")
    assert r["choice"] == "a"
    req = backend.requests[-1]
    content = req["messages"][1]["content"]
    assert isinstance(content, list) and [c["type"] for c in content] == ["image_url", "image_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/png")
    assert "images" not in content[2]["text"] and "app: x" in content[2]["text"]
    n = core.noul(state, "ok?", backend_url=backend.url, model="fake-model")
    assert isinstance(backend.requests[-1]["messages"][1]["content"], list)
    plain = core.choice({"app": "x"}, "pick", {"a": "A", "b": "B"}, backend_url=backend.url, model="fake-model")
    assert isinstance(backend.requests[-1]["messages"][1]["content"], str)
