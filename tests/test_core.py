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


def test_aggregate_counts_normalizes_within_labels_and_reports_coverage():
    # 10 サンプル: A x6, B x2, 宣言ラベル外 x2 -> coverage は宣言ラベルに乗った割合(8/10)
    samples = ["A"] * 6 + ["B"] * 2 + ["<junk>", ""]
    dist, cover = core.aggregate_counts(samples, ["A", "B"])
    assert dist["A"] == pytest.approx(6 / 8)
    assert dist["B"] == pytest.approx(2 / 8)
    assert cover == pytest.approx(0.8)


def test_aggregate_counts_strips_whitespace_before_matching():
    dist, cover = core.aggregate_counts([" A", "A ", "B"], ["A", "B"])
    assert cover == pytest.approx(1.0)
    assert dist["A"] == pytest.approx(2 / 3)


def test_aggregate_counts_case_insensitive_for_word_labels():
    dist, cover = core.aggregate_counts(["Yes", "yes", "NO"], ["yes", "no"], case_insensitive=True)
    assert cover == pytest.approx(1.0)
    assert dist["yes"] == pytest.approx(2 / 3)
    assert dist["no"] == pytest.approx(1 / 3)


def test_aggregate_counts_zero_matches_is_uniform_and_degraded_signal():
    dist, cover = core.aggregate_counts(["<junk>", "", "???"], ["A", "B", "C"])
    assert cover == 0.0
    assert all(v == pytest.approx(1 / 3) for v in dist.values())


def test_aggregate_counts_empty_sample_list_is_uniform_zero_coverage():
    dist, cover = core.aggregate_counts([], ["A", "B"])
    assert cover == 0.0
    assert dist == {"A": 0.5, "B": 0.5}


def test_estimator_mode_defaults_to_sampling(monkeypatch):
    monkeypatch.delenv("CHAKUHO_ESTIMATOR", raising=False)
    assert core.estimator_mode() == core.ESTIMATOR_SAMPLING == "sampling"


def test_estimator_mode_invalid_falls_back_to_sampling(monkeypatch, capsys):
    monkeypatch.setattr(core, "_estimator_warned", False)
    monkeypatch.setenv("CHAKUHO_ESTIMATOR", "bogus")
    assert core.estimator_mode() == core.ESTIMATOR_SAMPLING
    assert "不正" in capsys.readouterr().err
    monkeypatch.setenv("CHAKUHO_ESTIMATOR", "logprobs")
    assert core.estimator_mode() == core.ESTIMATOR_LOGPROBS


def test_sample_count_env_invalid_falls_back_to_default(monkeypatch, capsys):
    monkeypatch.setattr(core, "_samples_warned", False)
    monkeypatch.delenv("CHAKUHO_SAMPLES", raising=False)
    assert core.sample_count() == core.DEFAULT_SAMPLES
    monkeypatch.setenv("CHAKUHO_SAMPLES", "abc")
    assert core.sample_count() == core.DEFAULT_SAMPLES
    assert "不正" in capsys.readouterr().err
    monkeypatch.setenv("CHAKUHO_SAMPLES", "0")
    assert core.sample_count() == core.DEFAULT_SAMPLES
    monkeypatch.setenv("CHAKUHO_SAMPLES", "8")
    assert core.sample_count() == 8


def test_query_backend_sampling_requests_no_logprobs_temperature_one(backend, monkeypatch):
    """sampling リクエストは logprobs/top_logprobs を一切含めず、temperature=1.0・n を指定する
    (SGLang+投機的デコードが logprobs 付きリクエストを 400 で拒否するため)。"""
    monkeypatch.setenv("CHAKUHO_ESTIMATOR", "sampling")
    backend.sample_responder = lambda prompt, menu, n: ["A"] * n
    core.choice("s", "pick", {"a": "A", "b": "B"}, backend_url=backend.url, model="fake-model")
    req = backend.requests[-1]
    assert req["n"] == core.DEFAULT_SAMPLES
    assert req["temperature"] == 1.0
    assert "logprobs" not in req and "top_logprobs" not in req
    assert req["messages"][-1] == {"role": "assistant", "content": "Label:"}
    assert req["continue_final_message"] is True


def test_choice_sampling_counts_labels_into_probabilities(backend, monkeypatch):
    monkeypatch.setenv("CHAKUHO_ESTIMATOR", "sampling")
    monkeypatch.setenv("CHAKUHO_SAMPLES", "10")
    # B x8, A x2 -> y(B) が勝つ
    backend.sample_responder = lambda prompt, menu, n: (["B"] * 8 + ["A"] * 2)
    out = core.choice("s", "pick", {"x": "first", "y": "second"}, backend_url=backend.url)
    assert out["choice"] == "y"
    assert out["stages"] == 1
    assert out["coverage"] == pytest.approx(1.0)
    assert out["probabilities"]["y"] == pytest.approx(0.8)
    assert out["probabilities"]["x"] == pytest.approx(0.2)
    assert "degraded" not in out


def test_choice_sampling_degraded_when_no_sample_hits_label(backend, monkeypatch):
    monkeypatch.setenv("CHAKUHO_ESTIMATOR", "sampling")
    backend.sample_responder = lambda prompt, menu, n: [""] * n
    out = core.choice("s", "pick", ["x", "y"], backend_url=backend.url)
    assert out["degraded"] is True
    assert out["coverage"] == 0.0
    assert out["probabilities"] == {"x": 0.5, "y": 0.5}


def test_noul_sampling(backend, monkeypatch):
    monkeypatch.setenv("CHAKUHO_ESTIMATOR", "sampling")
    monkeypatch.setenv("CHAKUHO_SAMPLES", "10")
    backend.sample_responder = lambda prompt, menu, n: (["yes"] * 7 + ["no"] * 3)
    out = core.noul("s", "is it?", backend_url=backend.url)
    assert out["noul"] == pytest.approx(0.7)
    assert out["coverage"] == pytest.approx(1.0)


def test_score_sampling(backend, monkeypatch):
    monkeypatch.setenv("CHAKUHO_ESTIMATOR", "sampling")
    monkeypatch.setenv("CHAKUHO_SAMPLES", "10")
    # 3 段階 low=A mid=B high=C: A x2, B x3, C x5 -> 期待値 (2*0+3*0.5+5*1)/10
    backend.sample_responder = lambda prompt, menu, n: (["A"] * 2 + ["B"] * 3 + ["C"] * 5)
    out = core.score("s", "how bad", ["low", "mid", "high"], backend_url=backend.url)
    assert out["score"] == pytest.approx((3 * 0.5 + 5 * 1.0) / 10)
    assert out["probabilities"]["high"] == pytest.approx(0.5)


def test_strip_sample_text_removes_leading_cue_when_prefill_enabled(monkeypatch):
    monkeypatch.delenv("CHAKUHO_PREFILL", raising=False)  # prefill 既定 有効
    assert core._strip_sample_text("Label: A") == "A"
    assert core._strip_sample_text(" A") == "A"  # 実バックエンド(SGLang)は差分のみを返す(prefix なし)
    assert core._strip_sample_text("") == ""
    assert core._strip_sample_text(None) == ""


def test_strip_sample_text_leaves_content_untouched_when_prefill_disabled(monkeypatch):
    monkeypatch.setenv("CHAKUHO_PREFILL", "0")
    assert core._strip_sample_text(" A") == "A"
    assert core._strip_sample_text("Label: A") == "Label: A"  # プレフィックス無効時は剥がさない


def test_choice_sampling_without_prefill(backend, monkeypatch):
    monkeypatch.setenv("CHAKUHO_ESTIMATOR", "sampling")
    monkeypatch.setenv("CHAKUHO_PREFILL", "0")
    backend.sample_responder = lambda prompt, menu, n: ["A"] * n
    out = core.choice("s", "pick", {"a": "A", "b": "B"}, backend_url=backend.url, model="fake-model")
    req = backend.requests[-1]
    assert "continue_final_message" not in req
    assert req["messages"][-1]["role"] == "user" and req["messages"][-1]["content"].endswith("Label:")
    assert out["choice"] == "a"
    assert out["coverage"] == pytest.approx(1.0)


def test_tournament_sampling(backend, monkeypatch):
    """120 択のトーナメントが sampling 経路でも 2 段で決まり、全リクエストが sampling 形(n あり)。"""
    monkeypatch.setenv("CHAKUHO_ESTIMATOR", "sampling")
    monkeypatch.setenv("CHAKUHO_SAMPLES", "6")
    options = [f"opt_{i:03d}" for i in range(120)]

    def sample_responder(prompt, menu, n):
        labels = list(menu)
        win = labels[-1]  # 各チャンク・決勝とも末尾ラベルが勝つ
        return [win] * n

    backend.sample_responder = sample_responder
    out = core.choice("s", "pick", options, backend_url=backend.url)
    assert out["stages"] == 2
    assert len(backend.requests) == 4  # 3 チャンク + 決勝
    for req in backend.requests:
        assert "n" in req and "logprobs" not in req
    assert out["coverage"] == pytest.approx(1.0)
    assert sum(1 for v in out["probabilities"].values() if v > 0) >= 1


def test_answer_cue_is_assistant_prefill_by_default(backend, monkeypatch):
    from chakuho import core
    monkeypatch.delenv("CHAKUHO_PREFILL", raising=False)
    core.choice("s", "pick", {"a": "A", "b": "B"}, backend_url=backend.url, model="fake-model")
    req = backend.requests[-1]
    assert req["messages"][-1] == {"role": "assistant", "content": "Label:"}
    assert req["continue_final_message"] is True and req["add_generation_prompt"] is False
    assert not req["messages"][1]["content"].rstrip().endswith("Label:")
    monkeypatch.setenv("CHAKUHO_PREFILL", "0")
    core.choice("s", "pick", {"a": "A", "b": "B"}, backend_url=backend.url, model="fake-model")
    req = backend.requests[-1]
    assert req["messages"][-1]["role"] == "user" and req["messages"][-1]["content"].endswith("Label:")
    assert "continue_final_message" not in req
