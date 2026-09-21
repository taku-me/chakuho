"""distill/ 配下(eval.py / mlx_backend.py / train.py)のテスト。

このリポジトリの開発機には torch/mlx が無いため、ここでは:
- eval.py: 全体を実際に動かす(stdlib のみなので依存が要らない)
- mlx_backend.py: mlx 呼び出し(StudentModel)を避け、同じインターフェースの fake で
  純粋関数(select_top_k/log_softmax_at/build_response/validate_request)とハンドラを検証する
- train.py: torch 抜きでテストできる純粋関数(label_token_ids/kl_from_teacher_dict)だけを検証し、
  torch が要る関数は torch 不在時に RuntimeError を返すことだけ確認する
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error as urllib_error
from urllib import request as urllib_request

import pytest

from chakuho import core
from distill import eval as distill_eval
from distill import mlx_backend
from distill import train as distill_train

NONE = core.NONE_OPTION


def make_case(case_id: str, *, app: str = "TestApp", variant: str = "orig", criteria: dict[str, str], expected: list[str]) -> dict:
    return {
        "case_id": case_id,
        "app": app,
        "variant": variant,
        "instruction": "do something",
        "expected": expected,
        "request": {
            "state": {"app": app, "window": app},
            "questions": {
                "target": {
                    "type": "choice",
                    "instructions": "pick one element",
                    "criteria": criteria,
                }
            },
        },
    }


# ---------------------------------------------------------------------------
# distill/eval.py: normalize_choice / summarize
# ---------------------------------------------------------------------------


def test_normalize_choice_strips_id_and_passes_through_none():
    assert distill_eval.normalize_choice("12: [AXButton] Settings") == "AXButton:Settings"
    assert distill_eval.normalize_choice(NONE) == NONE
    assert distill_eval.normalize_choice("weird-format") == "weird-format"


def test_summarize_buckets_by_variant_app_and_none_tasks():
    cases = [
        make_case("0-orig", variant="orig", criteria={"1: [AXButton] Save": "w", NONE: "n"}, expected=["AXButton:Save"]),
        make_case("0-shuffled", variant="shuffled", criteria={"1: [AXButton] Save": "w", NONE: "n"}, expected=["AXButton:Save"]),
        make_case("1-orig", app="Other", variant="orig", criteria={"2: [AXButton] X": "w", NONE: "n"}, expected=[NONE]),
    ]
    answers = {
        "0-orig": {"choice": "1: [AXButton] Save", "coverage": 0.9},
        "0-shuffled": {"choice": "2: [AXButton] X", "coverage": 0.9},  # wrong element -> miss
        "1-orig": {"choice": NONE, "coverage": 0.2},  # low coverage but still correct
    }
    summary = distill_eval.summarize(cases, answers)
    assert summary["n_cases"] == 3
    assert summary["accuracy"]["all"] == {"ok": 2, "total": 3, "pct": pytest.approx(66.7, abs=0.1)}
    assert summary["accuracy"]["variant:orig"]["ok"] == 2
    assert summary["accuracy"]["variant:shuffled"]["ok"] == 0
    assert summary["accuracy"]["app:Other"]["ok"] == 1
    assert summary["accuracy"]["none-tasks"] == {"ok": 1, "total": 1, "pct": 100.0}
    assert summary["accuracy"]["has-target"] == {"ok": 1, "total": 2, "pct": 50.0}
    assert summary["coverage_lt_0_5"]["count"] == 1
    assert summary["coverage_lt_0_5"]["total"] == 3
    assert summary["coverage_lt_0_5"]["rate"] == pytest.approx(1 / 3, abs=1e-3)


def test_summarize_counts_missing_or_errored_answers_as_miss():
    cases = [make_case("0-orig", criteria={"1: [AXButton] Save": "w", NONE: "n"}, expected=["AXButton:Save"])]
    summary = distill_eval.summarize(cases, {"0-orig": {"error": "boom"}})
    assert summary["errors"] == 1
    assert summary["accuracy"]["all"] == {"ok": 0, "total": 1, "pct": 0.0}


def test_format_table_does_not_crash_on_missing_buckets():
    summary = distill_eval.summarize([], {})
    table = distill_eval.format_table(summary)
    assert "all 0/0" in table


# ---------------------------------------------------------------------------
# distill/eval.py: score_cases (importable, decide_fn based, used by train.py)
# ---------------------------------------------------------------------------


def _planned_decide(plan: list[str]):
    """score_cases は cases を順番に 1 回ずつ decide するので、呼び出し順に計画したラベルを返す fake。"""
    it = iter(plan)

    def decide(system: str, prompt: str, labels: list[str]) -> dict[str, float]:
        assert system == core.SYSTEM_PROMPT
        assert "OPTIONS:" in prompt
        target = next(it)
        assert target in labels
        return {label: (0.9 if label == target else 0.1 / max(1, len(labels) - 1)) for label in labels}

    return decide


def test_score_cases_with_perfect_decider_scores_100_percent():
    cases = [
        make_case("0-orig", criteria={"1: [AXButton] Save": "w", "2: [AXButton] Cancel": "w", NONE: "n"}, expected=["AXButton:Save"]),
        make_case("1-orig", criteria={"3: [AXButton] X": "w", NONE: "n"}, expected=[NONE]),
    ]
    # 案 0: A=Save, B=Cancel, C=__none__ -> A が正解
    # 案 1: A=X, B=__none__ -> B が正解
    decide = _planned_decide(["A", "B"])
    summary = distill_eval.score_cases(cases, decide)
    assert summary["accuracy"]["all"] == {"ok": 2, "total": 2, "pct": 100.0}
    assert summary["coverage_lt_0_5"]["count"] == 0


def test_score_cases_wrong_label_counts_as_miss():
    cases = [make_case("0-orig", criteria={"1: [AXButton] Save": "w", "2: [AXButton] Cancel": "w", NONE: "n"}, expected=["AXButton:Save"])]
    decide = _planned_decide(["B"])  # Cancel を選んでしまう
    summary = distill_eval.score_cases(cases, decide)
    assert summary["accuracy"]["all"] == {"ok": 0, "total": 1, "pct": 0.0}


def test_score_cases_zero_mass_is_degraded_uniform_and_low_coverage():
    cases = [make_case("0-orig", criteria={"1: [AXButton] Save": "w", "2: [AXButton] Cancel": "w"}, expected=["AXButton:Save"])]

    def decide(system, prompt, labels):
        return {label: 0.0 for label in labels}

    summary = distill_eval.score_cases(cases, decide)
    # 全滅 -> 一様分布 -> 先頭ラベル(A = Save)を選ぶので、たまたま正解になる
    assert summary["accuracy"]["all"]["ok"] == 1
    assert summary["coverage_lt_0_5"] == {"count": 1, "total": 1, "rate": 1.0}


def test_score_cases_partial_mass_is_reported_as_coverage():
    cases = [make_case("0-orig", criteria={"1: [AXButton] Save": "w", "2: [AXButton] Cancel": "w"}, expected=["AXButton:Save"])]

    def decide(system, prompt, labels):
        # raw mass の合計が 0.3 (coverage 相当) で、その中では A が優勢
        return {"A": 0.27, "B": 0.03}

    summary = distill_eval.score_cases(cases, decide)
    assert summary["accuracy"]["all"]["ok"] == 1
    assert summary["coverage_lt_0_5"] == {"count": 1, "total": 1, "rate": 1.0}


# ---------------------------------------------------------------------------
# distill/eval.py: score_cases のトーナメント経路(候補 > 52、core.choice() と同じ 2 段)
# ---------------------------------------------------------------------------

_MENU_LINE_RE = re.compile(r"^([A-Za-z]): (.+) — ")


def _labels_in_prompt(prompt: str) -> dict[str, str]:
    """menu の 'label: option — desc' 行から {option: label} を作る(テスト用、本番コードは読まない)。"""
    mapping: dict[str, str] = {}
    for line in prompt.splitlines():
        m = _MENU_LINE_RE.match(line)
        if m:
            mapping[m.group(2)] = m.group(1)
    return mapping


def _content_based_decide(target_option: str, *, prompts: list[str] | None = None, label_counts: list[int] | None = None):
    """prompt の menu に target_option が出てくるラウンドでだけ高い mass を返す fake。

    score_cases のトーナメントは呼び出し順を跨いで束→決勝と進むため、_planned_decide の
    「呼び出し順に計画したラベルを返す」方式ではラウンドを跨いだ検証ができない。
    menu の中身(本番コードが組んだプロンプト)から judge するので、実装の内部構造に依存しない。
    """

    def decide(system: str, prompt: str, labels: list[str]) -> dict[str, float]:
        assert system == core.SYSTEM_PROMPT
        if prompts is not None:
            prompts.append(prompt)
        if label_counts is not None:
            label_counts.append(len(labels))
        mapping = _labels_in_prompt(prompt)
        target_label = mapping.get(target_option)
        return {label: (0.9 if label == target_label else 0.01) for label in labels}

    return decide


def test_score_cases_tournament_scores_129_options_without_crashing():
    """#(20時間蒸留の後で落ちた実害): 候補129個(52を超える)が例外なく採点できること。"""
    criteria = {f"option-{i}": f"desc-{i}" for i in range(129)}
    target = "option-100"  # 3束中2つ目の束に入る位置(0-51/52-103/104-128)
    cases = [make_case("0-orig", criteria=criteria, expected=[target])]

    label_counts: list[int] = []
    decide = _content_based_decide(target, label_counts=label_counts)

    summary = distill_eval.score_cases(cases, decide)

    assert summary["errors"] == 0
    assert summary["accuracy"]["all"] == {"ok": 1, "total": 1, "pct": 100.0}
    assert all(n <= core.MAX_OPTIONS for n in label_counts)


def test_score_cases_tournament_keeps_none_in_every_chunk_and_final():
    """__none__ は全ての予選束と決勝に含まれる(本番 core.choice() と同じ扱い)。"""
    contenders = {f"option-{i}": f"desc-{i}" for i in range(80)}
    criteria = {**contenders, NONE: "該当なし"}
    cases = [make_case("0-orig", criteria=criteria, expected=[NONE])]

    prompts: list[str] = []
    label_counts: list[int] = []
    decide = _content_based_decide(NONE, prompts=prompts, label_counts=label_counts)

    summary = distill_eval.score_cases(cases, decide)

    assert summary["accuracy"]["all"] == {"ok": 1, "total": 1, "pct": 100.0}
    # per_chunk=51(NONE分を引く)、contenders=80 -> 束は51+29の2つ、+決勝の計3ラウンド
    assert len(prompts) == 3
    assert label_counts == [52, 30, 51]  # 各束は本体+NONE、決勝は上位25*2+NONE
    assert all(n <= core.MAX_OPTIONS for n in label_counts)
    assert all(NONE in p for p in prompts), "__none__ が含まれないラウンドがある"


def test_score_cases_exactly_max_options_is_single_round_not_tournament():
    """52 個ちょうど(境界)はこれまで通り 1 ラウンドのまま(トーナメント化されない)。"""
    criteria = {f"option-{i}": f"desc-{i}" for i in range(52)}
    target = "option-10"
    cases = [make_case("0-orig", criteria=criteria, expected=[target])]

    label_counts: list[int] = []
    decide = _content_based_decide(target, label_counts=label_counts)

    summary = distill_eval.score_cases(cases, decide)

    assert summary["accuracy"]["all"]["ok"] == 1
    assert label_counts == [52]  # 1 回だけ、52 ラベル全部で決着


def test_score_cases_small_cases_unaffected_by_tournament_change(monkeypatch):
    """候補 52 個以下のケースは、これまでと同じ選択・同じ decide 呼び出し回数(回帰しない)。"""
    cases = [
        make_case("0-orig", criteria={"1: [AXButton] Save": "w", "2: [AXButton] Cancel": "w", NONE: "n"}, expected=["AXButton:Save"]),
        make_case("1-orig", criteria={"3: [AXButton] X": "w", NONE: "n"}, expected=[NONE]),
    ]
    decide = _planned_decide(["A", "B"])
    summary = distill_eval.score_cases(cases, decide)
    assert summary["accuracy"]["all"] == {"ok": 2, "total": 2, "pct": 100.0}
    assert summary["coverage_lt_0_5"]["count"] == 0


def test_score_cases_raises_when_options_exceed_tournament_limit():
    """core.choice() と同じ上限(2704)を超えたら ValueError(decide は呼ばれない)。"""
    criteria = {f"option-{i}": "d" for i in range(core.MAX_TOURNAMENT_OPTIONS + 1)}
    cases = [make_case("0-orig", criteria=criteria, expected=["option-0"])]

    def decide(system, prompt, labels):
        raise AssertionError("decide は呼ばれてはいけない")

    with pytest.raises(ValueError, match="prefilter options"):
        distill_eval.score_cases(cases, decide)


# ---------------------------------------------------------------------------
# distill/eval.py: score_cases_via_url (HTTP モード、run_labels.py 相当)
# ---------------------------------------------------------------------------


class _FakeChakuhoServer:
    """POST /v1/systemone に固定の answers.target を返す fake(実 vLLM は叩かない)。"""

    def __init__(self) -> None:
        self.plan: dict[str, dict] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                return

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n))
                case_key = req["state"].get("_case_id")
                answer = outer.plan.get(case_key, {"choice": NONE, "coverage": 0.0})
                body = json.dumps({"answers": {"target": answer}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def fake_chakuho_server():
    srv = _FakeChakuhoServer()
    yield srv
    srv.close()


def test_score_cases_via_url_matches_run_labels_scoring(fake_chakuho_server):
    case = make_case("0-orig", criteria={"1: [AXButton] Save": "w", NONE: "n"}, expected=["AXButton:Save"])
    case["request"]["state"]["_case_id"] = "0-orig"
    fake_chakuho_server.plan["0-orig"] = {"choice": "1: [AXButton] Save", "coverage": 0.95}

    summary = distill_eval.score_cases_via_url([case], fake_chakuho_server.url, timeout=10)
    assert summary["accuracy"]["all"] == {"ok": 1, "total": 1, "pct": 100.0}
    assert summary["coverage_lt_0_5"]["count"] == 0


def test_score_cases_via_url_reports_unreachable_server_as_error():
    case = make_case("0-orig", criteria={"1: [AXButton] Save": "w", NONE: "n"}, expected=["AXButton:Save"])
    summary = distill_eval.score_cases_via_url([case], "http://127.0.0.1:1", timeout=1)
    assert summary["errors"] == 1
    assert summary["accuracy"]["all"] == {"ok": 0, "total": 1, "pct": 0.0}


# ---------------------------------------------------------------------------
# distill/mlx_backend.py: 純粋関数
# ---------------------------------------------------------------------------


def test_select_top_k_orders_by_logit_descending_with_stable_ties():
    logits = [0.1, 5.0, 5.0, -3.0, 2.0]
    assert mlx_backend.select_top_k(logits, 3) == [1, 2, 4]  # 同点(1,2)は id 昇順


def test_select_top_k_caps_at_vocab_size():
    assert mlx_backend.select_top_k([1.0, 2.0], 10) == [1, 0]


def test_select_top_k_rejects_non_positive_k():
    with pytest.raises(ValueError):
        mlx_backend.select_top_k([1.0], 0)


def test_log_softmax_at_matches_manual_softmax():
    import math

    logits = [0.0, 1.0, 2.0]
    out = mlx_backend.log_softmax_at(logits, [0, 1, 2])
    total = sum(math.exp(v) for v in out.values())
    assert total == pytest.approx(1.0)
    assert out[2] > out[1] > out[0]


def test_build_response_shape_matches_core_query_backend_parsing():
    response = mlx_backend.build_response(
        model_id="student-v1",
        token_texts={0: "A", 1: "B"},
        logits=[3.0, 1.0],
        top_k_ids=[0, 1],
        prompt_tokens=17,
    )
    # chakuho/core.py query_backend が実際に読む経路と同じ辿り方で検証する
    top_entries = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    top_logprobs = {e["token"]: e["logprob"] for e in top_entries}
    assert set(top_logprobs) == {"A", "B"}
    assert top_logprobs["A"] > top_logprobs["B"]
    assert response["usage"]["prompt_tokens"] == 17


def test_validate_request_accepts_defaults_and_rejects_other_max_tokens():
    messages, top_logprobs, enable_thinking = mlx_backend.validate_request(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
    )
    assert messages == [{"role": "user", "content": "hi"}]
    assert top_logprobs == mlx_backend.DEFAULT_TOP_LOGPROBS
    assert enable_thinking is False

    with pytest.raises(ValueError):
        mlx_backend.validate_request({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5})


def test_validate_request_honors_top_logprobs_and_enable_thinking():
    _, top_logprobs, enable_thinking = mlx_backend.validate_request(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "top_logprobs": 5,
            "chat_template_kwargs": {"enable_thinking": True},
        }
    )
    assert top_logprobs == 5
    assert enable_thinking is True


def test_validate_request_rejects_missing_messages():
    with pytest.raises(ValueError):
        mlx_backend.validate_request({"max_tokens": 1})


# ---------------------------------------------------------------------------
# distill/mlx_backend.py: HTTP ハンドラ(fake student、mlx 不使用)
# ---------------------------------------------------------------------------


class _FakeStudent:
    """StudentModel と同じ最小インターフェース。mlx は一切使わない。"""

    def __init__(self, vocab: list[str], logits: list[float], prompt_tokens: int = 42) -> None:
        self.model_path = "fake-student-v1"
        self.vocab = vocab
        self.logits = logits
        self.prompt_tokens = prompt_tokens
        self.calls: list[tuple[list[dict], bool]] = []

    def decide(self, messages, enable_thinking):
        self.calls.append((messages, enable_thinking))
        return list(self.logits), self.prompt_tokens

    def token_text(self, token_id: int) -> str:
        return self.vocab[token_id]


@pytest.fixture
def mlx_server():
    vocab = [chr(ord("A") + i) for i in range(26)] + [f"<junk{i}>" for i in range(4)]
    logits = [0.0] * len(vocab)
    logits[vocab.index("B")] = 5.0  # "B" が圧倒的に勝つ
    student = _FakeStudent(vocab, logits)
    # serve() は StudentModel(mlx_lm 必須)を作ってしまうので、ここでは make_handler を直接使う
    handler_cls = mlx_backend.make_handler(student)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield url, student
    httpd.shutdown()
    httpd.server_close()


def _post(url: str, payload: dict) -> tuple[int, dict]:
    req = urllib_request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib_request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib_error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_mlx_backend_get_models(mlx_server):
    url, _student = mlx_server
    with urllib_request.urlopen(url + "/v1/models", timeout=10) as resp:
        data = json.loads(resp.read())
    assert data == {"data": [{"id": "fake-student-v1"}]}


def test_mlx_backend_rejects_max_tokens_other_than_1(mlx_server):
    url, _ = mlx_server
    code, out = _post(url + "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
    assert code == 400
    assert "max_tokens" in out["error"]


def test_mlx_backend_returns_openai_shaped_response_and_honors_top_logprobs(mlx_server):
    url, student = mlx_server
    code, out = _post(
        url + "/v1/chat/completions",
        {
            "model": "ignored",
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}],
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 5,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert code == 200
    entries = out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    assert len(entries) == 5
    tokens = {e["token"] for e in entries}
    assert "B" in tokens  # 最尤トークンが候補に入っている
    assert out["usage"]["prompt_tokens"] == 42
    assert student.calls[-1][1] is False  # enable_thinking が伝わっている


def test_mlx_backend_end_to_end_via_core_query_backend(mlx_server):
    """chakuho/core.py の実際の query_backend() でこのレスポンスを読ませ、aggregate まで通す。"""
    url, student = mlx_server
    top_logprobs, prompt_tokens = core.query_backend(
        "Allowed labels: A, B\nLabel:", url + "/v1", "any-model-id"
    )
    assert prompt_tokens == 42
    dist, coverage = core.aggregate(top_logprobs, ["A", "B"])
    assert coverage > 0
    assert dist["B"] > dist["A"]


# ---------------------------------------------------------------------------
# distill/train.py: torch 抜きでテストできる純粋関数
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    def __init__(self, table: dict[str, list[int]]) -> None:
        self.table = table

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.table[text]


def test_label_token_ids_uses_first_token_of_multi_token_labels():
    tok = _FakeTokenizer({"yes": [20, 21], "no": [22]})
    assert distill_train.label_token_ids(tok, ["yes", "no"]) == [20, 22]


def test_label_token_ids_single_char_labels():
    tok = _FakeTokenizer({chr(c): [c] for c in range(ord("A"), ord("Z") + 1)})
    labels = distill_train.label_token_ids(tok, ["A", "B", "C"])
    assert labels == [ord("A"), ord("B"), ord("C")]


def test_label_token_ids_raises_on_collision():
    tok = _FakeTokenizer({"A": [10], "B": [10]})
    with pytest.raises(ValueError, match="not distinct"):
        distill_train.label_token_ids(tok, ["A", "B"])


def test_label_token_ids_raises_on_empty_encoding():
    tok = _FakeTokenizer({"A": []})
    with pytest.raises(ValueError, match="zero tokens"):
        distill_train.label_token_ids(tok, ["A"])


def test_kl_from_teacher_dict_zero_for_matching_distribution():
    import math

    teacher = {"A": 0.7, "B": 0.3}
    log_q = [math.log(0.7), math.log(0.3)]
    kl = distill_train.kl_from_teacher_dict(teacher, ["A", "B"], log_q)
    assert kl == pytest.approx(0.0, abs=1e-9)


def test_kl_from_teacher_dict_positive_when_mismatched():
    import math

    teacher = {"A": 0.9, "B": 0.1}
    log_q = [math.log(0.5), math.log(0.5)]
    kl = distill_train.kl_from_teacher_dict(teacher, ["A", "B"], log_q)
    assert kl > 0


def test_kl_from_teacher_dict_rejects_zero_mass_teacher():
    with pytest.raises(ValueError):
        distill_train.kl_from_teacher_dict({"A": 0.0, "B": 0.0}, ["A", "B"], [0.0, 0.0])


def test_load_dataset_jsonl_reads_records_and_respects_limit(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(json.dumps({"id": i}) for i in range(5)) + "\n")
    assert len(distill_train.load_dataset_jsonl(str(path))) == 5
    assert len(distill_train.load_dataset_jsonl(str(path), limit=2)) == 2


def test_build_argparser_defaults_match_design():
    args = distill_train.build_argparser().parse_args(
        ["--data", "d.jsonl", "--model", "Qwen/Qwen3-1.7B", "--out", "out/"]
    )
    assert args.epochs == 4
    assert args.lr == pytest.approx(1e-4)
    assert args.lora_r == 16
    assert args.batch == 8
    assert args.lm_weight == pytest.approx(0.0)
    assert args.eval_every_epoch is False


def test_torch_dependent_functions_fail_clearly_without_torch():
    if distill_train._TORCH_AVAILABLE:
        pytest.skip("torch is installed in this environment; nothing to guard against here")
    with pytest.raises(RuntimeError):
        distill_train.train_one_epoch(None, None, None, None, None, lm_weight=0.0, grad_accum=1)
    with pytest.raises(RuntimeError):
        distill_train.make_student_decider(None, None)


# ---------------------------------------------------------------------------
# distill/train.py: 保存済みアダプタからの再開(--resume-adapter / --start-epoch)
# ---------------------------------------------------------------------------


def test_resolve_start_epoch_defaults_to_1_without_resume():
    assert distill_train.resolve_start_epoch(None, 0) == 1


def test_resolve_start_epoch_derives_next_epoch_from_adapter_dir_name():
    assert distill_train.resolve_start_epoch("/data/run-46/out/epoch-1", 0) == 2
    assert distill_train.resolve_start_epoch("/data/run-46/out/epoch-7/", 0) == 8


def test_resolve_start_epoch_explicit_value_wins_over_dir_name():
    assert distill_train.resolve_start_epoch("/data/run-46/out/epoch-1", 5) == 5


def test_resolve_start_epoch_raises_when_dir_name_is_not_derivable():
    with pytest.raises(ValueError):
        distill_train.resolve_start_epoch("/data/run-46/out/best", 0)


def test_epoch_numbers_counts_forward_from_start():
    assert distill_train.epoch_numbers(2, 1) == [2]
    assert distill_train.epoch_numbers(2, 3) == [2, 3, 4]


def test_epoch_numbers_rejects_non_positive():
    with pytest.raises(ValueError):
        distill_train.epoch_numbers(0, 1)
    with pytest.raises(ValueError):
        distill_train.epoch_numbers(1, 0)


def test_check_resume_adapter_returns_path_when_adapter_config_present(tmp_path):
    d = tmp_path / "epoch-1"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert distill_train.check_resume_adapter(str(d)) == d


def test_check_resume_adapter_raises_when_missing(tmp_path):
    d = tmp_path / "epoch-1"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        distill_train.check_resume_adapter(str(d))


def test_assert_epoch_dirs_free_raises_when_target_would_be_overwritten(tmp_path):
    (tmp_path / "epoch-2").mkdir()
    with pytest.raises(FileExistsError):
        distill_train.assert_epoch_dirs_free(tmp_path, [2, 3])


def test_assert_epoch_dirs_free_passes_when_only_earlier_epochs_exist(tmp_path):
    (tmp_path / "epoch-1").mkdir()
    distill_train.assert_epoch_dirs_free(tmp_path, [2, 3])


def test_argparser_accepts_resume_adapter_and_start_epoch():
    args = distill_train.build_argparser().parse_args(
        ["--data", "d.jsonl", "--model", "m", "--out", "o",
         "--resume-adapter", "/data/run-46/out/epoch-1", "--start-epoch", "2"]
    )
    assert args.resume_adapter == "/data/run-46/out/epoch-1"
    assert args.start_epoch == 2


def test_argparser_defaults_have_no_resume():
    args = distill_train.build_argparser().parse_args(["--data", "d.jsonl", "--model", "m", "--out", "o"])
    assert args.resume_adapter is None
    assert args.start_epoch == 0


# ---------------------------------------------------------------------------
# distill/train.py: 定期評価と早期停止(停止条件 S1 / S2)
# ---------------------------------------------------------------------------


def test_goals_met_requires_both_overall_and_none_tasks():
    assert distill_train.goals_met({"all": 90.0, "none-tasks": 90.0}, 90.0, 90.0) is True
    assert distill_train.goals_met({"all": 92.0, "none-tasks": 51.3}, 90.0, 90.0) is False
    assert distill_train.goals_met({"all": 87.8, "none-tasks": 95.0}, 90.0, 90.0) is False


def test_goals_met_is_false_when_a_metric_is_missing():
    assert distill_train.goals_met({"all": 95.0}, 90.0, 90.0) is False


def test_plateaued_needs_patience_evaluations_without_gain():
    # 最良 87.8 のあと 2 回 +1.0 未満 → 頭打ち
    assert distill_train.plateaued([87.8, 88.0, 88.2], patience=2, min_delta=1.0) is True


def test_plateaued_is_false_when_a_recent_eval_improved_enough():
    assert distill_train.plateaued([87.8, 88.0, 89.5], patience=2, min_delta=1.0) is False


def test_plateaued_is_false_before_enough_evaluations():
    assert distill_train.plateaued([87.8], patience=2, min_delta=1.0) is False
    assert distill_train.plateaued([87.8, 88.0], patience=2, min_delta=1.0) is False


def test_plateaued_compares_against_best_so_far_not_previous():
    # 一度 92 まで上がってから下がった場合、直前比では改善でも最良比では未改善
    assert distill_train.plateaued([92.0, 88.0, 89.0], patience=2, min_delta=1.0) is True


def test_pct_of_reads_accuracy_bucket():
    ev = {"accuracy": {"all": {"pct": 87.8}, "none-tasks": {"pct": 51.3}}}
    assert distill_train.pct_of(ev) == {"all": 87.8, "none-tasks": 51.3}


def test_pct_of_tolerates_missing_buckets():
    assert distill_train.pct_of({"accuracy": {}}) == {}
    assert distill_train.pct_of({}) == {}


def test_argparser_accepts_checkpoint_and_early_stop_options():
    a = distill_train.build_argparser().parse_args(
        ["--data", "d", "--model", "m", "--out", "o",
         "--checkpoint-every-steps", "250", "--early-stop-patience", "2", "--early-stop-min-delta", "1.0"]
    )
    assert a.checkpoint_every_steps == 250
    assert a.early_stop_patience == 2
    assert a.early_stop_min_delta == 1.0


def test_argparser_checkpointing_is_off_by_default():
    a = distill_train.build_argparser().parse_args(["--data", "d", "--model", "m", "--out", "o"])
    assert a.checkpoint_every_steps == 0


def test_train_one_epoch_stops_when_callback_returns_true():
    """torch 無しでも、ループ制御だけは偽物で検証できる。"""
    calls = []

    def on_checkpoint(step):
        calls.append(step)
        return True  # 1 回目で停止を要求

    stopped = distill_train.checkpoint_due(step=250, checkpoint_every=250)
    assert stopped is True
    assert distill_train.checkpoint_due(step=249, checkpoint_every=250) is False
    assert distill_train.checkpoint_due(step=0, checkpoint_every=250) is False
    assert distill_train.checkpoint_due(step=250, checkpoint_every=0) is False


def test_decide_cases_returns_per_case_answers_matching_score_cases():
    """decide_cases の答えを summarize に通すと score_cases と一致する(経路が同じ)。"""
    from distill import eval as dist_eval

    cases = [
        {
            "case_id": "c1",
            "app": "X",
            "variant": "orig",
            "expected": ["AXButton:Go"],
            "request": {"state": {}, "questions": {"target": {
                "type": "choice", "instructions": "i",
                "criteria": {"0: [AXButton] Go": "d", "1: [AXButton] Stop": "d"}}}},
        }
    ]

    def decide(system, prompt, labels):
        return {labels[0]: 0.9}

    answers = dist_eval.decide_cases(cases, decide)
    assert set(answers) == {"c1"}
    assert answers["c1"]["choice"] == "0: [AXButton] Go"
    assert answers["c1"]["coverage"] == 0.9
    assert dist_eval.summarize(cases, answers) == dist_eval.score_cases(cases, decide)
