"""⑤ 評価: 336 ケースのベンチを chakuho core と同じ集約ルールで採点する(stdlib のみ)。

2 つのモード:

(a) CLI, ``--url http://host:9750``: cases.json(+ 任意で cases2.json)を chakuho サーバ
    (POST /v1/systemone)へ流し、bench/run_labels.py と同じ採点(全体・変種別・アプリ別・
    __none__ 課題・coverage<0.5 率)を出す。

(b) ライブラリ, :func:`score_cases`: ``decide(system, prompt, labels) -> {label: p}`` を
    実装した呼び出し可能オブジェクトを渡すと、同じケース集合・同じ採点ロジックでスコアが
    取れる。distill/train.py はこれで学生モデルを in-process 評価する(サーバ不要)。
    ここで言う ``p`` はラベルの生の(正規化前の)質量でよい。合計が 1 未満なら
    coverage(ラベルの形を守れた確率質量)として扱う。

ケース JSON の形式は bench/build_cases.py が書くものと同じ:
  {"case_id": str, "app": str, "variant": str, "instruction": str,
   "expected": [str, ...], "request": {"state": ..., "questions": {"target": {
       "type": "choice", "instructions": ..., "criteria": {option: desc, ...}}}}}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable
from urllib import error as urllib_error
from urllib import request as urllib_request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chakuho import core  # noqa: E402

NONE = core.NONE_OPTION
DEFAULT_TIMEOUT = 300.0

# choice の答えは "12: [AXButton] Settings" の形("id: [role] label")。
# bench の expected は role:label(id を含まない)なので、id を落として正規化する。
_ROLE_LABEL_RE = re.compile(r"^\d+: \[(\w+)\] (.*)$")


def normalize_choice(choice_value: str) -> str:
    """"12: [AXButton] Settings" -> "AXButton:Settings"。__none__ はそのまま。"""
    if choice_value == NONE:
        return NONE
    m = _ROLE_LABEL_RE.match(choice_value)
    return f"{m.group(1)}:{m.group(2)}" if m else choice_value


def load_cases(*paths: str) -> list[dict]:
    """1 つ以上の cases.json を連結して読む(重複 case_id は後勝ち)。"""
    by_id: dict[str, dict] = {}
    for path in paths:
        for case in json.loads(Path(path).read_text()):
            by_id[case["case_id"]] = case
    return list(by_id.values())


def _bump(buckets: dict[str, dict[str, int]], key: str, hit: bool) -> None:
    bucket = buckets.setdefault(key, {"ok": 0, "total": 0})
    bucket["total"] += 1
    bucket["ok"] += int(bool(hit))


def summarize(cases: list[dict], answers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """cases と、case_id -> {"choice": str, "coverage": float|None} または {"error": str} から
    集計(全体・変種別・アプリ別・__none__ 課題・coverage<0.5 率)を作る。"""
    buckets: dict[str, dict[str, int]] = {}
    coverage_values: list[float] = []
    errors = 0
    for case in cases:
        answer = answers.get(case["case_id"], {"error": "no answer"})
        if "error" in answer or "choice" not in answer:
            errors += 1
            hit = False
        else:
            hit = normalize_choice(answer["choice"]) in case["expected"]
            if isinstance(answer.get("coverage"), (int, float)):
                coverage_values.append(float(answer["coverage"]))
        is_none_task = case["expected"] == [NONE]
        _bump(buckets, "all", hit)
        _bump(buckets, f"variant:{case.get('variant', '?')}", hit)
        _bump(buckets, f"app:{case.get('app', '?')}", hit)
        _bump(buckets, "none-tasks" if is_none_task else "has-target", hit)

    def _pct(bucket: dict[str, int]) -> float | None:
        return round(100 * bucket["ok"] / bucket["total"], 1) if bucket["total"] else None

    accuracy = {key: {**bucket, "pct": _pct(bucket)} for key, bucket in buckets.items()}
    low_coverage = sum(1 for v in coverage_values if v < 0.5)
    return {
        "n_cases": len(cases),
        "errors": errors,
        "accuracy": accuracy,
        "coverage_lt_0_5": {
            "count": low_coverage,
            "total": len(coverage_values),
            "rate": round(low_coverage / len(coverage_values), 4) if coverage_values else None,
        },
    }


def format_table(summary: dict[str, Any]) -> str:
    """run_labels.py と同じ見た目の人間向け表を作る。"""
    acc = summary["accuracy"]

    def f(key: str) -> str:
        b = acc.get(key, {"ok": 0, "total": 0, "pct": None})
        pct = f"{b['pct']:.0f}%" if b["pct"] is not None else "-"
        return f"{b['ok']}/{b['total']} ({pct})"

    lines = [
        f"all {f('all')} | orig {f('variant:orig')} shuffled {f('variant:shuffled')} "
        f"distractors {f('variant:distractors')} | has-target {f('has-target')} none-tasks {f('none-tasks')}"
    ]
    apps = sorted(k[len("app:"):] for k in acc if k.startswith("app:"))
    if apps:
        lines.append("per app: " + ", ".join(f"{a} {f('app:' + a)}" for a in apps))
    cov = summary["coverage_lt_0_5"]
    if cov["total"]:
        lines.append(f"coverage<0.5: {cov['count']}/{cov['total']} ({cov['rate'] * 100:.1f}%)")
    else:
        lines.append("coverage<0.5: n/a(coverage 情報なし)")
    if summary["errors"]:
        lines.append(f"errors: {summary['errors']}")
    return "\n".join(lines)


def _round_prompt(case: dict, options: list[str]) -> tuple[str, str, list[str]]:
    """options(52 以下、1 ラウンド分)に対して chakuho core と同じ system/prompt/labels を組む。"""
    question = case["request"]["questions"]["target"]
    criteria = question["criteria"]
    labels = core.labels_for(len(options))
    menu = "\n".join(f"{label}: {option} — {criteria[option]}" for label, option in zip(labels, options))
    prompt = core._build_prompt(
        core._render(question.get("instructions", "")),
        menu,
        core._render(case["request"]["state"]),
        "choose one option.",
        labels,
    )
    return core.SYSTEM_PROMPT, prompt, labels


def _choice_prompt(case: dict) -> tuple[str, str, list[str], list[str]]:
    """case の request 全選択肢(52 以下)から chakuho core と同じ prompt を組む。

    Returns:
        (system, prompt, labels, options) — options[i] は labels[i] に対応する元の選択肢名。
    """
    options = list(case["request"]["questions"]["target"]["criteria"].keys())
    system, prompt, labels = _round_prompt(case, options)
    return system, prompt, labels, options


DecideFn = Callable[[str, str, list[str]], dict[str, float]]


def _decide_round(case: dict, options: list[str], decide: DecideFn) -> dict[str, float]:
    """options(52 以下、1 ラウンド分)を decide し、ラベルではなく option 名をキーにした raw mass を返す。"""
    system, prompt, labels = _round_prompt(case, options)
    raw = decide(system, prompt, labels)
    return {option: raw.get(label, 0.0) for label, option in zip(labels, options)}


def _tournament_decide(case: dict, options: list[str], decide: DecideFn) -> dict[str, float]:
    """52 個を超える選択肢を、chakuho core.choice() と同じ 2 段トーナメントで decide する。

    52 個(__none__ があれば 51 個)ずつの束へ分けて束ごとに decide し、各束の上位
    k 個(k = 束あたりの定員 // 束数)を決勝へ進める。__none__ は全ての束と決勝に必ず
    含める(束の得票には数えない)。予選落ちした選択肢の raw mass は 0.0 として返す
    (core.choice() の「予選落ち・決勝敗退は probabilities 0.0」と同じ扱い)。

    core.choice() は束ごとの decide を ThreadPoolExecutor で並列に投げるが、ここでは
    直列に呼ぶ。decide は HTTP ではなく in-process のモデル呼び出し(train.py の
    make_student_decider はスレッドセーフでない model.eval()/model.train() 切り替えを
    含む)なので、並列化すると結果が変わらなくても壊れうる。束の処理順序・上位k個の
    選び方は core.choice() と同じなので、並列/直列で最終結果(選択・probabilities)は
    変わらない — 変わるのは実行時間だけ。
    """
    has_none = core.NONE_OPTION in options
    contenders = [o for o in options if o != core.NONE_OPTION]
    per_chunk = core.MAX_OPTIONS - (1 if has_none else 0)
    chunks = [contenders[i : i + per_chunk] for i in range(0, len(contenders), per_chunk)]
    if has_none:
        chunks = [chunk + [core.NONE_OPTION] for chunk in chunks]
    top_k = max(1, per_chunk // len(chunks))

    winners: list[str] = []
    for chunk in chunks:
        raw = _decide_round(case, chunk, decide)
        ranked = sorted((o for o in chunk if o != core.NONE_OPTION), key=lambda o: raw[o], reverse=True)
        winners.extend(ranked[:top_k])
    if has_none:
        winners.append(core.NONE_OPTION)

    final_raw = _decide_round(case, winners, decide)
    return {option: final_raw.get(option, 0.0) for option in options}


def score_cases(cases: list[dict], decide: DecideFn) -> dict[str, Any]:
    """decide(system, prompt, labels) -> {label: raw_mass} を使って cases を採点する。

    raw_mass は正規化前でよい(合計 < 1 なら coverage として扱う)。全滅(合計 0)の時は
    一様分布とみなし choice は先頭ラベルになる(core.aggregate の degraded 挙動に合わせる)。
    候補が 52 個を超えるケースは core.choice() と同じ 2 段トーナメント(_tournament_decide)
    で decide する。2704 個(core.MAX_TOURNAMENT_OPTIONS)を超えると core.choice() と同じ
    ValueError になる。
    """
    answers: dict[str, dict[str, Any]] = {}
    for case in cases:
        options = list(case["request"]["questions"]["target"]["criteria"].keys())
        if len(options) > core.MAX_TOURNAMENT_OPTIONS:
            raise ValueError(
                f"prefilter options to {core.MAX_TOURNAMENT_OPTIONS} or fewer (got {len(options)})"
            )
        if len(options) <= core.MAX_OPTIONS:
            raw = _decide_round(case, options, decide)
        else:
            raw = _tournament_decide(case, options, decide)
        coverage = sum(raw.values())
        if coverage <= 0:
            dist = {option: 1.0 / len(options) for option in options}
        else:
            dist = {option: raw.get(option, 0.0) / coverage for option in options}
        best_option = max(dist, key=dist.get)
        answers[case["case_id"]] = {"choice": best_option, "coverage": coverage}
    return summarize(cases, answers)


def score_cases_via_url(cases: list[dict], url: str, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """bench/run_labels.py の via_chakuho と同じ形で、chakuho サーバへ流して採点する。"""
    answers: dict[str, dict[str, Any]] = {}
    endpoint = url.rstrip("/") + "/v1/systemone"
    for case in cases:
        body = json.dumps(case["request"]).encode("utf-8")
        req = urllib_request.Request(endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib_request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            target = data["answers"]["target"]
            answers[case["case_id"]] = {"choice": target["choice"], "coverage": target.get("coverage")}
        except (urllib_error.URLError, OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            answers[case["case_id"]] = {"error": str(exc)[:200]}
    return summarize(cases, answers)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True, help="chakuho サーバの base URL(例: http://localhost:9750)")
    ap.add_argument("--cases", default="cases.json")
    ap.add_argument("--cases2", default=None, help="任意: 追加のケースファイル(例: cases2.json)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = ap.parse_args()

    paths = [args.cases] + ([args.cases2] if args.cases2 else [])
    cases = load_cases(*paths)
    summary = score_cases_via_url(cases, args.url, timeout=args.timeout)
    print(json.dumps(summary, ensure_ascii=False))
    print(format_table(summary))


if __name__ == "__main__":
    main()
