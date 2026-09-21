"""実運用の判定ログを 27B(教師)へ流し直し、学生と食い違ったものを出す。

学生を常駐させた後、**「なんだか判定がおかしい」と感じた時に回す**道具。
新しいベンチを作らず、実際に起きた判定そのものを母集団にする。

学生の外し方は静かなので(coverage が高いまま自信を持って間違える)、
使っているだけでは気づけない。**気づくための手段を、常駐と一緒に置いておく。**

使い方:
    # 直近 200 件を 27B と突き合わせる
    python3 bench/replay_against_teacher.py \
        --logs ~/.ato/chakuho/decisions-*.jsonl --last 200 \
        --teacher http://gx10:8006/v1

    # 気になった質問名だけ
    python3 bench/replay_against_teacher.py --logs ... --question target

出力は質問名ごとの一致率と、食い違った事例。**母集団の定義を必ず添える。**
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chakuho import core  # noqa: E402


def load(paths: list[str], last: int, question: str | None) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") != 200 or "answers" not in r.get("response", {}):
                continue
            if question and question not in r["request"].get("questions", {}):
                continue
            rows.append(r)
    rows.sort(key=lambda r: r.get("t", 0))
    return rows[-last:] if last else rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="decisions-*.jsonl(複数可)")
    ap.add_argument("--teacher", required=True, help="27B の backend URL(例 http://gx10:8006/v1)")
    ap.add_argument("--last", type=int, default=200, help="末尾 N 件(0 で全件)")
    ap.add_argument("--question", default=None, help="この質問名を含むものだけ")
    ap.add_argument("--show", type=int, default=15, help="食い違いを何件まで出すか")
    ap.add_argument("--out", default=None, help="明細の書き出し先 JSON")
    args = ap.parse_args()

    rows = load(args.logs, args.last, args.question)
    if not rows:
        print("対象が 0 件。--logs のパスと --question を確かめること", file=sys.stderr)
        raise SystemExit(2)

    model = core.resolve_model(args.teacher)
    print(f"母集団: {len(rows)} レコード(status 200 かつ answers あり"
          + (f"、質問名 {args.question!r} を含むもの" if args.question else "")
          + f")。教師 = {model} @ {args.teacher}", flush=True)

    agree: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    noul_flip: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    diffs: list[dict] = []
    errors = 0
    for i, r in enumerate(rows):
        try:
            teacher = core.evaluate(r["request"].get("state"), r["request"]["questions"],
                                    backend_url=args.teacher, model=model)["answers"]
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"  {i}: 教師に聞けなかった {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        logged = r["response"]["answers"]
        for name, t in teacher.items():
            lg = logged.get(name)
            if not lg:
                continue
            if "choice" in t and "choice" in lg:
                agree[name][1] += 1
                ok = t["choice"] == lg["choice"]
                agree[name][0] += ok
                if not ok:
                    diffs.append({"i": i, "question": name, "運用時": lg["choice"], "教師": t["choice"],
                                  "運用時のcoverage": lg.get("coverage")})
            elif "noul" in t and "noul" in lg:
                noul_flip[name][1] += 1
                noul_flip[name][0] += (t["noul"] >= 0.5) != (lg["noul"] >= 0.5)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    print("\n== 質問名ごとの choice 一致(運用時の答え 対 いまの 27B)")
    for name, (ok, n) in sorted(agree.items(), key=lambda kv: kv[1][0] / max(1, kv[1][1])):
        print(f"  {name:24} {ok}/{n} = {100*ok/max(1,n):.1f}%")
    if noul_flip:
        print("\n== noul の真偽が反転した数")
        for name, (f, n) in sorted(noul_flip.items(), key=lambda kv: -kv[1][0]):
            print(f"  {name:24} {f}/{n} = {100*f/max(1,n):.1f}%")
    if errors:
        print(f"\n教師に聞けなかったレコード: {errors} 件(一致率の分母から外してある)")

    print(f"\n== 食い違い(先頭 {args.show} 件)")
    for d in diffs[:args.show]:
        print(f"  [{d['question']}] 運用時={d['運用時']!r}(coverage {d['運用時のcoverage']})")
        print(f"                   教師={d['教師']!r}")
    if len(diffs) > args.show:
        print(f"  ...他 {len(diffs)-args.show} 件")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"母集団": {"レコード": len(rows), "質問名": args.question, "教師": model,
                        "教師に聞けなかった": errors},
             "choice一致": {k: {"ok": v[0], "total": v[1]} for k, v in agree.items()},
             "noul反転": {k: {"flip": v[0], "total": v[1]} for k, v in noul_flip.items()},
             "食い違い": diffs}, ensure_ascii=False, indent=2))
        print(f"\n明細: {args.out}")


if __name__ == "__main__":
    main()
