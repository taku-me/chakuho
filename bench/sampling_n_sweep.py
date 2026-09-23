"""sampling 推定器(gx10-642)の n(サンプル数)を選ぶための一括測定。

実運用の判定ログ(SGLang+DSpark 切替前、2026-09-23 09:16 までの logprobs 推定器の答え)を
「logprob 時代の基準線」として使い、同じ state/questions を今の backend へ
sampling 推定器(n=16/32/64)で流し直して、choice の一致率と latency を比べる。

replay_against_teacher.py と同じ発想(新しいベンチデータを作らず、実際に起きた
判定そのものを母集団にする)だが、(1) 複数の n を掃引し (2) 一致率だけでなく
レイテンシ(median/p90)も測る点が異なる。母集団は choice 型の 1 質問レコードに絞り、
option 数の分布(小: <=20 / 中: 21-52 / トーナメント: >52)を均等に混ぜて偏りを避ける。

使い方:
    python3 bench/sampling_n_sweep.py \
        --logs ~/.ato/chakuho/decisions-*.jsonl \
        --backend http://localhost:8006/v1 \
        --n 16 32 64 --per-bucket 12 --out /tmp/sweep.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chakuho import core  # noqa: E402


def load_choice_records(paths: list[str]) -> list[dict]:
    """status 200 かつ choice 型 1 質問だけのレコードを、時刻順で返す。"""
    rows: list[dict] = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") != 200 or "answers" not in r.get("response", {}):
                continue
            qs = r["request"].get("questions", {})
            if len(qs) != 1:
                continue
            name, q = next(iter(qs.items()))
            if q.get("type") != "choice":
                continue
            crit = q.get("criteria")
            n_opts = len(crit) if isinstance(crit, (list, dict)) else 0
            if n_opts < 2:
                continue
            r["_qname"] = name
            r["_n_opts"] = n_opts
            rows.append(r)
    rows.sort(key=lambda r: r.get("t", 0))
    return rows


def bucket_of(n_opts: int) -> str:
    if n_opts <= 20:
        return "small(<=20)"
    if n_opts <= core.MAX_OPTIONS:
        return "medium(21-52)"
    return "tournament(>52)"


def stratified_sample(rows: list[dict], per_bucket: int) -> list[dict]:
    """3 バケット(small/medium/tournament)から、それぞれ直近 per_bucket 件を取る。
    重複 state を避けるため、質問名+state のハッシュで de-dup してから取る。"""
    buckets: dict[str, list[dict]] = {"small(<=20)": [], "medium(21-52)": [], "tournament(>52)": []}
    seen: set[str] = set()
    for r in reversed(rows):  # 新しい方から
        key = json.dumps(r["request"], sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        b = bucket_of(r["_n_opts"])
        if len(buckets[b]) < per_bucket:
            buckets[b].append(r)
    out = [r for bucket in buckets.values() for r in bucket]
    print("母集団の内訳: " + ", ".join(f"{k}={len(v)}" for k, v in buckets.items()), flush=True)
    return out


def run_sweep(rows: list[dict], backend_url: str, n_values: list[int]) -> dict:
    model = core.resolve_model(backend_url)
    print(f"backend={backend_url} model={model} レコード数={len(rows)}", flush=True)
    result: dict[str, dict] = {}
    for n in n_values:
        os.environ["CHAKUHO_ESTIMATOR"] = "sampling"
        os.environ["CHAKUHO_SAMPLES"] = str(n)
        agree_by_bucket: dict[str, list[int]] = {"small(<=20)": [0, 0], "medium(21-52)": [0, 0], "tournament(>52)": [0, 0]}
        lat_ms: list[float] = []
        coverages: list[float] = []
        errors = 0
        t_sweep0 = time.perf_counter()
        for i, r in enumerate(rows):
            qname, q = r["_qname"], r["request"]["questions"][r["_qname"]]
            b = bucket_of(r["_n_opts"])
            try:
                out = core.evaluate(r["request"]["state"], {qname: q}, backend_url=backend_url, model=model)
            except core.BackendError as exc:
                errors += 1
                print(f"  n={n} [{i}] backend error: {exc}", file=sys.stderr)
                continue
            a = out["answers"][qname]
            lat_ms.append(out["latency_ms"])
            coverages.append(a.get("coverage", 0.0))
            logged = r["response"]["answers"].get(qname, {})
            agree_by_bucket[b][1] += 1
            agree_by_bucket[b][0] += int(a.get("choice") == logged.get("choice"))
            if (i + 1) % 10 == 0:
                print(f"  n={n}: {i+1}/{len(rows)}", flush=True)
        dt_sweep = time.perf_counter() - t_sweep0
        total_ok = sum(v[0] for v in agree_by_bucket.values())
        total_n = sum(v[1] for v in agree_by_bucket.values())
        lat_sorted = sorted(lat_ms)
        p90 = lat_sorted[int(len(lat_sorted) * 0.9) - 1] if lat_sorted else None
        result[str(n)] = {
            "agree_total": f"{total_ok}/{total_n}",
            "agree_pct": round(100 * total_ok / max(1, total_n), 1),
            "agree_by_bucket": {k: f"{v[0]}/{v[1]} ({100*v[0]/max(1,v[1]):.0f}%)" for k, v in agree_by_bucket.items()},
            "latency_ms_median": round(st.median(lat_ms)) if lat_ms else None,
            "latency_ms_p90": p90,
            "coverage_median": round(st.median(coverages), 3) if coverages else None,
            "errors": errors,
            "wall_sec": round(dt_sweep, 1),
        }
        print(f"== n={n}: agree {total_ok}/{total_n} ({100*total_ok/max(1,total_n):.1f}%) "
              f"latency median {result[str(n)]['latency_ms_median']}ms p90 {p90}ms "
              f"coverage median {result[str(n)]['coverage_median']} errors={errors} "
              f"wall={dt_sweep:.0f}s", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--n", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--per-bucket", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths: list[str] = []
    for pattern in args.logs:
        paths.extend(sorted(glob.glob(os.path.expanduser(pattern))))
    rows = load_choice_records(paths)
    if not rows:
        print("対象が 0 件", file=sys.stderr)
        raise SystemExit(2)
    sample = stratified_sample(rows, args.per_bucket)
    result = run_sweep(sample, args.backend, args.n)
    if args.out:
        Path(args.out).write_text(json.dumps({
            "backend": args.backend, "per_bucket": args.per_bucket, "n_values": args.n,
            "sample_size": len(sample), "result": result,
        }, ensure_ascii=False, indent=2))
        print(f"書き出し: {args.out}")


if __name__ == "__main__":
    main()
