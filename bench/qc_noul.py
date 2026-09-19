"""キャラクター正典 QC を「生成しない判定」で行う実験: チェックリストの各項目を noul(yes/no)として
参照画像 + 候補画像を付けた state で chakuho core に流し、項目ごとの確率と 1 枚あたりの所要時間を出す。

使い方:
  python3 bench/qc_noul.py --checklist <checklist.yaml> --ref <ref.png> --images <dir> --out <jsonl>
      [--backend http://localhost:8009/v1] [--model NAME] [--approved a.png,b.png] [--context-json ctx.json] [--limit N]

checklist.yaml は common.system / groups[].items[].{id, rule, exception} の 2 段構成(count_field 項目は
「count_pass の期待どおりか」を yes/no に読み替える)。--approved は人の承認済みファイル名(正解)。
出力の集計: 項目ごとの pass 率、画像ごとの全項目 pass(= QC 合格)と承認との一致、誤落とし(承認済みなのに不合格)件数、
1 枚あたりの所要時間(全項目を並列に投げた wall time)。
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import io
import json
import statistics
import sys
import time
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chakuho import core  # noqa: E402


def data_url(path: Path, max_side: int) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def items_of(checklist: dict) -> list[dict]:
    out = []
    for g in checklist["groups"]:
        for it in g["items"]:
            q = it["rule"].strip()
            if it.get("exception"):
                q += " Exception: " + it["exception"].strip()
            if it.get("count_field"):
                q = f"Count the {it['count_of']}. Is the count consistent with the canon for this view? Canon: {json.dumps(it['count_pass'])}. " + q
            out.append({"id": it["id"], "group": g["id"], "question": q})
    return out


def judge_image(path: Path, ref_url: str, items: list[dict], system: str, context: str, backend: str, model: str, max_side: int) -> dict:
    cand_url = data_url(path, max_side)
    state = {"role_of_images": "image 1 = official REFERENCE, image 2 = CANDIDATE to inspect", "scene_intent": context,
             "inspector_rules": system, "images": [ref_url, cand_url]}
    t0 = time.time()

    def one(it):
        r = core.noul(state, f"Checklist item '{it['id']}': does the CANDIDATE satisfy this? {it['question']}",
                      {"true": "satisfies the canon (pass)", "false": "violates the canon (fail)"}, backend_url=backend, model=model)
        return it["id"], r["noul"], r["coverage"]
    with cf.ThreadPoolExecutor(len(items)) as ex:
        res = list(ex.map(one, items))
    return {"file": path.name, "seconds": round(time.time() - t0, 2), "items": {i: {"p_pass": round(p, 3), "coverage": round(c, 3)} for i, p, c in res}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checklist", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--backend", default=core.DEFAULT_BACKEND_URL); ap.add_argument("--model")
    ap.add_argument("--approved", default=""); ap.add_argument("--context-json", help='{"stem_prefix": "scene intent", ...}')
    ap.add_argument("--max-side", type=int, default=512, help="画像の長辺。1.6B VL は 2 枚 × 768px だとエンコーダ予算(2048 トークン)を超えて永久に待たされた実測(512px は 0.24 s)"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5, help="p_pass がこれ未満なら不合格")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true", help="--out に既にある画像を飛ばし、追記する(backend 断で落ちた時の再開用)")
    args = ap.parse_args()
    checklist = yaml.safe_load(open(args.checklist)); items = items_of(checklist); system = checklist["common"]["system"].strip()
    ctx = json.load(open(args.context_json)) if args.context_json else {}
    model = args.model or core.resolve_model(args.backend)
    core.configure_inflight(32)
    ref_url = data_url(Path(args.ref), args.max_side)
    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if args.limit: paths = paths[: args.limit]
    approved = set(a for a in args.approved.split(",") if a)

    def ctx_for(name: str) -> str:
        for k, v in ctx.items():
            if name.startswith(k): return v
        return ctx.get("_default", "")
    rows = []
    if args.resume and Path(args.out).exists():
        rows = [json.loads(l) for l in open(args.out) if l.strip()]
        done = {r["file"] for r in rows}
        paths = [p for p in paths if p.name not in done]
        print(f"resume: 既存 {len(rows)} 枚、残り {len(paths)} 枚", flush=True)

    def judge_retry(p: Path) -> dict:
        for attempt in range(3):
            try:
                return judge_image(p, ref_url, items, system, ctx_for(p.stem), args.backend, model, args.max_side)
            except core.BackendError as exc:
                if attempt == 2: raise
                print(f"{p.name}: backend error ({exc}); retry {attempt + 1}/3", flush=True); time.sleep(30 * (attempt + 1))

    with open(args.out, "a" if args.resume else "w") as fo, cf.ThreadPoolExecutor(args.workers) as ex:
        for row in ex.map(judge_retry, paths):
            rows.append(row); fo.write(json.dumps(row, ensure_ascii=False) + "\n"); fo.flush()
            print(f"{row['file']}: {row['seconds']}s  fails={[i for i, v in row['items'].items() if v['p_pass'] < args.threshold]}", flush=True)
    secs = [r["seconds"] for r in rows]
    per_item = {it["id"]: statistics.mean(r["items"][it["id"]]["p_pass"] >= args.threshold for r in rows) for it in items}
    summary = {"model": model, "n_images": len(rows), "n_items": len(items), "seconds_median": statistics.median(secs), "seconds_p90": sorted(secs)[int(len(secs) * 0.9) - 1] if len(secs) >= 10 else max(secs),
               "pass_rate_per_item": {k: round(v, 2) for k, v in per_item.items()}, "coverage_lt_0_5": sum(v["coverage"] < 0.5 for r in rows for v in r["items"].values())}
    if approved:
        passed = {r["file"] for r in rows if all(v["p_pass"] >= args.threshold for v in r["items"].values())}
        names = {r["file"] for r in rows}
        summary["approved_in_set"] = len(approved & names); summary["qc_pass_total"] = len(passed)
        summary["false_reject(approved_but_failed)"] = sorted((approved & names) - passed)
        summary["approved_and_passed"] = len(approved & passed)
        summary["rejected_by_human_but_passed"] = len(passed - approved)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
