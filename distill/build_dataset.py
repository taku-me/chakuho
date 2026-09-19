"""②③ リクエスト合成 + 教師ラベル: gen-*.json と AX スナップショットから 1 ラウンド(52 択以下)の教師データを作る。

使い方:
  python3 -m distill.build_dataset --gen <dir> --snapshots <snapshot.json ...> --out <dataset.jsonl>
      [--backend http://localhost:8006/v1] [--variants 3] [--max-options 40] [--inflight 4] [--log-dir ~/.ato/chakuho]

1 行 1 例:
  {"id": str, "kind": "choice"|"noul", "system": str, "prompt": str, "labels": [str], "teacher": {label: p},
   "coverage": float, "max_p": float, "meta": {"app", "instruction", "target_id"|None, "variant", "options": [option...]}}
学生は system + prompt をそのまま入力し、labels 上の分布を teacher に近づける(KL)。

除外するのは coverage < 0.5 だけ(答えの形を守れなかった応答)。迷い(max_p が低い)は保持する。
共用 vLLM を守るサーキットブレーカー: chakuho の判定ログ(他の呼び出し元の latency)を 60 秒ごとに読み、
中央値が開始前の 2 倍を超えたら一時停止、1.5 倍未満で再開、10 回続いたら中断。BackendError 3 連続 / 1 判定 30 秒超は即停止。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import os
import random
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chakuho import core  # noqa: E402

NONE = core.NONE_OPTION
RISK_WORDS = ("削除", "破棄", "捨て", "送信", "上書き", "消去", "リセット", "デフォルトに戻す", "ゴミ箱", "アンインストール", "初期化", "Delete", "Remove", "Discard")


class Breaker:
    """共用 backend の負荷を chakuho 判定ログから監視し、閾値超えで pause() を True にする。"""

    def __init__(self, log_dir: Path, baseline_ms: float | None = None) -> None:
        self.log_dir = log_dir
        self.baseline = baseline_ms or self._recent_median(50, since=None) or 1000.0
        self.paused = threading.Event()
        self.pause_count = 0
        self.consecutive_errors = 0
        self.abort = False
        self._lock = threading.Lock()

    def _recent_median(self, n: int, since: float | None) -> float | None:
        vals: list[float] = []
        for f in sorted(glob.glob(str(self.log_dir / "decisions-*.jsonl")))[-2:]:
            for line in open(f):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if since is not None and d.get("t", 0) < since:
                    continue
                lat = (d.get("response") or {}).get("latency_ms")
                if isinstance(lat, (int, float)):
                    vals.append(float(lat))
        vals = vals[-n:]
        return statistics.median(vals) if vals else None

    def tick(self) -> str:
        med = self._recent_median(20, since=time.time() - 120)
        if med is None:
            self._resume("no-traffic"); return "no-traffic"
        if med > 2 * self.baseline:
            self._pause(f"latency {med:.0f}ms > 2x{self.baseline:.0f}"); return "paused"
        if med < 1.5 * self.baseline:
            self._resume(f"latency {med:.0f}ms")
        return "ok"

    def _pause(self, why: str) -> None:
        with self._lock:
            if not self.paused.is_set():
                self.pause_count += 1
                print(f"[breaker] 一時停止 #{self.pause_count}: {why}", flush=True)
                if self.pause_count >= 10:
                    self.abort = True
                    print("[breaker] 10 回連続で負荷が下がらないので中断", flush=True)
            self.paused.set()

    def _resume(self, why: str) -> None:
        with self._lock:
            if self.paused.is_set():
                print(f"[breaker] 再開: {why}", flush=True)
            self.paused.clear()

    def note_result(self, ok: bool, seconds: float) -> None:
        with self._lock:
            self.consecutive_errors = 0 if ok else self.consecutive_errors + 1
            if self.consecutive_errors >= 3:
                self._pause("BackendError 3 連続")
            elif seconds > 30:
                self._pause(f"1 判定 {seconds:.0f}s")

    def wait(self) -> None:
        while self.paused.is_set() and not self.abort:
            time.sleep(5)


def load_screens(snapshots: list[str]) -> dict[str, list[dict]]:
    screens: dict[str, list[dict]] = {}
    for s in snapshots:
        screens.update(json.load(open(s))["apps"])
    return screens


def option_key(e: dict) -> str:
    return f"{e['id']}: [{e['role']}] {e.get('label') or ''}"


def option_desc(e: dict) -> str:
    return f"window: {e.get('window') or '-'}" + (f"; in: {e['in']}" if e.get("in") else "")


def make_choice_example(rng: random.Random, app: str, elements: list[dict], others: list[tuple[str, dict]],
                        instruction: str, target_id: int | None, variant: str, max_options: int) -> dict:
    by_id = {e["id"]: e for e in elements}
    pool = [e for e in elements if e["id"] != target_id]
    n = min(max_options - 1, len(pool))
    if variant == "distractors":
        extra = rng.sample(others, min(10, len(others)))
        pool = pool + [dict(e, id=9000 + j, window=f"(別アプリ {a}) {e.get('window') or ''}") for j, (a, e) in enumerate(extra)]
    chosen = rng.sample(pool, min(n, len(pool)))
    if target_id is not None:
        chosen.append(by_id[target_id])
    if variant != "orig":
        rng.shuffle(chosen)
    else:
        chosen.sort(key=lambda e: e["id"])
    criteria = {option_key(e): option_desc(e) for e in chosen}
    criteria[NONE] = "どの要素も指示に合わない(該当なし)"
    return {"kind": "choice", "state": {"app": app, "window": elements[0].get("window") or app,
                                        "note": "要素一覧は候補側に列挙されている。説明の in: はシート/ダイアログ内の要素"},
            "instructions": f"Mac のアプリを AX 木(アクセシビリティ要素)経由で操作する。今のステップ: {instruction}。操作すべき要素を 1 つ選ぶ。該当する要素が無ければ {NONE}。",
            "criteria": criteria, "expected": option_key(by_id[target_id]) if target_id is not None else NONE}


def make_noul_examples(app: str, elements: list[dict], instruction: str, target: dict | None) -> list[dict]:
    out = []
    screen = "\n".join(option_key(e) for e in elements[:60])
    if target is not None:
        out.append({"kind": "noul", "name": "risk",
                    "state": {"app": app, "planned_action": f"{option_key(target)} を操作する", "instruction": instruction, "screen": screen},
                    "instructions": "この操作は取り消しにくい、または損失を生みうる(削除・破棄・送信・上書き・設定の初期化など)か。",
                    "criteria": {"true": "破壊的・不可逆", "false": "安全・可逆"},
                    "expected": any(w in (target.get("label") or "") + instruction for w in RISK_WORDS)})
    out.append({"kind": "noul", "name": "done",
                "state": {"app": app, "instruction": instruction, "screen": screen},
                "instructions": "画面の要素一覧を見て、この指示が既に完了している(これ以上の操作が不要)と判断できるか。",
                "criteria": {"true": "完了している", "false": "まだ操作が必要"}, "expected": None})
    return out


def teacher_label(ex: dict, backend: str, model: str) -> dict:
    """chakuho core と同じ prompt を組み、1 ラウンドで教師分布を取る。"""
    if ex["kind"] == "choice":
        options = list(ex["criteria"].keys())
        labels = core.labels_for(len(options))
        menu = "\n".join(f"{l}: {o} — {ex['criteria'][o]}" for l, o in zip(labels, options))
        prompt = core._build_prompt(core._render(ex["instructions"]), menu, core._render(ex["state"]), "choose one option.", labels)
        ci = len(labels) <= 26
    else:
        labels = ["yes", "no"]
        menu = f"yes: {ex['criteria']['true']}\nno: {ex['criteria']['false']}"
        prompt = core._build_prompt(core._render(ex["instructions"]), menu, core._render(ex["state"]), "yes or no.", labels)
        options = labels; ci = True
    top, _ = core.query_backend(prompt, backend, model)
    dist, coverage = core.aggregate(top, labels, case_insensitive=ci)
    return {"system": core.SYSTEM_PROMPT, "prompt": prompt, "labels": labels, "teacher": dist,
            "coverage": coverage, "max_p": max(dist.values()), "options": options}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--snapshots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default=core.DEFAULT_BACKEND_URL)
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--max-options", type=int, default=40)
    ap.add_argument("--inflight", type=int, default=4)
    ap.add_argument("--log-dir", default=os.environ.get("CHAKUHO_LOG_DIR", str(Path.home() / ".ato" / "chakuho")))
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--limit", type=int, default=0, help="デバッグ用: 例の上限")
    args = ap.parse_args()

    screens = load_screens(args.snapshots)
    rng = random.Random(args.seed)
    others_all = [(a, e) for a, es in screens.items() for e in es]
    examples: list[dict] = []
    stats = {"screens": 0, "elements": 0, "instructions": 0, "none_instructions": 0, "variants": args.variants, "choice": 0, "noul": 0}
    for gf in sorted(glob.glob(str(Path(args.gen) / "gen-*.json"))):
        g = json.load(open(gf)); app = g["app"]
        if app not in screens:
            print(f"skip {gf}: 画面 {app} がスナップショットに無い", flush=True); continue
        elements = screens[app]; by_id = {e["id"]: e for e in elements}
        others = [(a, e) for a, e in others_all if a != app]
        stats["screens"] += 1; stats["elements"] += len(elements)
        variants = ["orig", "shuffled", "distractors"][:args.variants]
        for t in g["tasks"]:
            for ins in t["instructions"]:
                stats["instructions"] += 1
                for v in variants:
                    ex = make_choice_example(rng, app, elements, others, ins, t["id"], v, args.max_options)
                    ex["meta"] = {"app": app, "instruction": ins, "target_id": t["id"], "variant": v}
                    examples.append(ex)
                for ex in make_noul_examples(app, elements, ins, by_id[t["id"]]):
                    ex["meta"] = {"app": app, "instruction": ins, "target_id": t["id"], "variant": ex["name"]}
                    examples.append(ex)
        for ins in g["none"]:
            stats["none_instructions"] += 1
            for v in variants:
                ex = make_choice_example(rng, app, elements, others, ins, None, v, args.max_options)
                ex["meta"] = {"app": app, "instruction": ins, "target_id": None, "variant": v}
                examples.append(ex)
    if args.limit:
        rng.shuffle(examples); examples = examples[:args.limit]
    print(f"合成: {len(examples)} 例  {json.dumps(stats, ensure_ascii=False)}", flush=True)

    core.configure_inflight(args.inflight)
    model = core.resolve_model(args.backend)
    breaker = Breaker(Path(args.log_dir))
    print(f"[breaker] baseline latency {breaker.baseline:.0f}ms(chakuho ログの直近中央値)", flush=True)
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            breaker.tick(); stop.wait(60)
    threading.Thread(target=monitor, daemon=True).start()

    def label(idx_ex):
        idx, ex = idx_ex
        breaker.wait()
        if breaker.abort:
            return None
        t0 = time.time()
        try:
            r = teacher_label(ex, args.backend, model)
            breaker.note_result(True, time.time() - t0)
        except core.BackendError as exc:
            breaker.note_result(False, time.time() - t0)
            return {"error": str(exc), "idx": idx}
        return {"id": f"{ex['meta']['app']}#{idx}", "kind": ex["kind"], **r,
                "meta": {**ex["meta"], "options": r.pop("options"), "expected": ex.get("expected")}}

    kept = dropped_cov = errors = low_conf = 0
    t0 = time.time()
    with open(args.out, "w") as fo, cf.ThreadPoolExecutor(args.inflight) as ex:
        for i, rec in enumerate(ex.map(label, enumerate(examples))):
            if rec is None:
                break
            if "error" in rec:
                errors += 1; continue
            if rec["coverage"] < 0.5:
                dropped_cov += 1; continue
            if rec["max_p"] < 0.4:
                low_conf += 1
            stats[rec["kind"]] += 1
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); kept += 1
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(examples)} 済 ({time.time()-t0:.0f}s)", flush=True)
    stop.set()
    print(json.dumps({"kept": kept, "dropped_coverage_lt_0.5": dropped_cov, "kept_low_confidence_max_p_lt_0.4": low_conf,
                      "backend_errors": errors, "aborted": breaker.abort, "breaker_pauses": breaker.pause_count, **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
