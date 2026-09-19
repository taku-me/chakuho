"""① 指示文の逆生成: マスク済み AX スナップショットの各要素に対し、27B(生成モード)で日本語の指示文を作る。

使い方:
  python3 -m distill.gen_instructions <snapshot.json ...> --out <dir> [--backend http://localhost:8006/v1] [--per-element 3] [--none 5]

出力: <dir>/gen-<画面名>.json = {"app": ..., "tasks": [{"id": int, "instructions": [str, ...]}], "none": [str, ...]}
存在しない id や空文字は捨てる(捨てた件数を出力する)。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chakuho import core  # noqa: E402

ACTIONABLE = {"AXButton", "AXMenuItem", "AXMenuBarItem", "AXTextField", "AXTextArea", "AXCheckBox", "AXPopUpButton",
              "AXRadioButton", "AXTab", "AXLink", "AXDisclosureTriangle", "AXRow", "AXCell", "AXUnknown", "AXSlider",
              "AXComboBox", "AXTabGroup", "AXMenuButton", "AXImage", "AXToolbar", "AXOutline", "AXTable",
              "AXStaticText", "AXGroup"}  # Web/Electron 系アプリはクリック対象が StaticText/Group で露出する(実画面ベンチの正解 57/238 件がこの 2 つ)
STYLES = "丁寧な依頼 / ぶっきらぼうな命令 / 目的だけを言う(要素名を言わない)"

PROMPT = """あなたは macOS のアプリを操作するエージェントの評価データを作っている。
下の「画面の要素一覧」は、アプリ「{app}」の画面(アクセシビリティ木)から取ったものだ。
各要素について、その要素を操作したくなる自然な日本語の指示文を {k} 通り作れ。
言い回しは {styles} の 3 種類を混ぜ、要素のラベル文字列をそのまま繰り返さない指示も含めること。
「in: sheet」等の印がある要素は、シートやダイアログの中にある。
最後に、この画面の要素ではどうやっても実行できない指示を {n_none} 個作れ(別の画面や別アプリでしか出来ないこと)。

出力は JSON だけ。説明を書くな。形式:
{{"tasks": [{{"id": <要素id>, "instructions": ["...", "..."]}}, ...], "none": ["...", ...]}}

画面の要素一覧:
{elements}
"""


def chat(backend: str, model: str, prompt: str, *, max_tokens: int = 6000, timeout: float = 600) -> str:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(backend.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in output")
    return json.loads(m.group(0))


def element_line(e: dict) -> str:
    extra = f" (in: {e['in']})" if e.get("in") else ""
    win = f" [window: {e['window']}]" if e.get("window") else ""
    return f"id={e['id']} [{e['role']}] {e.get('label') or ''}{extra}{win}"


def gen_screen(app: str, elements: list[dict], backend: str, model: str, k: int, n_none: int, chunk: int,
               max_elements: int = 0, seed: int = 0) -> tuple[dict, dict]:
    actionable = [e for e in elements if e.get("role") in ACTIONABLE and (e.get("label") or "").strip()]
    if max_elements and len(actionable) > max_elements:  # 巨大画面(例: 700 要素超)は抽出して重複と時間を抑える
        actionable = sorted(random.Random(f"{seed}:{app}").sample(actionable, max_elements), key=lambda e: e["id"])
    valid_ids = {e["id"] for e in actionable}
    chunks = [actionable[i:i + chunk] for i in range(0, len(actionable), chunk)]
    tasks: dict[int, list[str]] = {}
    none: list[str] = []
    dropped = {"bad_id": 0, "empty": 0, "parse_fail": 0}

    def one(idx_els):
        idx, els = idx_els
        prompt = PROMPT.format(app=app, k=k, styles=STYLES, n_none=n_none if idx == 0 else 0,
                               elements="\n".join(element_line(e) for e in els))
        for attempt in range(3):
            try:
                return parse_json(chat(backend, model, prompt))
            except (ValueError, json.JSONDecodeError):
                if attempt == 2:
                    return None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:  # 共用 backend の過負荷で timeout することがある
                print(f"  [{app}] chunk {idx}: backend error ({exc}); retry {attempt + 1}/3", flush=True)
                time.sleep(30 * (attempt + 1))
        return None

    with cf.ThreadPoolExecutor(4) as ex:
        for out in ex.map(one, enumerate(chunks)):
            if out is None:
                dropped["parse_fail"] += 1
                continue
            for t in out.get("tasks", []):
                try:
                    tid = int(t.get("id"))
                except (TypeError, ValueError):
                    dropped["bad_id"] += 1
                    continue
                if tid not in valid_ids:
                    dropped["bad_id"] += 1
                    continue
                ins = [s.strip() for s in t.get("instructions", []) if isinstance(s, str) and s.strip()]
                dropped["empty"] += len(t.get("instructions", [])) - len(ins)
                tasks.setdefault(tid, []).extend(ins)
            none.extend(s.strip() for s in out.get("none", []) if isinstance(s, str) and s.strip())
    return {"app": app, "n_elements": len(elements), "n_actionable": len(actionable),
            "tasks": [{"id": i, "instructions": v} for i, v in sorted(tasks.items())], "none": none[:n_none]}, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshots", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default=core.DEFAULT_BACKEND_URL)
    ap.add_argument("--per-element", type=int, default=3)
    ap.add_argument("--none", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=15, help="1 リクエストに入れる要素数")
    ap.add_argument("--max-elements", type=int, default=0, help="1 画面あたり生成対象にする操作対象要素の上限(0 = 全部)")
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--skip-existing", action="store_true", help="出力ファイルが既にある画面は飛ばす(中断からの再開用)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = core.resolve_model(args.backend)
    total = {"screens": 0, "actionable": 0, "instructions": 0, "none": 0}
    for snap in args.snapshots:
        apps = json.load(open(snap))["apps"]
        for app, elements in apps.items():
            t0 = time.time()
            safe = re.sub(r"[^\w\-]+", "_", app)[:60]
            if args.skip_existing and (out / f"gen-{safe}.json").exists():
                print(f"{app}: skip(既存)", flush=True); continue
            res, dropped = gen_screen(app, elements, args.backend, model, args.per_element, args.none, args.chunk,
                                      max_elements=args.max_elements, seed=args.seed)
            safe = re.sub(r"[^\w\-]+", "_", app)[:60]
            (out / f"gen-{safe}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
            n_ins = sum(len(t["instructions"]) for t in res["tasks"])
            total["screens"] += 1; total["actionable"] += res["n_actionable"]; total["instructions"] += n_ins; total["none"] += len(res["none"])
            print(f"{app}: 要素 {res['n_elements']} / 操作対象 {res['n_actionable']} / 指示文 {n_ins} / none {len(res['none'])} / 捨てた {dropped} / {time.time()-t0:.0f}s", flush=True)
    print("合計:", json.dumps(total, ensure_ascii=False))


if __name__ == "__main__":
    main()
