import os
"""小型モデル vs 27B(chakuho 本番)の一致率。判定ログ + 合成 GUI ケースを、ollama ネイティブ API 経由で chakuho と同じプロンプトで流す。"""
import glob, json, math, sys, time, urllib.request, statistics as st
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from chakuho import core

OLLAMA = "http://localhost:11434/api/chat"
def ollama_backend(model):
    def q(prompt, backend_url, m, *, timeout=120):
        body = {"model": model, "messages": [{"role": "system", "content": core.SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                "stream": False, "think": False, "logprobs": True, "top_logprobs": 20, "options": {"num_predict": 1, "temperature": 0}}
        req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=timeout))
        top = {e["token"]: e["logprob"] for e in d["logprobs"][0]["top_logprobs"]}
        return top, d.get("prompt_eval_count", 0)
    return q

# --- データ: 判定ログ(27B 本番の答え付き)
cases = []; seen = set()
for f in sorted(glob.glob(os.path.expanduser(os.environ.get("CHAKUHO_LOG_DIR", "~/.ato/chakuho")) + "/decisions-*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        if r["status"] == 200 and "answers" in r["response"]:
            st_ = r["request"].get("state")
            if isinstance(st_, dict) and st_.get("app") in ("Demo", "Calendar"):
                continue  # lat_bench の合成要素と、下で改めて作る GUI ケースの重複を除く
            key = json.dumps(r["request"], sort_keys=True, ensure_ascii=False)
            if key in seen: continue
            seen.add(key)
            cases.append({"src": "log", "request": r["request"], "teacher": r["response"]["answers"]})
# --- 合成 GUI ケース(教師は本番 chakuho :9750 の 27B)
AX = ["[AXButton] Today (toolbar)", "[AXButton] Day", "[AXButton] Week", "[AXButton] Month", "[AXTextField] Search",
      "[AXButton] Add (toolbar, +)", "[AXButton] Previous month", "[AXButton] Next month", "[AXCheckBox] Home (calendar list)",
      "[AXCheckBox] Work (calendar list)", "[AXTextField] New Event (title, focused)", "[AXTextField] Add Location or Video Call",
      "[AXPopUpButton] All-day", "[AXDateField] starts 2026-09-20 10:00", "[AXDateField] ends 2026-09-20 11:00",
      "[AXPopUpButton] Repeat: Never", "[AXPopUpButton] Alert: None", "[AXPopUpButton] Calendar: Home", "[AXTextField] Add Invitees",
      "[AXTextField] Add Notes, URL, or Attachments", "[AXButton] Cancel", "[AXButton] Add (New Event sheet)", "[AXButton] Delete Event",
      "[AXMenuButton] File", "[AXMenuButton] Edit", "[AXButton] Close window", "[AXButton] Minimize", "__none__"]
crit = {e: ("どの要素も指示に合わない(該当なし)" if e == "__none__" else e) for e in AX}
steps = ["イベントのタイトルを入力する欄をクリック", "イベントを確定して保存する", "新しいイベントの作成を開始する", "開始日時を変更する",
         "やめてダイアログを閉じる", "このイベントを削除する", "検索ボックスを使う", "月表示に切り替える", "参加者を追加する",
         "クワァシャーワプを押す(架空)", "ウィンドウを閉じる", "通知の設定を変える"]
for s in steps:
    req = {"state": {"app": "Calendar", "window": "Calendar — September 2026", "focused": "[AXTextField] New Event"},
           "questions": {"target": {"type": "choice", "instructions": f"Mac の GUI を AX 木で操作する。今のステップ: {s}。操作する要素を 1 つ選ぶ", "criteria": crit},
                         "risk": {"type": "score", "instructions": "これから実行する操作の危険度", "criteria": ["安全", "要注意", "破壊的"]},
                         "done": {"type": "noul", "instructions": "目標は既に達成済みか", "criteria": {"true": "達成済み", "false": "未達成"}}}}
    body = json.dumps(req).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:9750/v1/systemone", data=body, headers={"Content-Type": "application/json"}), timeout=120))
    cases.append({"src": "gui", "request": req, "teacher": r["answers"]})
print(f"cases: {len(cases)} (log {sum(c['src']=='log' for c in cases)}, gui {sum(c['src']=='gui' for c in cases)})", flush=True)

if __name__ == "__main__":
    models = sys.argv[1:] or ["qwen3:4b-instruct-2507-q4_K_M", "qwen3:8b", "qwen3.8:27b"]
    for model in models:
        core.query_backend = ollama_backend(model)
        agree = {"log": [0, 0], "gui": [0, 0]}; cov = []; noul_d = []; score_d = []; lat = []
        for c in cases:
            t0 = time.time()
            try:
                out = core.evaluate(c["request"]["state"], c["request"]["questions"], model=model)
            except Exception as e:
                print("ERR", model, e); continue
            lat.append(time.time() - t0)
            for name, ans in out["answers"].items():
                t = c["teacher"][name]
                cov.append(ans["coverage"])
                if "choice" in ans:
                    agree[c["src"]][1] += 1; agree[c["src"]][0] += (ans["choice"] == t["choice"])
                elif "noul" in ans: noul_d.append(abs(ans["noul"] - t["noul"]))
                elif "score" in ans: score_d.append(abs(ans["score"] - t["score"]))
        def pct(a): return f"{a[0]}/{a[1]} ({100*a[0]/max(1,a[1]):.0f}%)"
        covs = sorted(cov)
        print(f"\n== {model}\n  choice 一致(log/mario 9択): {pct(agree['log'])}   choice 一致(gui 28択): {pct(agree['gui'])}\n"
              f"  coverage: median {covs[len(covs)//2]:.2f}  p10 {covs[len(covs)//10]:.2f}  min {covs[0]:.2f}  (<0.5: {sum(1 for x in cov if x<0.5)}/{len(cov)})\n"
              f"  noul |Δp| mean {st.mean(noul_d):.2f} (n={len(noul_d)})   score |Δ| mean {st.mean(score_d):.2f} (n={len(score_d)})\n"
              f"  latency/request median {st.median(lat):.2f}s (ollama、並列なし)", flush=True)
