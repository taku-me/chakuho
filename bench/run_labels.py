"""258 ケースを Jev / chakuho 27B / 8B / 4B に流し、作者期待答えとの正答率と相互一致率を出す(途中再開可)。"""
import json, sys, time, re, statistics as st, urllib.request
sys.path.insert(0, ".."); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from chakuho import core
import jev_gateway
from agree_eval import ollama_backend
import os
CASES = os.environ.get("CASES", "cases.json"); cases = json.load(open(CASES)); SUF = "" if CASES == "cases.json" else "-" + CASES.replace(".json", "")
NONE = "__none__"

def via_chakuho(req):
    body = json.dumps(req).encode()
    d = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:9750/v1/systemone", data=body, headers={"Content-Type": "application/json"}), timeout=300))
    return d["answers"]["target"]
def via_jev(req): return jev_gateway.evaluate(req["state"], req["questions"])["answers"]["target"]
def via_ollama(model):
    core.query_backend = ollama_backend(model)
    return lambda req: core.evaluate(req["state"], req["questions"], model=model)["answers"]["target"]

LABELERS = {"jev": via_jev, "chakuho-27B": via_chakuho, "qwen3-8b": via_ollama("qwen3:8b"), "qwen3-4b": via_ollama("qwen3:4b-instruct-2507-q4_K_M")}
only = sys.argv[1:] or list(LABELERS)
for name in only:
    fn = LABELERS[name]; path = f"labels-{name}{SUF}.json"
    try: saved = json.load(open(path))
    except FileNotFoundError: saved = {}
    t0 = time.time(); n = 0
    for c in cases:
        if c["case_id"] in saved: continue
        try:
            a = fn(c["request"]); saved[c["case_id"]] = {"choice": a["choice"], "p": max(a["probabilities"].values()), "coverage": a.get("coverage")}
        except Exception as e:
            saved[c["case_id"]] = {"error": str(e)[:200]}
        n += 1
        if n % 20 == 0: json.dump(saved, open(path, "w"), ensure_ascii=False)
    json.dump(saved, open(path, "w"), ensure_ascii=False)
    print(f"{name}: labeled {n} new in {time.time()-t0:.0f}s, errors {sum(1 for v in saved.values() if 'error' in v)}", flush=True)

def rl(choice):  # "12: [AXButton] Settings" -> "AXButton:Settings"
    if choice == NONE: return NONE
    m = re.match(r"^\d+: \[(\w+)\] (.*)$", choice); return f"{m.group(1)}:{m.group(2)}" if m else choice
labels = {name: json.load(open(f"labels-{name}{SUF}.json")) for name in only}
print("\n== 作者期待答えに対する正答率(258 ケース)")
for name in only:
    ok = {}; tot = {}
    for c in cases:
        v = labels[name].get(c["case_id"], {}); hit = ("choice" in v) and (rl(v["choice"]) in c["expected"])
        for k in ("all", c["variant"], c["app"], "none-tasks" if c["expected"] == [NONE] else "has-target"):
            tot[k] = tot.get(k, 0) + 1; ok[k] = ok.get(k, 0) + hit
    f = lambda k: f"{ok.get(k,0)}/{tot.get(k,0)} ({100*ok.get(k,0)/max(1,tot.get(k,0)):.0f}%)"
    print(f"  {name:12s} all {f('all')} | orig {f('orig')} shuffled {f('shuffled')} distractors {f('distractors')} | has-target {f('has-target')} none-tasks {f('none-tasks')}")
    print("      per app: " + ", ".join(f"{a} {f(a)}" for a in sorted(set(c['app'] for c in cases))))
print("\n== 相互一致率(choice が同じ要素、role:label で比較)")
names = only
for i, a in enumerate(names):
    for b in names[i+1:]:
        pairs = [(labels[a].get(c["case_id"], {}), labels[b].get(c["case_id"], {})) for c in cases]
        n = sum(1 for x, y in pairs if "choice" in x and "choice" in y); m = sum(1 for x, y in pairs if "choice" in x and "choice" in y and rl(x["choice"]) == rl(y["choice"]))
        print(f"  {a} vs {b}: {m}/{n} ({100*m/max(1,n):.0f}%)")
