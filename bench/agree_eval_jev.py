"""Jev を基準線に、chakuho 27B(vLLM 本番)・27B(ollama)・8B・4B の一致率を出す。"""
import glob, json, sys, time, urllib.request, statistics as st
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1])); sys.path.insert(0, ".")
from chakuho import core
import jev_gateway
from agree_eval import cases, ollama_backend  # 61 件(log 49 + gui 12)。gui の teacher は本番 27B
# NOTE: agree_eval を import すると本体が走るので、__main__ ガードが無い → 先に実行部を切り出す

# 1) Jev の答え(基準線)
import pathlib
CACHE = pathlib.Path("jev_answers.json"); saved = json.loads(CACHE.read_text()) if CACHE.exists() else {}
jev = []; lat = []
for c in cases:
    k = json.dumps(c["request"], sort_keys=True, ensure_ascii=False)
    if k not in saved:
        r = jev_gateway.evaluate(c["request"]["state"], c["request"]["questions"])
        saved[k] = {"answers": r["answers"], "latency_ms": r["latency_ms"]}
        CACHE.write_text(json.dumps(saved, ensure_ascii=False))  # 途中で 429 に止められても再開できる
    jev.append(saved[k]["answers"]); lat.append(saved[k]["latency_ms"])
print(f"Jev: {len(jev)} cases, latency median {st.median(lat):.0f}ms p95 {sorted(lat)[int(len(lat)*0.95)-1]}ms", flush=True)

def compare(name, get_answers):
    agree = {"log": [0, 0], "gui": [0, 0]}; nd = []; sd = []
    for c, j in zip(cases, jev):
        a = get_answers(c)
        if a is None: continue
        for qn, ja in j.items():
            x = a[qn]
            if "choice" in ja: agree[c["src"]][1] += 1; agree[c["src"]][0] += (x["choice"] == ja["choice"])
            elif "noul" in ja: nd.append(abs(x["noul"] - ja["noul"]))
            elif "score" in ja: sd.append(abs(x["score"] - ja["score"]))
    pct = lambda a: f"{a[0]}/{a[1]} ({100*a[0]/max(1,a[1]):.0f}%)"
    print(f"  {name:28s} choice一致 mario {pct(agree['log'])}  gui {pct(agree['gui'])}  noul|Δ| {st.mean(nd) if nd else 0:.2f}  score|Δ| {st.mean(sd) if sd else 0:.2f}", flush=True)

print("== vs Jev(基準線)")
compare("chakuho 27B (vLLM 本番)", lambda c: c["teacher"])
for model in ["qwen3.8:27b", "qwen3:8b", "qwen3:4b-instruct-2507-q4_K_M"]:
    core.query_backend = ollama_backend(model)
    cache = {}
    def get(c, model=model, cache=cache):
        k = id(c)
        if k not in cache:
            try: cache[k] = core.evaluate(c["request"]["state"], c["request"]["questions"], model=model)["answers"]
            except Exception as e: print("ERR", model, e); cache[k] = None
        return cache[k]
    compare(f"{model} (ollama)", get)
# Jev の分布の様子(coverage 相当の情報は無いので、最尤の確率を見る)
tops = [max(a[q]["probabilities"].values()) for a in jev for q in a if "choice" in a[q]]
print(f"Jev choice の最尤確率: median {st.median(tops):.2f}, <0.5 が {sum(1 for t in tops if t<0.5)}/{len(tops)}")
