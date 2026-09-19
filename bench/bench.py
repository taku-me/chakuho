import json, time, urllib.request, sys
sys.path.insert(0, ".")
from jev_local import noul, choice, score, ask_many, HOST, MODEL

STATE = """[terminal screen: worker-1, captured 2026-09-19 17:40]
$ python3 ops/health_check.py
[health] db: ok (0.4s)
[health] storage: ok
[health] inference server: connection refused (port 8007)
[health] queue relay: last apply 2026-09-19 03:12 (14h ago)
Traceback (most recent call last):
  File "ops/health_check.py", line 88, in <module>
    ModuleNotFoundError: No module named 'recovery_helpers'
Not logged in · Please run /login
$ """

QS = [
    ("noul", "Is this lane currently blocked by a live authentication failure (as opposed to old history text)?"),
    ("noul", "Does the screen show a Python import error caused by a moved or missing module?"),
    ("noul", "Is a human decision required before this lane can continue?"),
    ("choice", "What is the single most urgent issue on this screen?", ["authentication expired", "missing python module", "inference server down", "queue relay stale", "nothing urgent"]),
    ("choice", "Which lane should handle the follow-up?", ["this worker itself", "the coordinator", "user", "nobody"]),
    ("score", "How severe is the overall state of this lane?"),
]

# --- Jev-style: sequential
t0 = time.time(); seq = [ {"noul":noul,"choice":choice,"score":score}[q[0]](STATE, *q[1:]) for q in QS ]; t_seq = time.time()-t0
# --- Jev-style: parallel
t0 = time.time(); par = ask_many(STATE, QS); t_par = time.time()-t0

print("=== Jev-style results (parallel run) ===")
for q, r in zip(QS, par):
    slim = {k: v for k, v in r.items() if k in ("p_yes","choice","confidence","score","coverage","dist")}
    print(f"- {q[1][:70]}\n    {json.dumps(slim, ensure_ascii=False)}  [prompt_eval {r['prompt_eval_ms']:.0f}ms, total {r['total_ms']:.0f}ms]")
print(f"\nsequential 6 questions: {t_seq:.2f}s   parallel 6 questions: {t_par:.2f}s")

# --- conventional: thinking off, generate JSON with all 6 answers
prompt = STATE + "\n\nAnswer the following as a JSON object with keys q1..q6. q1-q3 yes/no, q4 one of [authentication expired, missing python module, inference server down, queue relay stale, nothing urgent], q5 one of [this worker itself, the coordinator, user, nobody], q6 integer 1-5 severity.\n" + "\n".join(f"q{i+1}: {q[1]}" for i,q in enumerate(QS))
body = {"model": MODEL, "messages":[{"role":"user","content":prompt}], "stream": False, "think": False, "format":"json", "options":{"temperature":0}}
req = urllib.request.Request(f"{HOST}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
t0=time.time(); d=json.load(urllib.request.urlopen(req, timeout=600)); t_conv=time.time()-t0
print(f"\n=== conventional (think off, JSON generation) ===\n{d['message']['content'].strip()}\n eval_count={d.get('eval_count')} tokens, wall {t_conv:.2f}s")

# --- conventional with thinking on
body["think"]=True
req = urllib.request.Request(f"{HOST}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
t0=time.time(); d=json.load(urllib.request.urlopen(req, timeout=900)); t_think=time.time()-t0
print(f"\n=== conventional (think ON) ===\n{d['message']['content'].strip()[:400]}\n eval_count={d.get('eval_count')} tokens (incl. thinking), wall {t_think:.2f}s")
