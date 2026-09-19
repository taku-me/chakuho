import json, time, urllib.request, statistics as st, random
URL="http://localhost:9750/v1/systemone"
roles=["AXButton","AXTextField","AXStaticText","AXMenuItem","AXCheckBox","AXPopUpButton","AXLink","AXImage"]
words=["Save","Open","Search","Settings","Delete","Cancel","Add","Edit","Share","Export","Import","Rename","Move","Copy","Paste","Undo","Redo","Zoom","Help","Close"]
def elements(n):
    random.seed(n)
    return [f"[{random.choice(roles)}] {random.choice(words)} {i} (pane {i%7})" for i in range(n)] + ["__none__"]
def run(n, with_extra, reps=8):
    lats=[]; stages=None
    for r in range(reps):
        els=elements(n); els[-2]=f"[AXButton] Save {n}{r} (toolbar)"  # 毎回少し変えて cache を避ける
        qs={"target":{"type":"choice","instructions":"Mac の GUI を AX 木で操作する。今のステップ: 保存ボタンを押す。操作する要素を 1 つ選ぶ","criteria":{e:e for e in els}}}
        if with_extra:
            qs["risk"]={"type":"score","instructions":"これから実行する操作の危険度","criteria":["安全","要注意","破壊的"]}
            qs["done"]={"type":"noul","instructions":"目標は既に達成済みか"}
        body=json.dumps({"state":{"app":"Demo","window":"Demo","focused":"none"},"questions":qs}).encode()
        t0=time.time(); d=json.load(urllib.request.urlopen(urllib.request.Request(URL,data=body,headers={"Content-Type":"application/json"}),timeout=300)); lats.append(time.time()-t0)
        stages=d["answers"]["target"]["stages"]; tok=d["usage"]["input_tokens"]
    lats.sort()
    print(f"elements={n:4d} questions={'3 (target+risk+done)' if with_extra else '1 (target)'}: p50 {lats[len(lats)//2]:.2f}s  p95 {lats[int(len(lats)*0.95)-1 if len(lats)>1 else 0]:.2f}s  max {lats[-1]:.2f}s  stages={stages}  input_tokens/req≈{tok}", flush=True)
print("bg vLLM running requests:", [l for l in urllib.request.urlopen("http://localhost:8006/metrics").read().decode().splitlines() if l.startswith("vllm:num_requests_running")][0].split()[-1])
for n in (8, 32, 100, 233):
    run(n, False); run(n, True)
