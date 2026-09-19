"""Vercel AI Gateway 経由で Jev を呼ぶ。chakuho/jev-mario の形式 → Gateway 形式へ変換し、答えを chakuho 互換に戻す。"""
import json, os, urllib.request, urllib.error, time
KEY = os.environ["VERCEL_AI_GATEWAY_KEY"]  # Vercel AI Gateway の API キー(環境変数で渡す)
URL = "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
HDR = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "ai-model-id": "typesafe-ai/jev",
       "ai-evaluation-model-specification-version": "4", "ai-gateway-protocol-version": "0.0.1"}

def to_gateway(questions):
    out = {}
    for name, q in questions.items():
        t = q["type"]; crit = q.get("criteria"); instr = q.get("instructions", "")
        if not isinstance(instr, str): instr = json.dumps(instr, ensure_ascii=False)
        if t == "choice":
            c = crit if isinstance(crit, dict) else {o: o for o in (crit or [])}
            out[name] = {"type": "choice", "instructions": instr, "criteria": c}
        elif t == "noul":
            g = {"type": "boolean", "instructions": instr}
            if isinstance(crit, dict): g["criteria"] = {"true": str(crit.get("true", "yes")), "false": str(crit.get("false", "no"))}
            out[name] = g
        elif t == "score":
            out[name] = {"type": "score", "instructions": instr, "criteria": [str(x) for x in crit]}
    return out

def evaluate(state, questions, retries=8):
    body = json.dumps({"state": state, "questions": to_gateway(questions)}).encode()
    for i in range(retries):
        try:
            t0 = time.time()
            with urllib.request.urlopen(urllib.request.Request(URL, data=body, headers=HDR), timeout=60) as r:
                d = json.load(r); lat = time.time() - t0
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:200]
            if e.code == 429 and i < retries - 1: time.sleep(30); continue  # 無料枠のレート制限: 30 秒空けて再試行
            if e.code in (500, 502, 503) and i < retries - 1: time.sleep(2 * (i + 1)); continue
            raise RuntimeError(f"gateway HTTP {e.code}: {msg}")
    answers = {}
    for name, a in d["answers"].items():
        q = questions[name]
        if a["type"] == "choice":
            answers[name] = {"choice": a["choice"], "probabilities": a["probabilities"]}
        elif a["type"] == "boolean":
            answers[name] = {"noul": a["probability"]}
        elif a["type"] == "score":
            n = len(q["criteria"]); levels = [str(x) for x in q["criteria"]]
            probs = {levels[int(k)]: v for k, v in a["probabilities"].items()}
            answers[name] = {"score": a["score"] / (n - 1) if n > 1 else 0.0, "score_raw": a["score"], "probabilities": probs}
    return {"answers": answers, "usage": {"input_tokens": d["usage"]["inputTokens"]}, "latency_ms": round(lat * 1000), "model": "typesafe-ai/jev"}

if __name__ == "__main__":
    # jev-mario 用の互換プロキシ: POST /v1/systemone を Gateway へ転送
    import sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9751
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            try:
                out = evaluate(req["state"], req["questions"]); code = 200
            except Exception as e:
                out, code = {"error": str(e)}, 502
            body = json.dumps(out).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            with open("jev_proxy.log.jsonl", "a") as fh: fh.write(json.dumps({"t": time.time(), "code": code, "resp": out}) + "\n")
    print(f"jev proxy on :{port} -> {URL}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
