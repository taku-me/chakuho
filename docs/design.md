# chakuho(択法)— System One 型の判定エンジン

## 目的

Jev(TypeSafe)と同じ形の「生成しない判定」をローカル LLM で提供する。
入力は state(任意 JSON か文字列)と、答えの形を宣言した質問。出力は選択肢上の確率分布。
文章もコードも書かない。座標も出さない。候補は呼ぶ側が実行時に列挙する(AX 木・DOM・タスク一覧など)。

用途: mac-do の要素選択(現状は生成で十数秒 → 1 秒)、監視ログの異常トリアージ、端末画面の分類、
ルーティング、compaction のスコアリング。Jev 互換 API なので呼び出し側は将来 Jev 本体や別モデルに差し替えられる。

## 現状の実測(2026-09-19、DGX Spark、vLLM qwen3.8-27b NVFP4)

- 1 トークン logprob 判定: 1 判定 0.6〜1.3 秒(prefill 律速、入力 300〜1600 トークン)。並列投入は vLLM がバッチ化(6 問 1.13 秒)
- jev-mario を shim 経由で無改変実行: 直接操作 1-1 で x=401(Jev 686)。負け方は数値条件付きルールの誤適用
- AX 要素 32 個からの選択: 5 問中 4 正答、1 ステップ 1.17 秒(要素選択と完了判定を並列)
- ollama は logprobs が先頭 1 トークンのみ(MTP の副作用と推定)。backend は vLLM(OpenAI 互換 + logprobs)を前提にする
- qwen3.5 系ハイブリッドは vLLM の prefix cache ブロックが 1600 トークン。短い prompt では cache は効かない

## 設計

### 置き場

- 独立リポジトリ github.com/taku-me/chakuho。実行機では git checkout をそのまま systemd から起動する
- パッケージ `chakuho/`(`core.py` 判定本体 / `server.py` HTTP / `client.py` Python クライアント / `cli.py`)。依存は stdlib のみ、`pyproject.toml` の scripts で `chakuho` コマンドを出す
- CLI: `chakuho serve [--host 0.0.0.0 --port 9750]` と `chakuho ask <request.json | ->`(env `CHAKUHO_URL`)
- 起動: 運用側のリポジトリに置いた起動スクリプト + systemd user unit(`chakuho.service`)。このリポジトリには入れない
- 既存の大きなツール群には同梱しない。呼ぶ側がそれを持つ理由が無いため
- 設定: env `CHAKUHO_BACKEND_URL`(既定 `http://localhost:8006/v1`)、`CHAKUHO_MODEL`(未指定なら backend の `/models` 先頭を使う。呼び出し側にモデル ID を知らせない)、`CHAKUHO_LOG_DIR`(既定 `~/.ato/chakuho/`)

### API(Jev 互換部分)

`POST /v1/systemone`
```json
{"state": <json|string>,
 "questions": {"<name>": {"type": "choice|noul|score", "instructions": <string|dict>, "criteria": <dict|list>}}}
```
- `choice`: criteria は `{option: description}` か `[option,...]`。返り値 `{"choice": str, "probabilities": {option: p}, "coverage": p}`
- `noul`: criteria は `{"true": str, "false": str}`(省略可)。返り値 `{"noul": p_yes, "coverage": p}`
- `score`: criteria は段階のリスト(低→高)。返り値 `{"score": 0..1 に正規化した期待値, "probabilities": {level: p}, "coverage": p}`
- 共通: `{"answers": {...}, "usage": {"input_tokens": n}, "latency_ms": n, "model": str}`
- `coverage` = 上位 logprob のうち宣言ラベルに載った確率質量。0 なら一様分布を返し `"degraded": true` を付ける(「判定できなかった」を「判定した」と同じ語彙で返さない)

`GET /health` → `{"ok": true, "backend": url, "model": str}`(backend の `/models` が取れなければ `ok: false`, HTTP 503)

### 判定の仕組み

- 選択肢にラベル A-Z, a-z(最大 52、1 トークン)を振り、prompt は「instructions → options(ラベル: option — description) → state → question → `Label:`」の順。state を最後に置く(recency)
- backend へ `max_tokens=1, temperature=0, logprobs=true, top_logprobs=20, chat_template_kwargs.enable_thinking=false`
- 返った top_logprobs をラベルへ集約(strip + 大文字小文字を正規化)、ラベル内で正規化
- 同一リクエスト内の複数質問はスレッドで並列投入
- 53 個以上の選択肢は chakuho 内で 2 段トーナメント: 52 個ずつのチャンクへ分け(並列)、各チャンクの上位 k 個(k = max(1, 52 // チャンク数)。233 個なら 5 チャンク × 上位 10)を集めて決勝を 1 回。上位 1 個だけでなく上位 k 個を通すのは、強い候補が同じチャンクで潰し合う偏りを減らすため(レビュー指摘)。`__none__` があれば全チャンクと決勝に含める。返り値の `probabilities` は決勝の分布(落選したものは 0)、`stages: 2` を付ける。上限は 52×52=2704 で、超えたら HTTP 400。実測根拠: mac-do の Electron アプリで AX 要素 233 個(mac-do README)。事前絞り込みは呼ぶ側の任意最適化
- backend 呼び出しは timeout 60 秒。失敗は HTTP 503 と `{"error": ...}`。ハングさせない
- 全リクエストを `CHAKUHO_LOG_DIR/decisions-YYYYMMDD.jsonl` に日付別で追記(state 全文、質問、答え、latency)。較正データの元になる。起動時と日付切替時に `CHAKUHO_LOG_KEEP_DAYS`(既定 14)より古いファイルを削除する(ディスク枯渇対策、レビュー指摘)
- backend への同時リクエスト数はセマフォで上限 `CHAKUHO_MAX_INFLIGHT`(既定 16)。超えた分は待つ(vLLM の max-num-seqs 112 を chakuho 単独で埋めない)

### やらないこと

- 自由文・座標の生成(呼ぶ側が生成モデルか決定論 API で持つ)
- 数値条件付きルールの判定(jev-mario で誤適用が敗因。コード側に置く)
- 確率の較正(素の softmax。閾値運用の前にラベル付きセットで測る。ログがその元)
- ollama backend(logprobs が先頭 1 トークンのみ)
- 複数質問を 1 回の生成で読む(同上)

## 検証 OK 条件(実出力で示す)

1. `uv run pytest tests/ -q` が単独実行で全 PASS(fake backend で GPU 不要)。トーナメント(例: 120 候補)も fake backend で通す
2. `systemctl --user status chakuho` が active、`curl -s localhost:9750/health` が `ok: true`
3. 別マシンから `curl http://<host>:9750/v1/systemone` で choice/noul/score の 3 種が返る(LAN 越しの実出力)
4. jev-mario を `JEV_URL=http://127.0.0.1:9750/v1/systemone` で 1-1 直接操作 → results.jsonl に行が追加される(到達距離は temperature 0 なので shim と同程度、値は記録する)
5. 異常系: backend を指さない URL で `ask` → 503 と error JSON が 60 秒以内に返る。53 個以上(例 120 個)の選択肢 → 2 段トーナメントが正常に動き `stages: 2`。2705 個 → 400。coverage 0 を fake backend で作る → `degraded: true`
6. `/health` は backend 停止時に 503

## 設計レビュー(3 人格の LLM 合議、CONCERN 2 / FAIL 1)と対応

- 検証条件 5 が設計(53 個以上はトーナメント)と矛盾 → 条件 5 を修正(上記)
- decisions.jsonl の無限成長 → 日付別ファイル + 保持日数で削除
- 並列投入の過負荷 → セマフォ上限
- トーナメントのチャンク境界バイアス → 各チャンク上位 k 個を決勝へ。ランダム化は再現性を壊すので採らない

## 反証してほしい点

- state を最後に置く配置は mac-do 型では正しいか(jev-mario では壁の誤判定が悪化した)
- トーナメントの決勝分布は候補間の比較として一貫しているか(チャンク境界で強い候補同士が予選で潰し合う偏り)
