# chakuho(択法)— System One 型の判定エンジン

## 目的

Jev(TypeSafe)と同じ形の「生成しない判定」をローカル LLM で提供する。
入力は state(任意 JSON か文字列)と、答えの形を宣言した質問。出力は選択肢上の確率分布。
文章もコードも書かない。座標も出さない。候補は呼ぶ側が実行時に列挙する(AX 木・DOM・タスク一覧など)。

用途: mac-do の要素選択、監視ログの異常トリアージ、端末画面の分類、ルーティング、compaction のスコアリング。
Jev 互換 API なので呼び出し側は将来 Jev 本体や別モデルに差し替えられる。

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
- `coverage` = 宣言ラベルに乗った質量の割合(推定方式による、下記)。0 なら一様分布を返し `"degraded": true` を付ける(「判定できなかった」を「判定した」と同じ語彙で返さない)

`GET /health` → `{"ok": true, "backend": url, "model": str}`(backend の `/models` が取れなければ `ok: false`, HTTP 503)

### 判定の仕組み

- 選択肢にラベル A-Z, a-z(最大 52、1 トークン)を振り、prompt は「instructions → options(ラベル: option — description) → state → question」の順。state を最後に置く(recency)。答えの合図 `Label:` は assistant 側の書き出し(prefill、`continue_final_message`)として渡す(env `CHAKUHO_PREFILL=0` で user 側に書く旧方式に戻せる。backend が prefill 非対応の時用)
- 同一リクエスト内の複数質問はスレッドで並列投入
- 53 個以上の選択肢は chakuho 内で 2 段トーナメント: 52 個ずつのチャンクへ分け(並列)、各チャンクの上位 k 個(k = max(1, 52 // チャンク数))を集めて決勝を 1 回。上位 1 個だけでなく上位 k 個を通すのは、強い候補が同じチャンクで潰し合う偏りを減らすため。`__none__` があれば全チャンクと決勝に含める。返り値の `probabilities` は決勝の分布(落選したものは 0)、`stages: 2` を付ける。上限は 52×52=2704 で、超えたら HTTP 400
- backend 呼び出しは timeout 60 秒。失敗は HTTP 503 と `{"error": ...}`。ハングさせない
- 全リクエストを `CHAKUHO_LOG_DIR/decisions-YYYYMMDD.jsonl` に日付別で追記(state 全文、質問、答え、latency)。較正データ・回帰比較(下記)の元になる。起動時と日付切替時に `CHAKUHO_LOG_KEEP_DAYS`(既定 90)より古いファイルを削除する
- backend への同時リクエスト数はセマフォで上限 `CHAKUHO_MAX_INFLIGHT`(既定 16)。超えた分は待つ

### 推定方式(`CHAKUHO_ESTIMATOR`、既定 `sampling`)

backend から確率分布を取り出す方式は 2 つあり、env `CHAKUHO_ESTIMATOR` で切り替える。

- **`sampling`(既定)**: `n`(env `CHAKUHO_SAMPLES`、既定 16)個の 1 トークンサンプルを
  `temperature=1.0` で要求し、宣言ラベルに一致したサンプルの出現頻度を数えて分布にする
  (`aggregate_counts()`)。`logprobs` は一切要求しない。coverage = 宣言ラベルに一致した
  サンプル数 / n。投機的デコード(DSpark 等)を含め、`n` サンプリングにさえ対応していれば
  backend を選ばない
- **`logprobs`**: 1 回の生成(`temperature=0`)の `top_logprobs` をラベルへ集約する
  (`aggregate()`)。backend が `logprobs`/`top_logprobs` を返せる時だけ使える。
  艦隊の temperature=0 禁止方針の例外はこの経路のみ

sampling が既定なのは、GX10 の常駐 backend(SGLang + 投機的デコード)が
`return_logprob` を HTTP 400 で拒否するため。`logprobs` に対応した backend
(vLLM 等)へ戻す時は `CHAKUHO_ESTIMATOR=logprobs` を明示する。

#### n(サンプル数)の測り方

固定値をここに書かない。`bench/sampling_n_sweep.py` が、実運用の判定ログ
(`CHAKUHO_LOG_DIR/decisions-*.jsonl`)の choice 型 1 質問レコードを母集団にして、
指定した `n` それぞれで今の backend に流し直し、ログに残っている(過去の推定方式による)
答えとの一致率と latency(median/p90)を出す。選択肢数の小・中・トーナメント級の 3 バケットに
均等割りするので、候補が少ない判定と多い判定の両方を見られる。

```bash
python3 bench/sampling_n_sweep.py \
  --logs ~/.ato/chakuho/decisions-*.jsonl \
  --backend http://localhost:8006/v1 \
  --n 16 32 64 --per-bucket 12 --out /tmp/sweep.json
```

`n` を選ぶ基準: 一致率の差がその回のサンプルサイズの統計誤差(二項分布の標準偏差)に
収まる `n` の中から、latency(特にトーナメント級での p90)が最も小さいものを採る。
一致率は `n` を増やしても単調には上がらない(小さな母集団では誤差の方が支配的になりうる)ので、
「一番大きい `n` が一番良い」を前提にしない。

同じログを使う近縁の道具として `bench/replay_against_teacher.py`
(推定方式 1 つを今の backend に流し直して食い違いを見る、常駐後の定期チェック用)がある。

### やらないこと

- 自由文・座標の生成(呼ぶ側が生成モデルか決定論 API で持つ)
- 数値条件付きルールの判定(意味の照合と順位付けだけを渡し、ルールはコード側に置く)
- 確率の較正(素の softmax・素の頻度。閾値運用の前にラベル付きセットで測る。ログがその元)
- `logprobs` の先頭 1 トークンしか返さない backend(投機的デコードの一部実装)を `logprobs` 推定方式で使うこと(sampling 推定方式を使う)
- 複数質問を 1 回の生成で読む
- 落ちた時に呼ぶ側が黙って別の答えへ降格すること。503 は「判定できなかった」であって「いいえ」ではない

## 検証 OK 条件(実出力で示す)

1. `uv run pytest tests/ -q` が単独実行で全 PASS(fake backend で GPU 不要)。sampling/logprobs 両推定方式、トーナメント(例: 120 候補)、coverage=0 の degraded もすべて fake backend で通す
2. `systemctl --user status chakuho` が active、`curl -s localhost:9750/health` が `ok: true`
3. 別マシンから `curl http://<host>:9750/v1/systemone` で choice/noul/score の 3 種が返る(LAN 越しの実出力)
4. 実運用の判定ログ(`~/.ato/chakuho/decisions-*.jsonl`)の直近レコードを `chakuho.client.decide()` で :9750 へ再送し、HTTP 200 と確率分布が返る(`bench/replay_against_teacher.py` と同じ母集団)
5. 異常系: backend を指さない URL で `ask` → 503 と error JSON が 60 秒以内に返る。53 個以上(例 120 個)の選択肢 → 2 段トーナメントが正常に動き `stages: 2`。2705 個 → 400。coverage 0 を fake backend で作る → `degraded: true`
6. `/health` は backend 停止時に 503
