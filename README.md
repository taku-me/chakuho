# chakuho(択法)

Jev 互換の「生成しない判定」エンドポイント。state と、答えの形を宣言した質問を受け取り、
ローカル LLM(OpenAI 互換 API)に 1 トークンだけ何度も出させ(既定)、または 1 回の logprobs を読み、
宣言した選択肢上の確率分布を返す。文章もコードも座標も出さない。候補は呼ぶ側が実行時に列挙する(AX 木、DOM、タスク一覧など)。

名前は七覚支の「択法(ちゃくほう)」から。諸法を吟味して正しいものを選び取る働き。
Jev(TypeSafe)の "System One model" を、手元の汎用モデルで代用するための道具。

本体(`chakuho/`)の依存は Python 標準ライブラリのみ。ベンチの道具は optional の `bench` 依存、蒸留は別の環境(`distill/`)。

## どこで動いているか

GX10 では systemd の user unit `chakuho.service` が常駐させる。unit の実体は task-sync `ops/fleet/systemd/chakuho.service`、
起動は task-sync `ops/fleet/ato-services/chakuho-start.sh`(tailscale の IP で待ち、backend・推定方式の env をここで決める)。
艦隊の中では、task_db の票のレーンの推定(task-sync `ops/fleet/lib/lane_estimate.py`)が呼ぶ。

今の設定と状態は次で見る:

```bash
systemctl --user status chakuho.service
grep -v '^#' ~/projects/task-sync/ops/fleet/ato-services/chakuho-start.sh   # 待つ IP・backend・推定方式
ss -tlnp | grep 9750
curl -s http://100.84.195.92:9750/health
```

## 起動

```bash
uv sync
uv run chakuho serve --port 9750          # 127.0.0.1 で待ち受け
curl -s localhost:9750/health
```

サーバに認証は無い。既定は `127.0.0.1` だけで待ち受ける。別の機械から呼ぶ時だけ
`--host 0.0.0.0`(または env `CHAKUHO_HOST`)で明示的に開く。その時は起動時に警告が出る。
開くのは信頼できるネットワーク内だけにする(届く相手は誰でも backend の推論を呼べ、
リクエスト本文は判定ログ `CHAKUHO_LOG_DIR` に書かれる)。

backend は OpenAI 互換 `/chat/completions`(`CHAKUHO_BACKEND_URL`)。
既定の推定方式(`CHAKUHO_ESTIMATOR=sampling`)は `n` サンプリングにさえ対応していれば動く
(vLLM / SGLang+投機的デコード / Ollama 等)。`CHAKUHO_ESTIMATOR=logprobs` にする時は
backend が `logprobs` / `top_logprobs` を返せる必要がある。
どちらの方式でも OpenAI 互換層は thinking を切る指定(`think: false`)を受け付けないことがあるため、
Qwen3 のような thinking モデルでは最初の 1 トークンが思考の書き出しになり判定にならない場合がある。
instruct モデルを指すか、`chat_template_kwargs.enable_thinking=false` が効く backend を使う。

## API

`POST /v1/systemone`

```json
{"state": <任意の JSON か文字列>,
 "questions": {
   "<name>": {"type": "choice|noul|score", "instructions": <文字列か dict>, "criteria": <dict か list>}
 }}
```

同じリクエスト内の質問は並列に評価される(質問どうしは互いの答えを知らない)。
`type` は Jev 本家の表記(`noul`)と Vercel AI Gateway の表記(`boolean`)の両方を受け付ける。
state が dict で `images` キー(画像 URL(`data:` / http)のリスト)を持てば、VLM の backend へ画像のパート(content parts)として渡す。

### choice(選択)

```bash
curl -s localhost:9750/v1/systemone -d '{
  "state": {"window": "Calendar", "focused": "[AXTextField] New Event"},
  "questions": {"target": {"type": "choice",
    "instructions": "目標: イベントを保存する。操作する要素を 1 つ選ぶ",
    "criteria": {"[AXButton] Add (toolbar)": "新規作成を開始", "[AXButton] Add (New Event sheet)": "シートを確定", "[AXButton] Cancel": "破棄", "__none__": "どの要素も合わない"}}}}'
```

返り値: `{"choice": "...", "probabilities": {option: p}, "coverage": p, "stages": 1|2}`

- 52 個までは 1 ラウンド。53〜2704 個は 52 個ずつのチャンクで並列に予選し、各チャンク上位 k 個(k = 52 // チャンク数)で決勝(`stages: 2`)。予選落ちは確率 0
- `__none__` という選択肢があれば、全チャンクと決勝に必ず含まれる(「該当なし」を明示的に答えさせる)
- 2705 個以上は HTTP 400

52 はラベル(A〜Z, a〜z)の数で決まる定数。

### noul / boolean(yes/no)

```bash
curl -s localhost:9750/v1/systemone -d '{"state": "画面にエラーダイアログが出ている",
  "questions": {"done": {"type": "noul", "instructions": "目標は達成済みか",
    "criteria": {"true": "達成済み", "false": "未達成"}}}}'
```

返り値: `{"noul": p_yes, "probability": p_yes, "coverage": p}`

### score(段階)

```bash
curl -s localhost:9750/v1/systemone -d '{"state": "これから Delete Account を押す",
  "questions": {"risk": {"type": "score", "instructions": "この操作の危険度",
    "criteria": ["安全", "要注意", "破壊的"]}}}'
```

返り値: `{"score": 0..1 に正規化した期待値, "probabilities": {level: p}, "coverage": p}`

### 共通

- レスポンス: `{"answers": {...}, "usage": {"input_tokens": n}, "latency_ms": n, "model": "...", "backend": "primary|fallback"}`
- `coverage` は宣言ラベルに載った確率質量。0 なら一様分布を返し `"degraded": true` が付く。低い時は判定として信用しない。ただし coverage は「答えの形を守ったか」の指標であり、「中身が正しいか」は示さない(`docs/benchmarks.md`)
- 形の壊れたリクエスト(`state` か `questions` が無い・question が object でない・質問の数が `CHAKUHO_MAX_QUESTIONS` を超える): HTTP 400。400 の時は判定ログに本文を残さない(質問の名前だけ)
- 本体が `CHAKUHO_MAX_BODY_BYTES` を超える: HTTP 413
- backend に届かない / 応答が壊れている: HTTP 503 と `{"error": ...}`。`CHAKUHO_FALLBACK_BACKEND_URL` があれば、主が 503 相当の時だけ予備で同じ判定をやり直し、`backend: "fallback"` と `primary_error` を付けて返す。backend への 1 回の呼び出しは `BACKEND_TIMEOUT_SEC` で打ち切る(ハングしない)
- `GET /health`: backend の `/models` を取り、さらに実際に 1 トークン生成させる(`/models` だけではエンジンが固着していても 200 になるため)。通れば `{"ok": true, "backend", "model"}`、通らなければ 503。主が落ちて予備が通る時は `{"ok": true, "degraded": true, ...}` の 200

上限の既定の値はコードで見る:

```bash
grep -n "^DEFAULT_\|^BACKEND_TIMEOUT_SEC\|^MAX_POOL_WORKERS\|^SOCKET_TIMEOUT_S" chakuho/core.py chakuho/server.py chakuho/netsafe.py
```

## Python クライアント / CLI

```python
from chakuho.client import decide
out = decide(state, questions, url="http://<host>:9750/v1/systemone")
```

```bash
uv run chakuho ask request.json        # ファイルか - で stdin。env CHAKUHO_URL で宛先を変える
```

## 環境変数

既定の値は上の `grep` で見る(表には書かない)。

| 変数 | 意味 |
|---|---|
| `CHAKUHO_HOST` | 待ち受けの host(`--host` が優先)。既定は `127.0.0.1` |
| `CHAKUHO_BACKEND_URL` | OpenAI 互換 backend |
| `CHAKUHO_FALLBACK_BACKEND_URL` | 主 backend が不通の時だけ使う予備 backend(既定なし) |
| `CHAKUHO_ESTIMATOR` | `sampling`(n 回サンプリングして頻度を数える)か `logprobs`(1 回の logprobs を読む。backend が対応している時だけ) |
| `CHAKUHO_SAMPLES` | `sampling` の n。不正値は既定値に倒す |
| `CHAKUHO_TOP_LOGPROBS` | `logprobs` 方式で backend に要求する top_logprobs 数。`mlx_lm.server` は上限が小さいので下げる。不正値は既定値に倒す |
| `CHAKUHO_PREFILL` | `1` で答えの合図 `Label:` を assistant 側の書き出し(prefill)として送る。`continue_final_message` 非対応の backend では `0`(user 側に書く) |
| `CHAKUHO_MODEL` | モデル ID。通常は指定しない(backend の `/models` の先頭を使う) |
| `CHAKUHO_MAX_INFLIGHT` | backend への同時リクエストの上限。backend が同時に処理できる本数に合わせる(SGLang の実際の上限は `journalctl --user -u resident-llm -b \| grep 'capped to'` で見る) |
| `CHAKUHO_MAX_QUESTIONS` | 1 リクエストの質問の数の上限 |
| `CHAKUHO_MAX_BODY_BYTES` | リクエスト本体の上限 |
| `CHAKUHO_LOG_DIR` | 判定ログ `decisions-YYYYMMDD.jsonl` の置き場(state 全文を含む。較正データの元) |
| `CHAKUHO_LOG_KEEP_DAYS` | これより古い日付のログを、起動時と日付が変わった時に消す |
| `CHAKUHO_URL` | クライアント側の宛先 |

## 仕組み

1. 選択肢にラベル A〜Z, a〜z を振り、`instructions → options → state → question → "Label:"` の順で prompt を組む(state は末尾)
2. `sampling`: backend へ `max_tokens=1, temperature=1.0, n=CHAKUHO_SAMPLES`(thinking 無効)で投げ、n 個の生成テキストを宣言ラベルへ数え上げて分布にする(logprobs は要求しない)。`logprobs`: `max_tokens=1, temperature=0, logprobs=true, top_logprobs=CHAKUHO_TOP_LOGPROBS` で 1 回だけ投げ、top logprobs をラベルへ集約する
3. どちらの方式もラベル内で正規化して分布にする。ラベル外に落ちた分は coverage の欠けとして報告する
4. backend への同時の呼び出しは `CHAKUHO_MAX_INFLIGHT` のセマフォで絞る。1 リクエストが作るスレッドは `MAX_POOL_WORKERS` まで。送り切らない接続は `SOCKET_TIMEOUT_S` で切る

Jev や NanoJev のような判断専用モデル(判断ヘッドを学習させたもの)ではなく、汎用 instruct モデルの次トークン分布を読む代用品。
利点は「既に常駐している汎用モデルをそのまま使える」こと、欠点は「候補数の上限(1 ラウンド 52)、較正されていない確率、算術を含む判断の弱さ」。

## ベンチマーク

測った日・条件つきの記録は `docs/benchmarks.md`(レイテンシ・Jev との一致率・実画面 336 ケースの GUI 要素選択・jev-mario)。
道具は `bench/`(データは含まない。マスク済み AX スナップショットのパスを `AX_SNAPSHOT` で渡す)。記録から読める設計上の結論:

- 律速は prefill(入力トークン数)で、decode ではない。同一リクエスト内の質問は backend がバッチ化するので、質問を増やしても時間はあまり増えない
- Jev は候補数にほとんど依存しない。chakuho 方式は入力トークン数に比例するので、候補が少なければ速く、多いと逆転する
- coverage は「答えの形を守ったか」で、正しさの指標ではない(小型モデルは coverage が高いまま中身だけ違う)
- 小型モデルを素のまま使うと「該当なし」を選べず、候補の順序や混入で崩れる。小型モデルを使うなら chakuho の判定ログからの蒸留が要る(`distill/`)

## 蒸留

27B の判定分布を小型の Qwen(LoRA)へ蒸留する道具が `distill/` にある。手順は `distill/README.md`、設計と検証の条件は `docs/distill-design.md`。

## 限界と設計上の判断

- 自由文と座標は出せない。テキストは生成モデル、座標は AX API や DOM に持たせる
- 数値条件の判定(「8 は 6 以上か」)は苦手。ルールはコード側に置き、モデルには意味の照合と順位付けだけを渡す
- 確率は較正されていない(ラベル内 softmax)。閾値で自動実行する前に、判定ログからラベル付きセットを作って測る
- 候補が多い時のトーナメントは、強い候補が同じチャンクで潰し合う偏りを持つ。各チャンク上位 k 個を決勝へ通して緩和しているが、消えてはいない
- ハイブリッド(mamba 系)のモデルは prefix cache のブロックが大きく、短い prompt では cache が効かないことがある。state を大きくしても速くはならない
- Ollama の OpenAI 互換層では `think: false` が効かないことがあるので、thinking モデルは 1 トークン判定に使えない(instruct モデルなら chakuho をそのまま向けられる)
- 落ちた時に呼ぶ側が黙って別の答えへ降格しないこと。503 は「判定できなかった」であって「いいえ」ではない

## 確かめ方

```bash
uv run pytest tests -q      # fake backend で完結、GPU 不要
```

設計と検証条件は `docs/design.md`。

## 改版履歴

- 2026-10-20 既定の待ち受けを 127.0.0.1 にし、本体の上限(413)・質問の数の上限・形の壊れた質問の 400・1 リクエストのスレッドの上限・ソケットのタイムアウトを入れる (#1)
- 2026-09-29 `/health` が backend に実際に 1 トークン生成させてから ok を返す (b51bc98)
- 2026-09-24 推定方式 `sampling`(n 回サンプリングして数える)を足し、既定にする。`CHAKUHO_ESTIMATOR` で `logprobs` に戻せる (b4d3058)
- 2026-09-20 `/health` が毎回 backend を叩いて生死を返す (17001d8)
- 2026-09-20 答えの合図 `Label:` を assistant の prefill で送る。`CHAKUHO_PREFILL=0` で user 側に書く (336f350)
- 2026-09-19 state の `images` を VLM の backend へ画像のパートとして渡す (bc6a6d8)
- 2026-09-19 蒸留の道具 `distill/`(逆生成・教師データ・LoRA 蒸留・評価・MLX の配信 backend)を足す (057c22b)
- 2026-09-19 chakuho を置く: Jev 互換の `POST /v1/systemone`(choice・noul・score)と `GET /health` (33d0e98)
