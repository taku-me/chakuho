# chakuho(択法)

Jev 互換の「生成しない判定」エンドポイント。state と、答えの形を宣言した質問を受け取り、
ローカル LLM(OpenAI 互換 API)に 1 トークンだけ何度も出させ(既定)、または 1 回の logprobs を読み、
宣言した選択肢上の確率分布を返す。文章もコードも座標も出さない。候補は呼ぶ側が実行時に列挙する(AX 木、DOM、タスク一覧など)。

名前は七覚支の「択法(ちゃくほう)」から。諸法を吟味して正しいものを選び取る働き。
Jev(TypeSafe)の "System One model" を、手元の汎用モデルで代用するための道具。

依存は Python 標準ライブラリのみ。

## 起動

```bash
uv sync
uv run chakuho serve --host 0.0.0.0 --port 9750
curl -s localhost:9750/health
```

backend は OpenAI 互換 `/chat/completions`(`CHAKUHO_BACKEND_URL`、既定 `http://localhost:8006/v1`)。
既定の推定方式(`CHAKUHO_ESTIMATOR=sampling`)は `n` サンプリングにさえ対応していれば動く
(vLLM / SGLang+投機的デコード / Ollama 等)。`CHAKUHO_ESTIMATOR=logprobs` にする時は
backend が `logprobs` / `top_logprobs` を返せる必要がある(vLLM、Ollama 0.33 で実測)。
どちらの方式でも OpenAI 互換層は thinking を切る指定(`think: false`)を受け付けないことがあるため、
Qwen3 のような thinking モデルでは最初の 1 トークンが思考の書き出しになり判定にならない場合がある。
instruct モデル(例: `qwen3:4b-instruct-2507`)を指すか `chat_template_kwargs.enable_thinking=false` が効く backend を使う。

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

- レスポンス: `{"answers": {...}, "usage": {"input_tokens": n}, "latency_ms": n, "model": "..."}`
- `coverage` は top logprobs のうち宣言ラベルに載った確率質量。0 なら一様分布を返し `"degraded": true` が付く。低い時は判定として信用しない。ただし coverage は「答えの形を守ったか」の指標であり、「中身が正しいか」は示さない(下記ベンチマーク参照)
- backend に届かない / 応答が壊れている: HTTP 503 と `{"error": ...}`。ハングしない(timeout 60 秒)
- `GET /health`: backend の `/models` が取れれば `{"ok": true, "backend", "model"}`、取れなければ 503

## Python クライアント / CLI

```python
from chakuho.client import decide
out = decide(state, questions, url="http://<host>:9750/v1/systemone")
```

```bash
uv run chakuho ask request.json        # ファイルか - で stdin。env CHAKUHO_URL で宛先を変える
```

## 環境変数

| 変数 | 既定 | 意味 |
|---|---|---|
| `CHAKUHO_BACKEND_URL` | `http://localhost:8006/v1` | OpenAI 互換 backend |
| `CHAKUHO_ESTIMATOR` | `sampling` | `sampling`(n 回サンプリングして頻度を数える) か `logprobs`(1 回の logprobs を読む。backend が対応している時だけ) |
| `CHAKUHO_SAMPLES` | `16` | `sampling` の n。不正値は既定値に倒す |
| `CHAKUHO_MODEL` | backend の `/models` 先頭 | モデル ID。通常は指定しない |
| `CHAKUHO_LOG_DIR` | `~/.ato/chakuho` | 判定ログ `decisions-YYYYMMDD.jsonl` の置き場(state 全文を含む。較正データの元) |
| `CHAKUHO_LOG_KEEP_DAYS` | `90` | これより古い日付のログを削除 |
| `CHAKUHO_MAX_INFLIGHT` | `4` | backend への同時リクエスト上限。backend が同時に処理できる本数に合わせる |
| `CHAKUHO_TOP_LOGPROBS` | `20` | `logprobs` 方式で backend に要求する top_logprobs 数。`mlx_lm.server` は上限 11。不正値は既定値に倒す |
| `CHAKUHO_PREFILL` | `1` | 答えの合図 `Label:` を assistant 側の書き出し(prefill)として送る。`continue_final_message` 非対応の backend では `0`(user 側に書く旧方式) |
| `CHAKUHO_FALLBACK_BACKEND_URL` | なし | 主 backend が不通の時だけ使う予備 backend。応答の `backend` が `primary` / `fallback` のどちらで答えたかを示す |
| `CHAKUHO_URL` | `http://localhost:9750/v1/systemone` | クライアント側の宛先 |

## 仕組み

1. 選択肢にラベル A〜Z, a〜z を振り、`instructions → options → state → question → "Label:"` の順で prompt を組む(state は末尾)
2. 既定(`sampling`): backend へ `max_tokens=1, temperature=1.0, n=16`(thinking 無効)で投げ、n 個の生成テキストを宣言ラベルへ数え上げて分布にする(logprobs は要求しない)。`logprobs` 方式: `max_tokens=1, temperature=0, logprobs=true, top_logprobs=20` で 1 回だけ投げ、top logprobs をラベルへ集約する
3. どちらの方式もラベル内で正規化して分布にする。ラベル外に落ちた分は coverage の欠けとして報告する

Jev や NanoJev のような判断専用モデル(判断ヘッドを学習させたもの)ではなく、汎用 instruct モデルの次トークン分布を読む代用品。
利点は「既に常駐している汎用モデルをそのまま使える」こと、欠点は「候補数の上限(1 ラウンド 52)、較正されていない確率、算術を含む判断の弱さ」。

## ベンチマーク

すべて 2026-09-19 の実測。環境: NVIDIA DGX Spark(GB10、128GB ユニファイドメモリ)、
backend は vLLM の Qwen3.8-27B NVFP4(prefix caching 有効、MTP 投機デコード有効)。
測定中は別の生成リクエストが 1 本同居していた(GPU 使用率 95%)。数字は同じ条件でしか再現しない。

### レイテンシ(候補数別)

合成した AX 要素(`[role] label i (pane k)` 形式、毎回 1 要素を変えて cache を避ける)、各 8 回。

| 候補数 | 質問 1(choice) p50 / p95 | 質問 3(choice + score + noul) p50 / p95 | stages | 入力トークン |
|---|---|---|---|---|
| 8 | 0.61 / 0.62 s | 0.69 / 0.69 s | 1 | 356 / 556 |
| 32 | 0.88 / 0.88 s | 1.01 / 1.01 s | 1 | 1138 / 1338 |
| 100 | 3.18 / 3.18 s | 3.27 / 3.28 s | 2 | 5212 / 5412 |
| 233 | 6.35 / 6.36 s | 5.26 / 5.26 s | 2 | 10258 / 10458 |

律速は prefill(入力トークン数)で、decode ではない。同一リクエスト内の質問は vLLM がバッチ化するので、質問を 3 つに増やしても 1 割程度しか増えない。
参考: 同じ画面に 6 問聞く比較で、chakuho 型(6 問並列)1.13 s、thinking off の JSON 生成 2.3〜3.5 s、thinking on 80 s。

#### 蒸留した学生・Jev と並べた場合(2026-09-21、質問 1 問(choice)の p50)

同じ合成ケース。学生は in-process(素の PyTorch、バッチ無し)、教師は同じ機械の vLLM、Jev は Gateway 経由。

| 候補数 | 学生 Qwen3-1.7B | 教師 Qwen3.8-27B | Jev |
|---|---|---|---|
| 8 | **0.08 s** | 0.44 s | 0.48 s |
| 32 | **0.19 s** | 0.69 s | 0.45 s |
| 100 | 0.90 s (2 段) | 2.18 s | **0.48 s** |
| 233 | 1.83 s (2 段) | 2.28 s | 0.73 s (p95 3.23 s) |

- **Jev は候補数にほとんど依存しない**(100 候補まで 0.48 s 前後)。chakuho 方式は入力トークン数に比例するので、
  **候補が少なければ速く、多いと逆転する。分岐点は 50 候補前後**
- **学生の 233 候補 1.83 s は教師 27B の 2.28 s とほぼ並ぶ。** 16 倍小さいのにこうなるのは
  **モデルではなく実行経路が遅い**ため(素の PyTorch、バッチ無し、トーナメントの各ラウンドが直列)。
  推論サーバに載せれば余地がある。**この経路の数字を「1.7B の速度」として読まないこと**
- 1 トークンしか生成しないので、**重みの読み出し(メモリ帯域)は律速ではない**。
  8bit で約 1.8 GB なので、360 GB/s 級の GPU なら 1 回読みは 5 ms 程度

### Jev 本家との一致率

基準 = Jev(`typesafe-ai/jev`、Vercel AI Gateway 経由、応答中央値 578 ms)。
同じ 35 件(GUI 要素選択 12 件: カレンダー画面を模した 28 要素 + `__none__` / ゲーム判断 23 件: jev-mario の 9 択)を各モデルに流し、
choice の一致率と、noul / score の平均絶対差を出した。小型モデルは Ollama のネイティブ API 経由で chakuho と同一プロンプト・同一集約。

| モデル | GUI 12 件 | ゲーム 23 件 | noul 平均差 | score 平均差 |
|---|---|---|---|---|
| chakuho: Qwen3.8-27B NVFP4(vLLM) | 10/12 (83%) | 9/23 (39%) | 0.09 | 0.05 |
| Qwen3.8-27B Q4_K_M(Ollama) | 11/12 (92%) | 7/23 (30%) | 0.10 | 0.09 |
| Qwen3-8B(Ollama、thinking off) | 10/12 (83%) | 15/23 (65%) | 0.15 | 0.02 |
| Qwen3-4B-Instruct-2507 Q4(Ollama) | 7/12 (58%) | 11/23 (48%) | 0.21 | 0.05 |

読み方:

- GUI の要素選択では 27B も 8B も Jev と 8 割強で一致する。同じ 27B でも backend/量子化が違うと 1 割ずれる(27B 同士の一致率 92%)ので、この差は測定のノイズ床と同程度。**この範囲では chakuho は Jev と同等**
- ゲーム判断は Jev 自身も迷っており(Jev の最尤確率の中央値 0.76、35 件中 14 件が 0.5 未満)、どのモデルも一致率が低い。問題の難しさであって、モデル差は読めない
- 小型モデルは coverage が 1.00 のまま中身だけ違う。**coverage は「答えの形を守ったか」で、正しさの指標ではない**
- 12 件は少ない。336 件での再測定は次節

### GUI 要素選択ベンチマーク(実画面 336 ケース)

macOS の実アプリ 11 画面(通常ウィンドウ 8 + シート/ダイアログ 3。アクセシビリティ木を取得し個人情報をマスク済み)から
112 タスクを作り、各タスクを 3 変種(そのまま / 要素順シャッフル / 別アプリの要素 15 個を混入)で流した。
候補は 1 画面 26〜233 要素 + `__none__`。期待答えは作題者が決め、`__none__` が正解のタスクを 39 件含む。
Jev は Gateway 経由、chakuho は 27B NVFP4(vLLM)、小型モデルは Ollama で同一プロンプト・同一集約。

| モデル | 通常画面 258 件 | シート 78 件 | 通常: `__none__` 正解 30 件 | 順序シャッフル / 混入で落ちるか |
|---|---|---|---|---|
| chakuho: Qwen3.8-27B NVFP4 | **245 (95%)** | **72 (92%)** | 29 (97%) | 落ちない(93→97→95%) |
| Jev(`typesafe-ai/jev`) | 230 (89%) | 64 (82%) | 27 (90%) | 落ちない(85→91→92%) |
| Qwen3-8B(thinking off) | 162 (63%) | 34 (44%) | 3 (10%) | 落ちる(67→60→60%) |
| Qwen3-4B-Instruct-2507 Q4 | 157 (61%) | 33 (42%) | 3 (10%) | 落ちる(66→58→58%) |
| **chakuho 蒸留: Qwen3-1.7B(LoRA 融合済み、bf16)** | **231 (90%)** | **65 (83%)** | 17 (57%) | 落ちない(89→89→86%) |
| Kev 4B(`jaredpalmer/kev`、零ショット、MPS) | 163 (63%) | 23 (29%) | 16 (53%) | 落ちない(候補数の帯別 62 / 50 / 60%) |
| Kev 0.6B(同) | 5 (2%) | 0 (0%) | 0 | 崩壊(先頭候補を選び続ける) |

一致率(同じ要素を選んだ割合): Jev × chakuho 86%(通常)/ 90%(シート)、Jev × 8B 58%、8B × 4B 93%。

読み方:

- **27B の chakuho は Jev と同等以上。** 差の大半はシステム設定(Jev 78% / chakuho 88%)と Docker Desktop(87% / 98%)で、要素名が抽象的な画面で Jev が迷う
- **蒸留した 1.7B は Jev に並ぶ。** 通常 231 対 230、シート 65 対 64。ただし**中身は正反対**で、
  要素を当てる方は学生が強く(通常の正解あり 214 対 203)、**「該当なし」は弱い(17/30 対 27/30)**。
  合計が並ぶのはこの二つが相殺した結果。外す場所も違い(両方正解 263 / Jev のみ 31 / 学生のみ 33 /
  両方不正解 9)、**どちらかが当てたケースは 327/336 = 97%**
- **訓練に無いアプリでは学生が Jev を上回る(200/228 対 194/228)。** 336 件のうち 228 件は
  学習に使っていないアプリ(Docker Desktop / eqMac / システム設定 / テキストエディット等)で、
  学習済みアプリ 108 件での 89% と 3 ポイントしか違わない。画面を覚えて点を取っているのではない
- **8B / 4B は足りない。** 特に「該当なし」をほぼ選べず(30 件中 3 件)、候補の順序や混入で崩れる。8B と 4B の答えは 93% 同じで、間違え方も同じ
- 破棄確認シート(削除 / キャンセル / 保存)の危険な取り違えは Jev・chakuho ともゼロ。両者が落とした残りは「書類を印刷する → ファイルメニュー」のように作題者の期待(`__none__`)が厳しすぎるものと、「削除せず何もしない → `__none__`」のように解釈が割れるもの
- 上の 35 件比較で「8B も 8 割」と出ていたのは母数の小ささによる。**小型モデルを使うなら chakuho の判定ログからの蒸留が要る**
- Kev(Qwen + LoRA + 読み出しヘッド、System One 互換)の公開重みを零ショットで当てると 4B は素の Qwen3-8B と同じ帯(63%)で、`__none__` の扱いは素のモデルより良い(16/30)が、シート画面では崩れる。GUI 用の学習をしていない重みなので、Kev の構造そのものの評価ではない。1 判定は Apple Silicon(MPS、他モデルと GPU 共有)で 2.5〜10 秒

### jev-mario での比較

[4esv/jev-mario](https://github.com/4esv/jev-mario) を無改変(API の URL だけ環境変数化)で実行し、Jev の呼び先を chakuho に差し替えた。
数値は到達距離 x(旗は約 3160)。Jev の列は本物の Jev(Gateway 経由)で同日に再現した値。

| モード | chakuho(27B) | Jev(再現) | Jev(README 記載) | rules / search |
|---|---|---|---|---|
| 直接操作 1-1 | 401 | 686 | 686 | 1129 |
| 直接操作 2-1 | 308 | 476 | 473 | 741 |
| 直接操作 3-1 | 445 | 607 | 607〜841 | 608 |
| live 1-1 | 315 | 312 | 315 | 315 |
| branch 1-1 | 2370 | (未実施) | 旗 | 2370 |

chakuho の負け方は一貫していて、数値条件付きルールの誤適用(「届かない、かつ 6 タイル未満なら後退」を、届く上に 8 タイル離れた壁で発動させて後退ループ)。
候補を測った結果から選ぶ branch モードでは 4 分の 3 まで進み、決定論の search と同じ地点で止まった。
1 判定のレイテンシは chakuho 0.86 s(入力約 1100 トークン)、Jev 0.61 s(日本から Gateway 経由。公称は 0.1〜0.25 s)。

### Vercel AI Gateway で Jev を呼ぶ時の注意

Jev は 2026-09-25 まで Gateway 上で無料だが、AI Credit を購入していないチームは「free tier」として約 10 分に数リクエストへ絞られる(429、Retry-After 無し)。
最小額の AI Credit を購入すると解除される(Jev の利用額自体は 0 のまま)。エンドポイントは `POST https://ai-gateway.vercel.sh/v4/ai/evaluation-model`、
ヘッダーは `ai-model-id: typesafe-ai/jev` / `ai-evaluation-model-specification-version: 4` / `ai-gateway-protocol-version: 0.0.1`。
レスポンスは `boolean` が `probability`、`score` が 0..n-1 の補間値なので、chakuho 形式へ揃えるなら noul = probability、score = score / (n-1)。

## 限界と設計上の判断

- 自由文と座標は出せない。テキストは生成モデル、座標は AX API や DOM に持たせる
- 数値条件の判定(「8 は 6 以上か」)は苦手。ルールはコード側に置き、モデルには意味の照合と順位付けだけを渡す
- 確率は較正されていない(ラベル内 softmax)。閾値で自動実行する前に、判定ログからラベル付きセットを作って測る
- 候補が多い時のトーナメントは、強い候補が同じチャンクで潰し合う偏りを持つ。各チャンク上位 k 個を決勝へ通して緩和しているが、消えてはいない
- Qwen3.5 系のハイブリッドモデルは vLLM の prefix cache ブロックが 1600 トークンで、短い prompt では cache が効かない(hit 0%)。state を大きくしても速くはならない
- Ollama(0.33)は OpenAI 互換層でも logprobs を返す。ただし `think: false` が効かないので thinking モデルは 1 トークン判定に使えない(instruct モデルなら chakuho をそのまま向けられる)。ネイティブ API の logprobs は MTP 付きモデル(qwen3.8:27b)では先頭 1 トークンだけ、通常モデル(qwen3:8b)では全トークン返る。chakuho は先頭 1 トークンしか要らないので、どちらでも判定自体は可能
- 落ちた時に呼ぶ側が黙って別の答えへ降格しないこと。503 は「判定できなかった」であって「いいえ」ではない

## 開発

```bash
uv run pytest tests -q      # fake backend で完結、GPU 不要
```

設計と検証条件は `docs/design.md`、蒸留(MacBook 常駐の小型 backend)は `docs/distill-design.md`。
ベンチマークのスクリプトは `bench/`(データは含まない。マスク済み AX スナップショットのパスを `AX_SNAPSHOT` で渡す)。
