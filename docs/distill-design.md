# chakuho 蒸留 — MacBook 常駐の小型 backend を 27B の判定から作る

## 目的

mac-do(macOS GUI 操作エージェント)が GPU サーバに依存せず要素選択できるようにする。
MacBook 上で常駐する小型モデルを chakuho の backend にし、chakuho の API・core はそのまま使う。
27B(GPU サーバ(DGX Spark)、vLLM)は教師と、常駐が使えない時のフォールバック。

ユーザー決定(2026-09-19、grill の結果):
- 常駐方式: 蒸留した小型 Qwen(LoRA)を `mlx_lm.server` で常駐、chakuho core を載せる。NanoJev 型(決定ヘッド)は採らない
- 教師データ: AX スナップショットから 27B が指示文を逆生成し、chakuho の分布を soft label にする。実運用の判定ログは学習に混ぜず hold-out 検証に使う
- Jev: 9/25 まで第二意見、以後は使わない

## 現状の実測(2026-09-19)

GUI 要素選択 336 ケース(実画面 11 枚、112 タスク × 3 変種、`__none__` 正解 39 件)。README「GUI 要素選択ベンチマーク」節。

| モデル | 通常 258 | シート 78 | `__none__` 30 件 |
|---|---|---|---|
| chakuho 27B NVFP4 | 95% | 92% | 29 |
| Jev | 89% | 82% | 27 |
| Qwen3-8B 素 | 63% | 44% | 3 |
| Qwen3-4B-Instruct 素 | 61% | 42% | 3 |

- 8B/4B は「該当なし」をほぼ選べず、候補の順序シャッフル・混入で崩れる。8B と 4B の答えは 93% 同じ
- 27B の判定ログ: 535 件/日(`CHAKUHO_LOG_DIR/decisions-*.jsonl`、保持 90 日へ変更済み)。うち mac-do 実運用は Calendar/Demo の done,risk,target が 116 件、残りはベンチ由来
- `mlx_lm.server` は `/v1/chat/completions` で `logprobs` / `top_logprobs`(上限 11)を返す(mlx-lm main の server.py を確認)。chakuho は `top_logprobs: 20` を要求している
- GPU サーバに torch は無い。vLLM は `vllm/vllm-openai:v0.20.2rc1-cu130` コンテナで動いており、学習も CUDA 13 コンテナで行う(GB10 は sm_121、pip の汎用 wheel は使えない)
- 画面は 11 枚しか無い。教師データの多様性は画面数で決まる。mac-do に 30 画面以上を依頼済み

## 設計

### 全体

```
AX スナップショット(マスク済み、mac-do が取得)
  → ① 指示文の逆生成(27B、生成モード): 要素ごとに 2〜3 通り + 画面で出来ないこと(`__none__` 用)
  → ② リクエスト合成: chakuho 形式、3 変種(そのまま / シャッフル / 別画面の要素を混入)、noul(done / risk)も同じ画面から作る
  → ③ 教師ラベル: chakuho(27B)へ投げ、ラベル上の確率分布をそのまま保存(soft label)
  → ④ 学習: 学生(Qwen3-1.7B、比較で 4B)に LoRA。損失はラベルトークン位置の KL(教師分布 ‖ 学生分布)
  → ⑤ 評価: 336 ケースのベンチ + hold-out(mac-do 実運用ログ)+ MacBook でのレイテンシ
  → ⑥ 配布: LoRA を融合して MLX へ変換(`mlx_lm.convert`)、mac-do 側で `mlx_lm.server` + chakuho
```

### 置き場

- `chakuho/distill/`(このリポジトリ): `gen_instructions.py`(①) / `build_dataset.py`(②③) / `train.py`(④) / `eval.py`(⑤)。
  学習用依存は `distill/pyproject.toml` で本体と分ける(本体は stdlib のみを保つ)
- データ: `CHAKUHO_LOG_DIR/distill/`(スナップショット・生成データ・教師ラベル・学生の重み)。git に入れない。画面データは個人環境なので公開しない
- 学習の実行: `docker run --gpus all nvcr.io/nvidia/pytorch:<tag>`(aarch64 + CUDA 13)。コンテナ内で `pip install peft`。GB10 のメモリは 128GB 共有で、1.7B の LoRA は bf16 で余裕

### ① 指示文の逆生成

- 画面の要素一覧を 27B(生成モード、vLLM `/v1/chat/completions`)に渡し、要素ごとに「この要素を操作したくなる日本語の指示」を 2〜3 通り出させる。言い回しの幅(丁寧 / ぶっきらぼう / 目的だけ言う)を指定する
- 同じ画面で「この画面では出来ないこと」を 5 件出させ、`__none__` 正解にする
- 要素は AXStaticText 等の操作対象にならないものを除く(操作対象: Button / MenuItem / MenuBarItem / TextField / TextArea / CheckBox / PopUpButton / RadioButton / Tab / Link / DisclosureTriangle / Row / Cell / AXUnknown で label あり)
- 生成は JSON で受け、要素 id と指示文の対応を機械で検証する(存在しない id は捨てる)

### ② リクエスト合成

- ベンチの `build_cases.py` と同じ形(候補 = `id: [role] label`、説明 = window と `in`)。3 変種
- noul: `done`(指示は完了したか。画面の前後 2 枚が要るので、同じ画面で「完了している/いない」を指示文側で作る)と `risk`(この操作は破壊的か)。risk は削除/破棄/送信/上書きの要素に対して yes、それ以外に no が期待される。教師の分布をそのまま使う
- 変種ごとに seed を固定し再現可能にする

### ③ 教師ラベル

- chakuho の `/v1/systemone` に投げる。返る `probabilities` をそのまま保存
- **除外するのは `coverage < 0.5` だけ**。coverage は「上位 logprob の確率質量のうちラベル文字に載った割合」で、教師が答えの形(1 文字のラベル)を守れたかを測る。ラベル間の迷い(最尤確率の低さ)とは別の量。形を守れなかった応答はどのラベルへの分布でもないので教師にならない。迷いは soft label として全部残す(下記)
- 教師の最尤確率が低いケースは外さない(soft label は迷いごと学ぶのが蒸留)。最尤 0.4 未満の件数は「保持した件数」として出力する
- 目標: 30 画面 × 平均 60 要素 × 2.5 指示 × 3 変種 ≈ 13,000 件 + `__none__` 450 件 + noul 数千件
- GPU サーバの vLLM は mac-do の実運用と共用。バッチは同時実行 4 に絞り、`build_dataset.py` 自身がサーキットブレーカーを持つ: 60 秒ごとに chakuho の判定ログから直近の latency 中央値を読み、開始前の 2 倍を超えたら自動で一時停止、1 倍台に戻ったら再開、一時停止が 10 回続いたら中断して件数を出力する(手動監視に頼らない)

### ④ 学習

- 学生: `Qwen/Qwen3-1.7B`(比較で `Qwen3-4B-Instruct-2507`)。chakuho と同じプロンプト・同じラベル(A-Z a-z)。教師と学生でプロンプトを変えない
- LoRA(r=16、全 linear)。損失 = ラベルトークン集合上の KL(teacher ‖ student)。学生の logits はラベルトークンだけを取り出して softmax(chakuho の集約と同じ)
- 毎 epoch 336 ベンチを評価する。2 epoch 終了時に 80% 未満または `__none__` 20/30 未満なら 4B へ切り替える。80〜90% なら 4 epoch まで続け、それでも 90% 未満なら 4B。1 epoch で判断しない(過小学習との区別がつかない)
- 学習は 30 画面分のデータが揃ってから始める。それまでは 11 画面(336 ケース)で配管の確認だけ行い、その正答率は目標に対する判定に使わない(検証条件 10)
- `__none__` の比率を学習データで 5〜10% に保つ(素の 8B が 10% しか選べなかった弱点)

### ⑤ 評価(検証 OK 条件と同じ)

### ⑥ 配布

- `mlx_lm.fuse` で LoRA を融合、`mlx_lm.convert --q-bits 8` で 8bit 化。bf16 と 8bit の差を 336 ベンチで測り 2 ポイント以内を確認する。4bit は 8bit が遅い時だけ
- 学生の配信は `distill/mlx_backend.py`(OpenAI 互換 `/chat/completions`、`max_tokens: 1` 専用)。mlx_lm の Python API で最終位置の logits を取り、**宣言ラベル全部**の logprob を `top_logprobs` として返す。これで coverage は上限 11 に縛られず、27B と同じ定義で測れる。`mlx_lm.server` を使う経路は予備で、その時は `CHAKUHO_TOP_LOGPROBS=11`(実測は下記)
- chakuho 側の変更は済み: `CHAKUHO_TOP_LOGPROBS`(不正値は既定へ)と `CHAKUHO_FALLBACK_BACKEND_URL`(主が不通なら予備で判定、応答に `backend: primary|fallback`)
- mac-do 側: `mlx_backend.py` を launchd 常駐(`KeepAlive` に `ThrottleInterval` 30 秒以上を付け、クラッシュ時の再起動ループで CPU を食わない)、chakuho は主 = localhost、予備 = GPU サーバの vLLM。切り戻しは `CHAKUHO_BACKEND_URL` を GPU サーバへ戻して chakuho を再起動するだけ(学生の重みは消さない)
- `mlx_backend.py` は要求の `top_logprobs` を尊重する(既定は全ラベル)。教師と同じ定義で coverage を比べたい時は 20 を指定する(検証条件 5)

## やらないこと

- NanoJev 型の決定ヘッド学習(ユーザー決定。別系統になる)
- Jev のラベルを教師に混ぜる(27B より弱かった)
- 実運用ログを学習データにする(hold-out を失う)
- 生成データを chakuho のリポジトリに入れる(個人の画面由来)
- score の蒸留(mac-do は使っていない。choice と noul のみ)
- 学生側で `__none__` を特別扱いする実装(データの比率で対処する)

## 検証 OK 条件(着手前に確定)

1. データ: `build_dataset.py` の実出力に「画面数 / 要素数 / 指示文数 / 変種数 / `__none__` 件数 / noul 件数 / coverage < 0.5 で除外した件数 / 最尤 0.4 未満を保持した件数とその分布」が出る。30 画面以上、10,000 件以上
2. 学習: `train.py` の実出力に epoch ごとの KL と、336 ベンチの正答率が出る
3. 精度(336 ベンチ、`eval.py` の実出力): 学生が同じベンチの 27B から 5 ポイント以内(27B 95% → 90% 以上)、`__none__` ≥ 27/30、シャッフル・混入変種で 5 ポイント以上落ちない
4. hold-out(mac-do 実運用ログ、GPU サーバの decisions-*.jsonl から Calendar/Demo 由来を除いた実運用分): 27B との一致 ≥ 85%
5. 異常系: `build_dataset.py` が「教師の coverage < 0.5 で除外した件数」を出力し、学習データ側にはその条件のケースが残っていない(除外件数 ≥ 0、残存 0)。学生の coverage は教師と同じ定義で測る: `mlx_backend.py` に `top_logprobs=20` を指定して 336 ベンチを流し、`coverage < 0.5` の率が 27B(top 20 で 32/336)以下であること
6. MacBook 実機: `mlx_backend.py` + chakuho で 50 候補の choice が 1 判定 500 ms 以下(mac-do が測る。1.7B の prefill 約 1000 トークンを M 系で回す前提の目安で、超えたら 4bit を測る)。GPU サーバ停止時に `backend: primary`(学生)で mac-do が動く実出力。学生プロセスを止めた時に `backend: fallback`(GPU サーバ)で答え、`/health` が `degraded: true` を返す実出力。学生を戻すと `primary` に復帰する実出力
7. bf16 → 8bit の差が 336 ベンチで 2 ポイント以内。4bit 化した場合も同様に測り README に書く
8. 分布のずれ: 生成データの held-out 分割での 27B 一致率と、条件 4(実運用 hold-out)の一致率の差を出す。差が 10 ポイント超なら、実運用ログの指示文の言い回し(内容ではなく文体)を逆生成のプロンプトへ反映して再生成する
9. 教師バッチの負荷: 逆生成と教師ラベル取得は同時実行 4 に絞り、サーキットブレーカーが実際に働く実出力(人工的に負荷をかけて「一時停止 → 再開」のログが出ること)
10. 段階: 11 画面での配管確認(学習が回り、評価が出る)を先に通す。30 画面のデータが揃う前に条件 3 の判定をしない

## 設計レビュー(3 人格の LLM 合議、CONCERN 2 / 有効 2 of 3)と対応

| 指摘 | 対応 |
|---|---|
| `top_logprobs` 上限 11 では 52 ラベルの coverage が定義できず、条件 5 が成立しない(バルタザール・カスパー・メルキオール) | 学生の配信を `mlx_backend.py`(全ラベルの logprob を返す)にし、coverage を 27B と同じ定義に揃えた。`mlx_lm.server` 経路の影響は 27B で実測(下記) |
| 27B へのフォールバックが mac-do 任せで、常駐が死ぬと運用が止まる(カスパー・メルキオール) | chakuho に予備 backend を実装済み。条件 6 に「学生停止 → fallback / 復帰 → primary」の実出力を追加。切り戻し手順を配布節に明記 |
| 4B へ切り替える基準が数値でない(バルタザール) | 1 epoch 後に 80% 未満または `__none__` 20/30 未満で 4B |
| 8bit 変換の劣化検証が無い(バルタザール) | 条件 7 を bf16 対 8bit の差 2 ポイント以内に |
| 逆生成した指示文と実運用の分布のずれ(バルタザール) | 条件 8 で差を測り、10 ポイント超なら文体を反映して再生成 |
| 教師バッチが実運用中の vLLM を圧迫する(メルキオール) | 同時実行 4、latency 中央値が 2 倍で停止(条件 9) |
| `CHAKUHO_TOP_LOGPROBS` の不正値(メルキオール) | 既定へ倒して警告、テスト追加済み |
| 条件 5 の「0 件」と「除外件数を出す」の二重性(バルタザール) | 除外件数を出し、学習データ側に残存 0、と書き分けた |
| 絶対値 90% は 1.7B に厳しすぎる(カスパー) | 相対(27B から 5 ポイント以内)に書き換え。数値は同じ |
| 300 ms は厳しい(カスパー) | 500 ms へ。超えたら 4bit を測る |
| 教師の迷いを捨てると難例を学べない(自己反証) | 捨てない。soft label のまま学ぶ。除外するのは coverage < 0.5(形を守れなかった応答)だけで、これは迷いとは別の量 |

### `top_logprobs=11` の実測(27B、336 ケース)

同じ 336 ケースを 27B に `top_logprobs=11` と `20` で流した(`bench/` の計測と同じ要求、集約は chakuho core)。

| | top_logprobs=20 | top_logprobs=11 |
|---|---|---|
| choice が同じ | 317/336 (94%) | ← |
| coverage 中央値 | 0.95 | 0.93 |
| coverage 10 パーセンタイル | 0.51 | 0.44 |
| coverage < 0.5 の件数 | 32 | 42 |

11 に落とすと答えが 6% 変わり、coverage < 0.5 が 10 件増える。使えなくはないが、判定の定義が変わる。
学生の配信は全ラベルの logprob を返す `mlx_backend.py` を主にし、`mlx_lm.server` は予備経路とする根拠。

## 設計レビュー 2 回目(FAIL 1 / CONCERN 2)と対応

| 指摘 | 対応 |
|---|---|
| 「迷いは捨てない」と「coverage < 0.5 を除外」が矛盾(FAIL) | 矛盾ではなく別の量だった。coverage の定義(ラベルの形を守れたか)を教師ラベルの節に明記し、条件 1 を「保持した件数」に書き換えた |
| 教師(top 20)と学生(全ラベル)で coverage の定義が違う | 条件 5 は学生にも `top_logprobs=20` を指定して同じ定義で測る |
| 30 画面が「依頼済み」の前提 | データが揃うまで条件 3 は判定しない。11 画面は配管確認だけ(条件 10) |
| 1 epoch で 4B へ切り替える判断は早い | 2 epoch で判定、80〜90% は 4 epoch まで続行 |
| 負荷時の停止が手動 | `build_dataset.py` に自動のサーキットブレーカー。条件 9 で発火の実出力を求める |
| launchd の再起動ループ | `ThrottleInterval` を配布節に明記 |

## 反証してほしい点

- KL 蒸留がラベルトークン 1 位置だけで足りるか。プロンプト全体の LM 損失を混ぜないと学生が「ラベルの形」を保てなくなる懸念
- 逆生成した指示文の分布が実運用の言い回しと違い、hold-out で落ちる可能性。生成時の言い回し指定で足りるか
- 1.7B で 90% に届くか。届かない時に 4B へ行く判断を 1 epoch で下してよいか
- `top_logprobs` 11 で 52 ラベルの coverage が落ちないか(学生は迷いが大きい)
- coverage < 0.5(ラベルの形を守れなかった応答)の除外が、実は難例の除外になっていないか。除外された 32/336 の中身を見て確かめる
