# chakuho 蒸留

27B(chakuho backend、vLLM)の判定分布を、mac 常駐用の小型 Qwen(LoRA)へ蒸留する。
設計・検証 OK 条件は [`docs/distill-design.md`](../docs/distill-design.md) を参照。

依存はステップごとに分かれている。本体(`chakuho/`)は stdlib のみのまま保つため、
学習系の依存(`torch` / `transformers` / `peft` / `datasets`)はこのパッケージには含めず、
学習は別環境(CUDA コンテナ)で行う。

## 手順

### ① 指示文の逆生成 + ② リクエスト合成 + ③ 教師ラベル

マスク済み AX スナップショット(`{"apps": {app: [element, ...]}}`)から、27B(chakuho backend)
を使って指示文と教師分布を作る。stdlib のみ、この repo の Python でそのまま動く。

```bash
python3 -m distill.gen_instructions snapshot1.json snapshot2.json \
    --out gen/ --backend http://<gpu-host>:8006/v1 --per-element 3 --none 5

python3 -m distill.build_dataset --gen gen/ --snapshots snapshot1.json snapshot2.json \
    --out dataset.jsonl --backend http://<gpu-host>:8006/v1 --variants 3 --inflight 4
```

`build_dataset.py` は共有 vLLM への負荷をサーキットブレーカーで自動抑制する(latency
中央値が開始時の 2 倍を超えたら一時停止)。除外件数(`coverage < 0.5`)と、保持した
低確信度件数(`max_p < 0.4`)を実出力に出す。`--resume` は既存の出力にある id を飛ばして追記する(backend 断からの再開用)。除外した例は `<out>.dropped.jsonl` に
残す(内訳の確認用。学習には使わない)。

### ④ 学習(CUDA コンテナ内)

学習コードはこの repo をマウントし、コンテナ側で `peft` / `accelerate` を足した
イメージを使う(`distill/docker/Dockerfile`、ベースは `nvcr.io/nvidia/pytorch:<tag>`
— aarch64 + CUDA 版のタグを環境に合わせて選ぶ)。

```bash
docker build -t chakuho-distill-train -f distill/docker/Dockerfile distill/docker

docker run --rm --gpus all \
    -v "$(pwd)":/work -w /work \
    chakuho-distill-train \
    python3 -m distill.train \
        --data dataset.jsonl \
        --model Qwen/Qwen3-1.7B \
        --out out/qwen3-1.7b-distill \
        --epochs 4 --lr 1e-4 --lora-r 16 --batch 8 --lm-weight 0.0 \
        --eval-cases cases.json --eval-cases cases2.json \
        --eval-cases2 holdout.json \
        --eval-every-epoch --seed 0 --merge
```

- 損失: プロンプト末尾 1 位置のラベルトークン上に制限した KL(teacher ‖ student)。
  `--lm-weight w > 0` でプロンプト全体の LM 損失を重み `w` で加算する(検証 OK 条件 11 の比較用)
- 各 epoch の終わりにアダプタを `out/.../epoch-N/` へ保存し、`--eval-cases` が
  指定されていれば(`--eval-every-epoch` 無指定時は最終 epoch のみ)336 ベンチを
  in-process で評価する(サーバ不要、`distill/eval.py` の `score_cases` を使う)。
  `--eval-cases2` は hold-out(実運用ログ由来のケース)を別枠で評価する
- epoch ごとの KL 平均・件数・評価結果を JSON 行として標準出力へ流す
- `--limit N` で学習例数を絞って配管確認(smoke test)ができる
- `--merge` で最後に LoRA を `merge_and_unload()` してフルモデルを `out/.../merged/` に保存する
  (`mlx_lm.convert` の入力になる)

torch/transformers/peft が無い環境でも `distill/train.py` は import・`--help` ができる
(重い依存は遅延 import で、`label_token_ids` 等の純粋関数は torch 抜きでテストできる)。
実際の学習はこれらの依存が揃った環境でのみ実行できる。

### 変換(mac、MLX)

```bash
mlx_lm.fuse --model out/qwen3-1.7b-distill/merged --adapter-path out/qwen3-1.7b-distill/epoch-4 \
    --save-path models/2026-09-xx-qwen3-1.7b/bf16
mlx_lm.convert --hf-path models/2026-09-xx-qwen3-1.7b/bf16 \
    --mlx-path models/2026-09-xx-qwen3-1.7b/8bit --q-bits 8

ln -sfn "$(pwd)/models/2026-09-xx-qwen3-1.7b/8bit" models/current
```

世代ディレクトリ(`models/<日付-学生名>/`)を上書きせず、`models/current` の symlink を
切り替えて配信する。切り戻しは symlink を戻すだけ(重みは消さない)。

### ⑤ 評価

```bash
# サーバ経由(chakuho が student backend を向いている状態、または backend を直接指定)
python3 -m distill.eval --url http://<mac-host>:9750 --cases cases.json --cases2 cases2.json

# ライブラリとして(train.py が使っているのと同じ経路、サーバ不要)
python3 -c "
from distill.eval import score_cases, load_cases
cases = load_cases('cases.json', 'cases2.json')
print(score_cases(cases, my_decide_fn))
"
```

出力は JSON 1 行(全体・変種別・アプリ別・`__none__` 課題別の正答率、`coverage < 0.5`
の率)と、人間向けの表。

### ⑥ 配信(mac)

```bash
python3 -m distill.mlx_backend --model models/current --host 0.0.0.0 --port 8006
```

OpenAI 互換の `/v1/models`・`/v1/chat/completions`(`max_tokens: 1` 専用)を出す。
`top_logprobs` はリクエスト側の指定を尊重し(既定 64)、宣言ラベル全部を確実に含められる
ようにする(教師と同じ定義で coverage を測るため)。mlx_lm がインストールされた mac
上でのみモデルロードが動く(この開発機には無いため、request/response のヘルパー関数は
fake モデルでユニットテストしている)。

chakuho を student backend に向ける:

```bash
export CHAKUHO_BACKEND_URL=http://localhost:8006/v1
export CHAKUHO_FALLBACK_BACKEND_URL=http://<gpu-host>:8006/v1   # GPU サーバの vLLM(27B)を予備に
uv run chakuho serve --host 0.0.0.0 --port 9750
```

`/health` は主 backend が死んでいて予備が生きていれば `{"ok": true, "degraded": true, ...}`
を返す。判定応答には `"backend": "primary" | "fallback"` が付く。切り戻しは
`CHAKUHO_BACKEND_URL` を GPU サーバへ戻して chakuho を再起動するだけ(学生の重みは消さない)。

## テスト

```bash
python3 -m pytest tests/test_distill.py -q
```

torch/mlx が無い環境でも通る(`distill/eval.py` は stdlib のみ、`distill/mlx_backend.py`
と `distill/train.py` は重い依存の呼び出しを fake で差し替えてテストする)。
