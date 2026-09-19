# 学習コンテナ: NGC PyTorch(aarch64 + CUDA)に peft を足すだけ。学習コードはホストの repo をマウントする。
ARG BASE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${BASE}
# NGC 同梱の torchao(0.14)は peft 0.21 が拒否する(0.16 以上のみ対応)。使わないので外す
RUN pip uninstall -y torchao && pip install --no-cache-dir "peft>=0.13" "accelerate>=1.0"
WORKDIR /work
