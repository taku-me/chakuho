# 学習コンテナ: NGC PyTorch(aarch64 + CUDA)に peft を足すだけ。学習コードはホストの repo をマウントする。
ARG BASE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${BASE}
RUN pip install --no-cache-dir "peft>=0.13" "accelerate>=1.0"
WORKDIR /work
