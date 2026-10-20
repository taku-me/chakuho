"""待ち受けアドレスとリクエスト本体の安全側の既定(stdlib のみ)。

このリポジトリのサーバには認証が無い。既定は 127.0.0.1 で待ち受け、LAN へ開く時は
--host か環境変数で明示させ、起動時に警告を出す。
"""

from __future__ import annotations

import ipaddress
import os
import sys
from typing import BinaryIO

DEFAULT_HOST = "127.0.0.1"
DEFAULT_MAX_BODY_BYTES = 8 * 1024 * 1024


class BodyTooLarge(ValueError):
    """Content-Length が上限を超えた(413 で返す)。"""


def resolve_host(cli_host: str | None, env_var: str) -> str:
    """--host > 環境変数 > 127.0.0.1 の順で待ち受けアドレスを決める。"""
    if cli_host:
        return cli_host
    return os.environ.get(env_var) or DEFAULT_HOST


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def warn_if_exposed(host: str, port: int, name: str) -> bool:
    """ループバック以外で待ち受ける時に stderr へ警告し、警告したかを返す。"""
    if is_loopback(host):
        return False
    print(
        f"WARNING: {name} は認証なしで {host}:{port} に待ち受けます。"
        " 届く相手は誰でも推論を呼べます。信頼できるネットワーク内だけで使ってください。",
        file=sys.stderr,
        flush=True,
    )
    return True


def max_body_bytes(env_var: str) -> int:
    try:
        return max(1, int(os.environ.get(env_var, str(DEFAULT_MAX_BODY_BYTES))))
    except ValueError:
        return DEFAULT_MAX_BODY_BYTES


def read_body(headers, rfile: BinaryIO, limit: int) -> bytes:
    """Content-Length ぶんだけ読む。欠落・負・数値でない時は ValueError、上限超えは BodyTooLarge。"""
    raw = headers.get("Content-Length")
    if raw is None:
        raise ValueError("Content-Length is required")
    try:
        length = int(raw)
    except ValueError:
        raise ValueError(f"bad Content-Length: {raw!r}") from None
    if length < 0:
        raise ValueError(f"bad Content-Length: {raw!r}")
    if length > limit:
        raise BodyTooLarge(f"request body too large ({length} > {limit} bytes)")
    return rfile.read(length)
