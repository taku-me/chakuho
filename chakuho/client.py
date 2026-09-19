"""chakuho サーバへ判定を依頼する Python クライアント(stdlib のみ)。

    from chakuho.client import decide
    answers = decide({"screen": "..."}, {"target": {"type": "choice", "criteria": [...]}})
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

DEFAULT_URL = "http://localhost:9750/v1/systemone"


class ChakuhoError(RuntimeError):
    """サーバが 2xx 以外を返した、または到達できなかった。"""


def decide(
    state: Any,
    questions: dict[str, dict[str, Any]],
    url: str | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Jev 互換リクエストを送り、レスポンス JSON(answers / usage / latency_ms / model)を返す。

    url 省略時は env CHAKUHO_URL、それも無ければ DEFAULT_URL。
    """
    target = url or os.environ.get("CHAKUHO_URL", DEFAULT_URL)
    body = json.dumps({"state": state, "questions": questions}).encode("utf-8")
    req = urllib_request.Request(
        target, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ChakuhoError(f"chakuho {target} returned HTTP {exc.code}: {detail}") from exc
    except (urllib_error.URLError, TimeoutError, OSError) as exc:
        raise ChakuhoError(f"chakuho {target} unreachable: {exc}") from exc
