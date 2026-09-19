"""chakuho コマンド: serve(サーバ起動)/ ask(JSON リクエストを投げて答えを表示)。"""

from __future__ import annotations

import argparse
import json
import sys

from chakuho import __version__
from chakuho.client import ChakuhoError, decide


def _cmd_serve(args: argparse.Namespace) -> int:
    from chakuho import server

    httpd = server.serve(args.host, args.port)
    backend = server._config_from_env()["backend_url"]
    print(f"chakuho {__version__} listening on {args.host}:{args.port} -> {backend}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    raw = sys.stdin.read() if args.request == "-" else open(args.request, encoding="utf-8").read()
    req = json.loads(raw)
    try:
        out = decide(req["state"], req["questions"], url=args.url, timeout=args.timeout)
    except ChakuhoError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chakuho", description=__doc__)
    parser.add_argument("--version", action="version", version=f"chakuho {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Jev 互換の判定サーバを起動する")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=9750)
    p_serve.set_defaults(func=_cmd_serve)

    p_ask = sub.add_parser("ask", help="リクエスト JSON(ファイルか - で stdin)を投げて答えを表示する")
    p_ask.add_argument("request", help="Jev 形式のリクエスト JSON ファイル。- で stdin")
    p_ask.add_argument("--url", default=None, help="サーバ URL(既定: env CHAKUHO_URL か localhost:9750)")
    p_ask.add_argument("--timeout", type=float, default=60.0)
    p_ask.set_defaults(func=_cmd_ask)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
