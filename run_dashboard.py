#!/usr/bin/env python3
"""Run the local dashboard.

Usage:
    python run_dashboard.py
"""

from __future__ import annotations

import argparse

from autonomous_rl_trading_bot.dashboard import create_dash_app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_dash_app()
    app.run_server(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

