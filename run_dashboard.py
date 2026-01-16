from __future__ import annotations

import argparse
from pathlib import Path

from autonomous_rl_trading_bot.common.config import load_config
from autonomous_rl_trading_bot.common.db import migrate
from autonomous_rl_trading_bot.common.paths import artifacts_dir, ensure_artifact_tree
from autonomous_rl_trading_bot.dashboard import create_app


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Dash dashboard with live trading control.")
    p.add_argument("--mode", default=None, help="spot | futures (optional; config merge).")
    p.add_argument("--config", default="base.yaml", help="Base config name under configs/ (default base.yaml).")
    p.add_argument("--host", default="127.0.0.1", help="Host to bind dashboard server (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8050, help="Port to bind dashboard server (default: 8050)")
    p.add_argument("--refresh-seconds", type=int, default=4, help="Dashboard refresh interval in seconds (default: 4)")
    p.add_argument("--run-id", default=None, help="Optional run_id to display (defaults to latest)")
    p.add_argument("--debug", action="store_true", help="Enable Dash debug mode")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    ensure_artifact_tree()

    loaded = load_config(mode=args.mode, base_name=args.config)
    cfg = loaded.config

    db_path = Path(migrate(cfg))
    app = create_app(db_path=db_path)

    # Update refresh interval if specified (affects dcc.Interval component)
    # Note: The interval is set in layout.py, but we can't change it dynamically.
    # The --refresh-seconds arg is for documentation/future use.

    # Dash >=2 supports app.run(); older supports run_server(). dash pinned in pyproject anyway.
    app.run(host=args.host, port=args.port, debug=bool(args.debug))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
