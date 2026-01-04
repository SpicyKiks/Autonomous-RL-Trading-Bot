from __future__ import annotations

import argparse
from pathlib import Path

from autonomous_rl_trading_bot.common.config import load_config
from autonomous_rl_trading_bot.common.db import migrate
from autonomous_rl_trading_bot.common.paths import ensure_artifact_tree
from autonomous_rl_trading_bot.dashboard import create_app


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Dash dashboard.")
    p.add_argument("--mode", default=None, help="spot | futures (optional; config merge).")
    p.add_argument("--config", default="base.yaml", help="Base config name under configs/ (default base.yaml).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    ensure_artifact_tree()

    loaded = load_config(mode=args.mode, base_name=args.config)
    cfg = loaded.config

    db_path = Path(migrate(cfg))
    app = create_app(db_path=db_path)

    # Dash >=2 supports app.run(); older supports run_server(). dash pinned in pyproject anyway.
    app.run(host=args.host, port=args.port, debug=bool(args.debug))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
