#!/usr/bin/env python3
"""Live demo runner (placeholder for later steps).

Step-0 goal: this file must not be empty and must give you a safe, offline smoke-check.

Usage:
    python run_live_demo.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from autonomous_rl_trading_bot.common.config import load_config
from autonomous_rl_trading_bot.common.paths import ensure_artifact_tree
from autonomous_rl_trading_bot.version import __version__


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml", help="Path to base config yaml")
    parser.add_argument("--mode", default=None, help="Optional mode override (configs/modes/<mode>.yaml)")
    args = parser.parse_args()

    ensure_artifact_tree()

    base_name = Path(args.config).name
    loaded = load_config(mode=args.mode, base_name=base_name)
    cfg = loaded.config

    print(f"autonomous-rl-trading-bot v{__version__}")
    print("Live mode is not implemented yet (will be built in later steps).\n")
    print("Loaded config:")
    print(f"  project.name={cfg.get('project', {}).get('name')}")
    print(f"  run.seed={cfg.get('run', {}).get('seed')}")
    print("\nNext:")
    print("  - implement exchange clients + paper broker")
    print("  - implement SpotEnv/FuturesEnv")
    print("  - implement a live loop with safeguards\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
