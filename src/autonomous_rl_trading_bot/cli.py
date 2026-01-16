#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autonomous_rl_trading_bot.version import __version__


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="arbt",
        description="Autonomous RL Trading Bot - Unified CLI",
    )
    parser.add_argument("--version", action="version", version=__version__)

    subparsers = parser.add_subparsers(dest="command", required=True)

    # =========================
    # DATASET COMMAND (with subcommands)
    # =========================
    dataset_parser = subparsers.add_parser("dataset", help="Dataset operations")
    dataset_sub = dataset_parser.add_subparsers(dest="subcommand", required=True)

    # dataset fetch
    fetch_parser = dataset_sub.add_parser("fetch", help="Fetch market data")
    fetch_parser.add_argument("--mode", default=None)
    fetch_parser.add_argument("--symbol", default=None)
    fetch_parser.add_argument("--interval", default=None)
    fetch_parser.add_argument("--minutes", type=int, default=None)

    # dataset build
    build_parser = dataset_sub.add_parser("build", help="Build dataset")
    build_parser.add_argument("--mode", default=None)
    build_parser.add_argument("--symbol", default=None)
    build_parser.add_argument("--interval", default=None)
    build_parser.add_argument("--minutes", type=int, default=None)
    build_parser.add_argument("--strict", action="store_true")
    build_parser.add_argument("--no-strict", action="store_true")
    build_parser.add_argument("--train-frac", type=float, default=0.75)
    build_parser.add_argument("--val-frac", type=float, default=0.10)
    build_parser.add_argument("--test-frac", type=float, default=0.15)
    build_parser.add_argument("--scaler", default="robust")

    # =========================
    # TRAIN COMMAND
    # =========================
    train_parser = subparsers.add_parser("train", help="Train RL model")

    # =========================
    # BACKTEST COMMAND
    # =========================
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")

    # =========================
    # LIVE COMMAND
    # =========================
    live_parser = subparsers.add_parser("live", help="Run live demo")

    # =========================
    # DASHBOARD COMMAND
    # =========================
    dashboard_parser = subparsers.add_parser("dashboard", help="Launch dashboard")

    # =========================
    # BASELINES COMMAND
    # =========================
    baselines_parser = subparsers.add_parser("baselines", help="Run baselines")

    args, remainder = parser.parse_known_args(argv)

    if args.command == "dataset":
        if args.subcommand == "fetch":
            from autonomous_rl_trading_bot.run_fetch import main as fetch_main

            forwarded: list[str] = []

            if args.mode:
                forwarded += ["--mode", args.mode]
            if args.symbol:
                forwarded += ["--symbol", args.symbol]
            if args.interval:
                forwarded += ["--interval", args.interval]
            if args.minutes is not None:
                forwarded += ["--minutes", str(args.minutes)]

            forwarded += remainder
            return int(fetch_main(forwarded))

        if args.subcommand == "build":
            from autonomous_rl_trading_bot.run_dataset import main as dataset_main

            forwarded: list[str] = []

            if args.mode:
                forwarded += ["--mode", args.mode]
            if args.symbol:
                forwarded += ["--symbol", args.symbol]
            if args.interval:
                forwarded += ["--interval", args.interval]
            if args.minutes is not None:
                forwarded += ["--minutes", str(args.minutes)]

            forwarded += ["--train-frac", str(args.train_frac)]
            forwarded += ["--val-frac", str(args.val_frac)]
            forwarded += ["--test-frac", str(args.test_frac)]
            forwarded += ["--scaler", str(args.scaler)]

            if args.strict:
                forwarded += ["--strict"]
            if args.no_strict:
                forwarded += ["--no-strict"]

            forwarded += remainder
            return int(dataset_main(forwarded))

        dataset_parser.print_help()
        return 2

    if args.command == "train":
        from autonomous_rl_trading_bot.run_train import main as train_main
        return int(train_main(remainder))

    if args.command == "backtest":
        from autonomous_rl_trading_bot.run_backtest import main as backtest_main
        return int(backtest_main(remainder))

    if args.command == "live":
        # Add project root to path to import run_live_demo
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        import run_live_demo
        return int(run_live_demo.main(remainder))

    if args.command == "dashboard":
        # Add project root to path to import run_dashboard
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        import run_dashboard
        return int(run_dashboard.main(remainder))

    if args.command == "baselines":
        from autonomous_rl_trading_bot.evaluation.baselines import main as baselines_main
        return int(baselines_main(remainder))

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
