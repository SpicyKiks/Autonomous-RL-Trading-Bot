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
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest evaluation")
    
    # Mode A: Backtest by run-id
    backtest_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID to load dataset and model from (mutually exclusive with --symbol/--interval/--model)",
    )
    
    # Mode B: Backtest by explicit params
    backtest_parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Trading symbol (e.g., BTCUSDT) - required if --run-id not provided",
    )
    backtest_parser.add_argument(
        "--interval",
        type=str,
        default=None,
        help="Timeframe interval (e.g., 1m) - required if --run-id not provided",
    )
    backtest_parser.add_argument(
        "--model",
        type=str,
        default="models/ppo_trader.zip",
        help="Path to SB3 model file (.zip) - required if --policy=ppo, ignored otherwise",
    )
    backtest_parser.add_argument(
        "--policy",
        type=str,
        choices=["ppo", "buyhold", "sma", "rsi"],
        default="ppo",
        help="Policy to backtest: ppo (requires --model), buyhold, sma, or rsi (baseline strategies)",
    )
    
    # Common args
    backtest_parser.add_argument(
        "--mode",
        type=str,
        choices=["spot", "futures"],
        required=True,
        help="Market mode: spot or futures",
    )
    backtest_parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Train/test split ratio (default: 0.8)",
    )
    backtest_parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for reports (default: reports)",
    )

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
    
    # =========================
    # REPRO COMMAND
    # =========================
    repro_parser = subparsers.add_parser("repro", help="Run complete reproducibility pipeline")
    repro_parser.add_argument("--symbol", type=str, required=True, help="Trading symbol (e.g., BTCUSDT)")
    repro_parser.add_argument("--interval", type=str, required=True, help="Timeframe interval (e.g., 1m)")
    repro_parser.add_argument("--days", type=int, required=True, help="Number of days of data to download")
    repro_parser.add_argument("--window", type=int, default=30, help="State window size (default: 30)")
    repro_parser.add_argument("--timesteps", type=int, default=200000, help="Training timesteps (default: 200000)")
    repro_parser.add_argument("--mode", type=str, choices=["spot", "futures"], required=True, help="Market mode")
    repro_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    repro_parser.add_argument("--output-dir", type=str, default="reports/repro", help="Output directory (default: reports/repro)")
    
    # =========================
    # VERIFY COMMAND
    # =========================
    verify_parser = subparsers.add_parser("verify", help="Run verification tests (pytest + smoke repro)")

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
        from autonomous_rl_trading_bot.backtest.runner import run_backtest
        
        # Validate arguments
        if args.run_id and (args.symbol or args.interval):
            print("Error: --run-id cannot be used together with --symbol/--interval")
            backtest_parser.print_help()
            return 2
        
        if not args.run_id and (not args.symbol or not args.interval):
            print("Error: Either --run-id or both --symbol and --interval must be provided")
            backtest_parser.print_help()
            return 2
        
        # Validate policy-specific requirements
        if args.policy == "ppo" and not args.model:
            print("Error: --model is required when --policy=ppo")
            backtest_parser.print_help()
            return 2
        
        try:
            result = run_backtest(
                mode=args.mode,
                policy=args.policy,
                model_path=args.model if args.policy == "ppo" else None,
                symbol=args.symbol,
                interval=args.interval,
                run_id=args.run_id,
                train_split=args.train_split,
                output_dir=args.output_dir,
            )
            
            print(f"\nBacktest complete!")
            print(f"  Report JSON: {result['report_json']}")
            print(f"  Trades CSV: {result['trades_csv']}")
            print(f"  Equity CSV: {result['equity_csv']}")
            print(f"  Total Return: {result['metrics']['total_return']:.4f}")
            print(f"  Sharpe Ratio: {result['metrics']['sharpe']:.4f}")
            print(f"  Max Drawdown: {result['metrics']['max_drawdown']:.4f}")
            print(f"  Trades: {result['num_trades']}")
            
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1

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
    
    if args.command == "repro":
        from autonomous_rl_trading_bot.repro.runner import run_repro
        
        try:
            result = run_repro(
                symbol=args.symbol,
                interval=args.interval,
                days=args.days,
                window=args.window,
                timesteps=args.timesteps,
                mode=args.mode,
                seed=args.seed,
                output_dir=args.output_dir,
            )
            
            print(f"\n✓ Reproducibility run complete!")
            print(f"  Run ID: {result['run_id']}")
            print(f"  Run directory: {result['run_dir']}")
            print(f"  Comparison: {result['comparison']}")
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1
    
    if args.command == "verify":
        return _run_verify()

    parser.print_help()
    return 2


def _run_verify() -> int:
    """Run verification tests."""
    import subprocess
    
    print("=" * 60)
    print("VERIFICATION TEST SUITE")
    print("=" * 60)
    
    all_passed = True
    
    # 1. Run pytest
    print("\n[1/3] Running pytest...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        capture_output=True,
        text=True,
    )
    
    if result.returncode == 0:
        print("  ✓ pytest PASSED")
    else:
        print(f"  ✗ pytest FAILED")
        print(result.stdout)
        print(result.stderr)
        all_passed = False
    
    # 2. Quick smoke repro
    print("\n[2/3] Running smoke reproducibility test...")
    try:
        from autonomous_rl_trading_bot.repro.runner import run_repro
        
        repro_result = run_repro(
            symbol="BTCUSDT",
            interval="1m",
            days=2,  # Minimal data
            window=30,
            timesteps=2000,  # Minimal training
            mode="spot",
            seed=42,
            output_dir="reports/verify",
        )
        
        # Verify outputs exist
        run_dir = Path(repro_result["run_dir"])
        required_files = [
            "run_config.json",
            "training_metrics.json",
            "ppo/backtest_report.json",
            "sma/backtest_report.json",
            "comparison.json",
            "comparison.md",
        ]
        
        missing = []
        for file in required_files:
            if not (run_dir / file).exists():
                missing.append(file)
        
        if missing:
            print(f"  ✗ Missing files: {missing}")
            all_passed = False
        else:
            print("  ✓ Smoke repro PASSED")
            print(f"    Run ID: {repro_result['run_id']}")
    except Exception as e:
        print(f"  ✗ Smoke repro FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # 3. Quick dataset load test
    print("\n[3/3] Testing dataset load...")
    try:
        from autonomous_rl_trading_bot.training.train_pipeline import load_dataset
        
        df = load_dataset("BTCUSDT", "1m")
        if len(df) > 0:
            print(f"  ✓ Dataset load PASSED ({len(df)} rows)")
        else:
            print("  ✗ Dataset load FAILED (empty dataset)")
            all_passed = False
    except Exception as e:
        print(f"  ✗ Dataset load FAILED: {e}")
        all_passed = False
    
    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL VERIFICATION TESTS PASSED")
        print("=" * 60)
        return 0
    else:
        print("✗ SOME VERIFICATION TESTS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
