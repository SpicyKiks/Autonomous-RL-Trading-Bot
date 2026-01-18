from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from autonomous_rl_trading_bot.common.config import load_config
from autonomous_rl_trading_bot.common.db import connect, migrate, upsert_run
from autonomous_rl_trading_bot.common.hashing import short_hash
from autonomous_rl_trading_bot.common.logging import configure_logging
from autonomous_rl_trading_bot.common.paths import artifacts_dir, ensure_artifact_tree
from autonomous_rl_trading_bot.common.reproducibility import set_global_seed
from autonomous_rl_trading_bot.rl.dataset import load_dataset_npz, select_latest_dataset
from autonomous_rl_trading_bot.training.trainer import TrainConfig, train_and_evaluate


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_run_id(mode: str, dataset_id: str, algo: str, cfg_hash: str) -> str:
    # Example: 20260109T195326Z_spot_train_ds123_ppo_ab12cd34
    return f"{_utc_ts()}_{mode}_train_{dataset_id}_{algo}_{short_hash(cfg_hash, 8)}"


def _train_day2_parquet(
    args: argparse.Namespace,
    dataset_path: Path,
    logger_info: Any,
) -> int:
    """Day-2 training pipeline using parquet dataset."""
    from autonomous_rl_trading_bot.common.reproducibility import set_global_seed
    from autonomous_rl_trading_bot.training.train_pipeline import (
        load_dataset,
        split_dataset,
        train_ppo,
        evaluate_ppo,
    )
    
    # Set seed
    seed = int(args.seed)
    set_global_seed(seed)
    
    # Handle smoke test
    timesteps = 2000 if args.smoke_test else int(args.timesteps)
    train_split = 0.8 if args.smoke_test else float(args.train_split)
    
    # Convert legacy fee-bps to taker-fee if provided
    taker_fee = float(args.taker_fee)
    if args.fee_bps is not None:
        taker_fee = float(args.fee_bps) / 10000.0
    
    # Load dataset
    if logger_info:
        logger_info(f"[Day-2] Loading dataset: {dataset_path}")
    df = load_dataset(args.symbol, args.interval)
    
    # Smoke test: use smaller slice
    if args.smoke_test and len(df) > 1000:
        df = df.iloc[:1000].copy()
        if logger_info:
            logger_info(f"[Smoke test] Using first 1000 rows")
    
    # Split dataset
    train_df, test_df = split_dataset(df, train_split=train_split)
    
    if logger_info:
        logger_info(f"[Day-2] Dataset path: {dataset_path}")
        logger_info(f"[Day-2] Train size: {len(train_df):,} rows")
        logger_info(f"[Day-2] Test size: {len(test_df):,} rows")
    
    # Setup paths
    run_id = f"{_utc_ts()}_day2_{args.symbol}_{args.interval}_seed{seed}"
    tensorboard_log_dir = str(Path(args.tensorboard_dir) / run_id)
    model_out = args.model_out
    report_out = args.report_out
    
    if logger_info:
        logger_info(f"[Day-2] Run ID: {run_id}")
        logger_info(f"[Day-2] Model output: {model_out}")
        logger_info(f"[Day-2] TensorBoard logs: {tensorboard_log_dir}")
        logger_info(f"[Day-2] Report output: {report_out}")
    
    # Train
    if logger_info:
        logger_info(f"[Day-2] Training PPO for {timesteps:,} timesteps...")
    
    model_path = train_ppo(
        train_df,
        timesteps=timesteps,
        seed=seed,
        tensorboard_log_dir=tensorboard_log_dir,
        model_out=model_out,
        taker_fee=taker_fee,
        slippage_bps=float(args.slippage_bps),
        risk_penalty=float(args.risk_penalty),
        position_fraction=float(args.position_fraction),
    )
    
    # Save run_config.json with symbol/interval/dataset info
    run_dir = Path(tensorboard_log_dir).parent
    run_config_path = run_dir / "run_config.json"
    run_config_path.parent.mkdir(parents=True, exist_ok=True)
    run_config = {
        "run_id": run_id,
        "symbol": args.symbol,
        "interval": args.interval,
        "dataset_path": str(dataset_path),
        "train_split": train_split,
        "taker_fee": taker_fee,
        "slippage_bps": float(args.slippage_bps),
        "risk_penalty": float(args.risk_penalty),
        "position_fraction": float(args.position_fraction),
        "initial_balance": 10000.0,  # Default from TradingEnv
        "seed": seed,
        "model_path": model_path,
    }
    import json
    with open(run_config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    
    if logger_info:
        logger_info(f"[Day-2] Training complete. Model saved: {model_path}")
    
    # Evaluate
    metrics = {}
    if args.eval and not args.no_eval:
        if logger_info:
            logger_info(f"[Day-2] Evaluating on test set...")
        
        metrics = evaluate_ppo(
            model_path,
            test_df,
            seed=seed,
            taker_fee=taker_fee,
            slippage_bps=float(args.slippage_bps),
            risk_penalty=float(args.risk_penalty),
            position_fraction=float(args.position_fraction),
            report_out=report_out,
        )
        
        if logger_info:
            logger_info(f"[Day-2] Evaluation complete. Metrics:")
            logger_info(f"  Total Return: {metrics.get('total_return', 0):.4f}")
            logger_info(f"  Sharpe Ratio: {metrics.get('sharpe', 0):.4f}")
            logger_info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.4f}")
            logger_info(f"  Trades: {metrics.get('trades', 0)}")
            logger_info(f"  Win Rate: {metrics.get('win_rate', 0):.4f}")
            logger_info(f"  Avg Trade PnL: {metrics.get('avg_trade_pnl', 0):.4f}")
    
    print(f"OK: Day-2 training complete (run_id={run_id}, model={model_path}, metrics={report_out})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 7: Offline RL training runner (SB3).")
    parser.add_argument("--mode", default=None, help="spot|futures. Overrides base.yaml.")
    parser.add_argument("--dataset-id", default=None, help="Dataset id under artifacts/datasets/ (default: latest for mode).")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol for Day-2 parquet dataset (e.g., BTCUSDT).")
    parser.add_argument("--interval", default="1m", help="Timeframe for Day-2 parquet dataset (e.g., 1m).")
    parser.add_argument("--algo", default="ppo", help="ppo|dqn")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps.")
    parser.add_argument("--lookback", type=int, default=30, help="Lookback window for observations (validation only for Day-2).")
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Train split fraction (0.0-1.0) for Day-2 parquet, or train|val|test for legacy NPZ.",
    )
    parser.add_argument(
        "--eval-split",
        default=None,
        help="Dataset split for evaluation: train|val|test (default: val from meta, legacy only)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--taker-fee", type=float, default=0.0004, help="Taker fee rate (default 0.0004 = 0.04%).")
    parser.add_argument("--slippage-bps", type=float, default=1.0, help="Slippage in basis points (default 1.0 bps = 0.01%).")
    parser.add_argument("--risk-penalty", type=float, default=0.1, help="Risk penalty coefficient.")
    parser.add_argument("--position-fraction", type=float, default=1.0, help="Fraction of equity to size position.")
    parser.add_argument("--model-out", default="models/ppo_trader.zip", help="Output path for trained model.")
    parser.add_argument("--tensorboard-dir", default="logs/tensorboard", help="Directory for TensorBoard logs.")
    parser.add_argument("--report-out", default="reports/training_metrics.json", help="Output path for evaluation metrics.")
    parser.add_argument("--eval", action="store_true", default=True, help="Run evaluation after training (default: True).")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation after training.")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test (2000 timesteps, smaller dataset).")
    # Legacy args (kept for backward compatibility)
    parser.add_argument("--fee-bps", type=float, default=None, help="[Legacy] Fee bps per trade (use --taker-fee instead).")
    parser.add_argument("--futures-leverage", type=float, default=3.0, help="Leverage used for futures sizing (legacy).")
    parser.add_argument("--reward", default="log_equity", help="log_equity|delta_equity (legacy).")
    args = parser.parse_args(argv)

    ensure_artifact_tree()

    # Check if using Day-2 parquet dataset
    use_day2_parquet = False
    dataset_path = Path("data/processed") / f"{args.symbol.upper()}_{args.interval}_dataset.parquet"
    
    if dataset_path.exists():
        use_day2_parquet = True
        # Day-2 parquet path: use new training pipeline
        return _train_day2_parquet(args, dataset_path, print)
    
    # Fall back to legacy NPZ dataset
    loaded = load_config(mode=args.mode)
    cfg = loaded.config
    cfg_hash = loaded.config_hash
    mode = cfg["mode"]["id"]
    if mode not in ("spot", "futures"):
        raise SystemExit(f"ERROR: mode must be spot|futures, got {mode}")

    # Ensure DB migrated before run creation
    db_path = migrate(cfg)

    # Dataset selection
    datasets_dir = artifacts_dir() / "datasets"
    if args.dataset_id:
        ds_dir = datasets_dir / args.dataset_id
        dataset = load_dataset_npz(ds_dir)
    else:
        dataset = select_latest_dataset(datasets_dir, market_type=mode)

    market_type = str(dataset.meta.get("market_type") or mode)
    dataset_id = dataset.dataset_id
    
    # Legacy NPZ path: continue with existing code

    algo = str(args.algo).strip().lower()
    if algo not in ("ppo", "dqn"):
        raise SystemExit(f"ERROR: --algo must be ppo|dqn, got {algo}")

    run_id = _make_run_id(mode, dataset_id, algo, cfg_hash)
    run_dir = artifacts_dir() / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    log_level = (cfg.get("logging", {}) or {}).get("level", "INFO")
    log_console = bool((cfg.get("logging", {}) or {}).get("console", True))
    log_file = bool((cfg.get("logging", {}) or {}).get("file", True))

    per_run_log: Optional[str] = None
    global_log: Optional[str] = None
    log_paths = []

    if log_file:
        per_run_log_p = run_dir / "run.log"
        global_log_p = artifacts_dir() / "logs" / f"{run_id}.log"
        per_run_log = str(per_run_log_p)
        global_log = str(global_log_p)
        log_paths = [per_run_log, global_log]
        logger = configure_logging(
            level=log_level,
            console=log_console,
            file_paths=[per_run_log_p, global_log_p],
            run_id=run_id,
        )
    else:
        logger = configure_logging(level=log_level, console=log_console, file_paths=None, run_id=run_id)

    seed = int(args.seed if args.seed is not None else int(cfg["run"]["seed"]))
    seed_report = set_global_seed(seed)

    created_utc = _iso_utc_now()
    run_json_path = str(run_dir / "run.json")

    train_split = str(args.train_split).strip().lower() if args.train_split else None
    eval_split = str(args.eval_split).strip().lower() if args.eval_split else None

    tcfg = TrainConfig(
        dataset_id=dataset_id,
        mode=mode,
        algo=algo,  # type: ignore[arg-type]
        timesteps=int(args.timesteps),
        seed=seed,
        train_split=train_split,
        eval_split=eval_split,
        lookback=int(args.lookback),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        initial_equity=1000.0,
        position_fraction=float(args.position_fraction),
        futures_leverage=float(args.futures_leverage),
        reward_kind=str(args.reward),
        run_dir=artifacts_dir() / "runs",
        models_dir=artifacts_dir() / "models",
    )

    run_meta: Dict[str, Any] = {
        "run_id": run_id,
        "created_utc": created_utc,
        "kind": "train",
        "mode": mode,
        "market_type": market_type,
        "dataset_id": dataset_id,
        "config_hash": cfg_hash,
        "config": cfg,
        "seed": seed,
        "seed_report": seed_report,
        "log_paths": log_paths,
        "db_path": str(db_path),
        "status": "CREATED",
        "train_params": {
            "algo": algo,
            "timesteps": int(args.timesteps),
            "lookback": int(args.lookback),
            "train_split": train_split,
            "eval_split": eval_split,
            "fee_bps": float(args.fee_bps),
            "slippage_bps": float(args.slippage_bps),
            "position_fraction": float(args.position_fraction),
            "futures_leverage": float(args.futures_leverage),
            "reward_kind": str(args.reward),
            "dataset_dir": str(dataset.dataset_dir),
            "npz_path": str(dataset.npz_path),
        },
    }
    (run_dir / "run.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # Record in DB: runs(CREATED)
    with connect(db_path) as conn:
        upsert_run(
            conn,
            run_id=run_id,
            kind="train",
            mode=mode,
            created_utc=created_utc,
            config_hash=cfg_hash,
            seed=seed,
            status="CREATED",
            run_dir=str(run_dir),
            run_json_path=run_json_path,
            run_log_path=per_run_log,
            global_log_path=global_log,
        )
        conn.commit()

    logger.info("Training starting")
    logger.info("run_id=%s market_type=%s dataset_id=%s algo=%s timesteps=%s", run_id, market_type, dataset_id, algo, tcfg.timesteps)

    status = "DONE"
    finished_utc: Optional[str] = None
    try:
        with connect(db_path) as conn:
            upsert_run(
                conn,
                run_id=run_id,
                kind="train",
                mode=mode,
                created_utc=created_utc,
                config_hash=cfg_hash,
                seed=seed,
                status="RUNNING",
                run_dir=str(run_dir),
                run_json_path=run_json_path,
                run_log_path=per_run_log,
                global_log_path=global_log,
            )
            conn.commit()

        out = train_and_evaluate(dataset=dataset, cfg=tcfg, run_id=run_id, out_dir=run_dir)
        finished_utc = _iso_utc_now()

        run_meta["status"] = "DONE"
        run_meta["finished_utc"] = finished_utc
        run_meta["train_output"] = out
        (run_dir / "run.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info("Training DONE")
        logger.info("Model: %s", out.get("model_path"))
        logger.info("Metrics: %s", out.get("metrics"))

    except Exception:
        status = "FAILED"
        finished_utc = _iso_utc_now()
        tb = traceback.format_exc()
        (run_dir / "error.txt").write_text(tb, encoding="utf-8")
        run_meta["status"] = "FAILED"
        run_meta["finished_utc"] = finished_utc
        (run_dir / "run.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.error("Training FAILED. See error.txt")
        logger.error(tb)

    with connect(db_path) as conn:
        upsert_run(
            conn,
            run_id=run_id,
            kind="train",
            mode=mode,
            created_utc=created_utc,
            config_hash=cfg_hash,
            seed=seed,
            status=status,
            run_dir=str(run_dir),
            run_json_path=run_json_path,
            run_log_path=per_run_log,
            global_log_path=global_log,
        )
        conn.commit()

    if status == "DONE":
        print(f"OK: training {run_id} DONE (mode={mode}, dataset_id={dataset_id}, run_dir={run_dir})")
        return 0
    print(f"ERROR: training {run_id} FAILED (see {run_dir}/error.txt)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
