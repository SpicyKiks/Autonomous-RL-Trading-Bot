from __future__ import annotations

import argparse
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from autonomous_rl_trading_bot.common.config import load_config
from autonomous_rl_trading_bot.common.db import connect, migrate, upsert_run
from autonomous_rl_trading_bot.common.hashing import short_hash
from autonomous_rl_trading_bot.common.logging import configure_logging
from autonomous_rl_trading_bot.common.paths import artifacts_dir, ensure_artifact_tree
from autonomous_rl_trading_bot.common.reproducibility import set_global_seed
from autonomous_rl_trading_bot.evaluation import (
    BacktestConfig,
    load_dataset,
    make_strategy,
    persist_backtest_to_db,
    persist_futures_backtest_to_db,
    plot_equity_and_drawdown,
    plot_price_with_trades,
    run_futures_backtest,
    run_spot_backtest,
    write_run_summary,
)
from autonomous_rl_trading_bot.evaluation.reporting import (
    write_equity_csv,
    write_json,
    write_trades_csv,
)


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _norm(name: str) -> str:
    return (name or "").strip().lower().replace("-", "_").replace(" ", "_")


def _make_run_id(market_type: str, cfg_hash: str, dataset_id: str, strategy: str) -> str:
    # Collision-proof: add a short random suffix so same-second runs can never collide.
    # Determinism of backtest is unaffected (run_id is only an identifier).
    rand = secrets.token_hex(4)  # 8 hex chars
    base = f"{_utc_ts()}_{market_type}_backtest_{dataset_id}_{strategy}"
    return f"{base}_{short_hash(cfg_hash, 8)}_{rand}"


def _read_dataset_market_type(ds_dir: Path) -> Optional[str]:
    mp = ds_dir / "meta.json"
    if not mp.exists():
        return None
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
        mt = str(meta.get("market_type") or "").strip().lower()
        return mt if mt in ("spot", "futures") else None
    except Exception:
        return None


def _latest_dataset_dir(desired_market_type: str) -> Path:
    base = artifacts_dir() / "datasets"
    if not base.exists():
        raise FileNotFoundError(f"datasets folder not found: {base}")

    desired = str(desired_market_type or "").strip().lower()
    if desired not in ("spot", "futures"):
        desired = "spot"

    candidates: List[Tuple[Path, float]] = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        if not (p / "dataset.npz").exists():
            continue
        mt = _read_dataset_market_type(p)
        if mt == desired:
            candidates.append((p, p.stat().st_mtime))

    if not candidates:
        raise FileNotFoundError(f"No datasets found in {base} matching market_type={desired}")

    candidates.sort(key=lambda x: x[1])
    return candidates[-1][0]


def _resolve_dataset_dir(
    dataset_id: Optional[str],
    dataset_path: Optional[str],
    desired_market_type: str,
) -> Tuple[Path, bool]:
    """Return (dataset_dir, was_explicit)."""
    if dataset_path:
        p = Path(dataset_path)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.is_file() and p.name.lower() == "dataset.npz":
            p = p.parent
        return p, True

    if dataset_id:
        return artifacts_dir() / "datasets" / dataset_id, True

    return _latest_dataset_dir(desired_market_type), False


def _strategy_default_params(cfg: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
    bt_cfg = (cfg.get("evaluation", {}) or {}).get("backtest", {}) or {}
    strat_cfg = bt_cfg.get("strategies", {}) or {}
    want = _norm(strategy_name)
    for k, v in strat_cfg.items():
        if _norm(str(k)) == want and isinstance(v, dict):
            return dict(v)
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest runner (deterministic offline engine).")

    parser.add_argument("--mode", default=None, help="spot/futures. Used for defaults only.")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--split", default="test", help="Dataset split to use: train|val|test (default: test)")

    parser.add_argument("--initial-cash", type=float, default=None)
    parser.add_argument("--order-size-quote", type=float, default=None)
    parser.add_argument("--taker-fee-rate", type=float, default=None)
    parser.add_argument("--slippage-bps", type=float, default=None)

    # futures-only overrides (ignored for spot engine)
    parser.add_argument("--leverage", type=float, default=None)
    parser.add_argument("--maintenance-margin-rate", type=float, default=None)
    parser.add_argument("--allow-short", type=int, default=None, help="1/0 futures-only")
    parser.add_argument("--stop-on-liquidation", type=int, default=None, help="1/0 futures-only")

    # Step 6 baseline params
    parser.add_argument("--fast", type=int, default=None)
    parser.add_argument("--slow", type=int, default=None)
    parser.add_argument("--rsi-period", type=int, default=None)
    parser.add_argument("--rsi-low", type=float, default=None)
    parser.add_argument("--rsi-high", type=float, default=None)

    args = parser.parse_args()

    ensure_artifact_tree()

    loaded = load_config(mode=args.mode)
    cfg = loaded.config
    cfg_hash = loaded.config_hash

    requested_mode = str(args.mode or cfg["mode"]["id"]).strip().lower()
    if requested_mode not in ("spot", "futures"):
        requested_mode = "spot"

    # Apply migrations (this will also apply the new uniqueness migration)
    db_path = migrate(cfg)

    bt_cfg = (cfg.get("evaluation", {}) or {}).get("backtest", {}) or {}
    strategy_name = str(args.strategy or bt_cfg.get("strategy") or "buy_and_hold")
    strategy_norm = _norm(strategy_name)

    initial_cash = float(
        args.initial_cash if args.initial_cash is not None else bt_cfg.get("initial_cash", 1000.0)
    )
    order_size_quote = float(
        args.order_size_quote
        if args.order_size_quote is not None
        else bt_cfg.get("order_size_quote", 0.0)
    )
    taker_fee_rate = float(
        args.taker_fee_rate
        if args.taker_fee_rate is not None
        else bt_cfg.get("taker_fee_rate", 0.001)
    )
    slippage_bps = float(
        args.slippage_bps if args.slippage_bps is not None else bt_cfg.get("slippage_bps", 5)
    )

    leverage = float(args.leverage if args.leverage is not None else bt_cfg.get("leverage", 3.0))
    maintenance_margin_rate = float(
        args.maintenance_margin_rate
        if args.maintenance_margin_rate is not None
        else bt_cfg.get("maintenance_margin_rate", 0.005)
    )

    allow_short = bt_cfg.get("allow_short", True)
    if args.allow_short is not None:
        allow_short = bool(int(args.allow_short))

    stop_on_liquidation = bt_cfg.get("stop_on_liquidation", True)
    if args.stop_on_liquidation is not None:
        stop_on_liquidation = bool(int(args.stop_on_liquidation))

    # Mode-aware dataset auto-pick
    dataset_dir, was_explicit = _resolve_dataset_dir(
        args.dataset_id, args.dataset_path, requested_mode
    )
    
    # Get split (default to test for backtests)
    split = str(args.split).strip().lower() if args.split else "test"
    if split not in ("train", "val", "test"):
        raise SystemExit(f"ERROR: --split must be train|val|test, got {split}")
    
    # Enforce test split unless explicitly overridden
    if not was_explicit and split != "test":
        logger.warning(
            "Backtests should use test split. Override with --split if needed. Using split=%s", split
        )
    
    dataset_meta, arrays = load_dataset(dataset_dir, split=split)
    dataset_id = str(dataset_meta.get("dataset_id") or dataset_dir.name)

    dataset_market_type = str(dataset_meta.get("market_type") or requested_mode).strip().lower()
    if dataset_market_type not in ("spot", "futures"):
        dataset_market_type = requested_mode

    # Truth for execution + DB is dataset market_type
    market_type = dataset_market_type

    # Strategy params (yaml defaults + CLI overrides)
    strategy_params: Dict[str, Any] = _strategy_default_params(cfg, strategy_norm)

    if args.fast is not None:
        strategy_params["fast"] = int(args.fast)
    if args.slow is not None:
        strategy_params["slow"] = int(args.slow)
    if args.rsi_period is not None:
        strategy_params["period"] = int(args.rsi_period)
    if args.rsi_low is not None:
        strategy_params["low"] = float(args.rsi_low)
    if args.rsi_high is not None:
        strategy_params["high"] = float(args.rsi_high)

    run_id = _make_run_id(market_type, cfg_hash, dataset_id, strategy_norm)

    run_dir = artifacts_dir() / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    log_level = (cfg.get("logging", {}) or {}).get("level", "INFO")
    log_console = bool((cfg.get("logging", {}) or {}).get("console", True))
    log_file = bool((cfg.get("logging", {}) or {}).get("file", True))

    per_run_log: Optional[str] = None
    global_log: Optional[str] = None
    log_paths: List[str] = []

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
        logger = configure_logging(
            level=log_level, console=log_console, file_paths=None, run_id=run_id
        )

    if was_explicit and market_type != requested_mode:
        logger.warning(
            "Requested mode=%s but dataset market_type=%s (explicit dataset provided). Proceeding with market_type=%s.",
            requested_mode,
            market_type,
            market_type,
        )

    seed = int(cfg["run"]["seed"])
    seed_report = set_global_seed(seed)

    created_utc = datetime.now(timezone.utc).isoformat()
    run_json_path = str(run_dir / "run.json")

    # Create initial run record (mode stored as truth market_type)
    with connect(db_path) as conn:
        upsert_run(
            conn,
            run_id=run_id,
            kind="backtest",
            mode=market_type,
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

    logger.info(
        "Step 6 backtest starting. run_id=%s market_type=%s dataset_id=%s",
        run_id,
        market_type,
        dataset_id,
    )
    logger.info("strategy=%s params=%s", strategy_norm, strategy_params)

    started_utc = datetime.now(timezone.utc).isoformat()

    bt_config = BacktestConfig(
        initial_cash=initial_cash,
        order_size_quote=order_size_quote,
        taker_fee_rate=taker_fee_rate,
        slippage_bps=slippage_bps,
        leverage=leverage,
        maintenance_margin_rate=maintenance_margin_rate,
        allow_short=bool(allow_short),
        stop_on_liquidation=bool(stop_on_liquidation),
    )

    params_json = json.dumps(
        {
            "strategy": strategy_norm,
            "strategy_params": strategy_params,
            "requested_mode": requested_mode,
            "market_type": market_type,
            "initial_cash": initial_cash,
            "order_size_quote": order_size_quote,
            "taker_fee_rate": taker_fee_rate,
            "slippage_bps": slippage_bps,
            "leverage": leverage,
            "maintenance_margin_rate": maintenance_margin_rate,
            "allow_short": bool(allow_short),
            "stop_on_liquidation": bool(stop_on_liquidation),
        },
        ensure_ascii=False,
    )

    try:
        strategy = make_strategy(strategy_norm, params=strategy_params)

        if market_type == "futures":
            equity_rows, trade_rows, metrics_obj, extra = run_futures_backtest(
                dataset_meta=dataset_meta,
                arrays=arrays,
                strategy=strategy,
                cfg=bt_config,
            )
        else:
            equity_rows, trade_rows, metrics_obj, extra = run_spot_backtest(
                dataset_meta=dataset_meta,
                arrays=arrays,
                strategy=strategy,
                cfg=bt_config,
            )

        metrics = {
            **(metrics_obj.to_dict()),
            "dataset_id": dataset_id,
            "requested_mode": requested_mode,
            "market_type": market_type,
            "symbol": dataset_meta.get("symbol"),
            "interval": dataset_meta.get("interval"),
            "strategy": strategy_norm,
            "strategy_params": strategy_params,
            "extra": extra,
        }

        equity_csv_path = run_dir / "equity.csv"
        trades_csv_path = run_dir / "trades.csv"
        metrics_json_path = run_dir / "metrics.json"
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Write CSV files
        write_equity_csv(equity_csv_path, equity_rows)
        write_trades_csv(trades_csv_path, trade_rows, include_pnl=True)
        write_json(metrics_json_path, metrics)

        # Generate plots
        equity_vals = [float(r["equity"]) for r in equity_rows]
        drawdown_vals = [float(r["drawdown"]) for r in equity_rows]
        open_time_ms_vals = [int(r["open_time_ms"]) for r in equity_rows]
        prices = [float(r["price"]) for r in equity_rows]

        plot_equity_and_drawdown(
            equity=equity_vals,
            drawdown=drawdown_vals,
            open_time_ms=open_time_ms_vals,
            output_path=plots_dir / "equity_drawdown.png",
            title=f"{strategy_norm} - Equity & Drawdown",
        )

        # Try to get OHLC data for candlestick, fallback to line plot
        open_prices = None
        high_prices = None
        low_prices = None
        if "open" in arrays and "high" in arrays and "low" in arrays:
            # Map prices to equity_rows timestamps
            price_map = {int(arrays["open_time_ms"][i]): i for i in range(len(arrays["open_time_ms"]))}
            open_prices = [float(arrays["open"][price_map.get(ms, 0)]) if ms in price_map else prices[i] for i, ms in enumerate(open_time_ms_vals)]
            high_prices = [float(arrays["high"][price_map.get(ms, 0)]) if ms in price_map else prices[i] for i, ms in enumerate(open_time_ms_vals)]
            low_prices = [float(arrays["low"][price_map.get(ms, 0)]) if ms in price_map else prices[i] for i, ms in enumerate(open_time_ms_vals)]

        plot_price_with_trades(
            prices=prices,
            open_time_ms=open_time_ms_vals,
            trades=trade_rows,
            output_path=plots_dir / "price_trades.png",
            title=f"{strategy_norm} - Price with Trade Markers",
            use_candlestick=(open_prices is not None and high_prices is not None and low_prices is not None),
            open_prices=open_prices,
            high_prices=high_prices,
            low_prices=low_prices,
        )

        # Write run summary
        split_used = split if split != "test" or was_explicit else None  # Only show if explicitly set or not default

        write_run_summary(
            output_path=run_dir / "run_summary.md",
            run_id=run_id,
            dataset_id=dataset_id,
            dataset_meta=dataset_meta,
            split_used=split_used,
            config_snapshot={
                "initial_cash": initial_cash,
                "order_size_quote": order_size_quote,
                "taker_fee_rate": taker_fee_rate,
                "slippage_bps": slippage_bps,
                "leverage": leverage if market_type == "futures" else None,
                "maintenance_margin_rate": maintenance_margin_rate if market_type == "futures" else None,
                "allow_short": bool(allow_short) if market_type == "futures" else None,
                "stop_on_liquidation": bool(stop_on_liquidation) if market_type == "futures" else None,
            },
            metrics=metrics,
            strategy_name=strategy_norm,
            strategy_params=strategy_params,
        )

        finished_utc = datetime.now(timezone.utc).isoformat()

        with connect(db_path) as conn:
            if market_type == "futures":
                status = "LIQUIDATED" if bool(extra.get("liquidated", False)) else "DONE"
                persist_futures_backtest_to_db(
                    conn=conn,
                    backtest_id=run_id,
                    run_id=run_id,
                    mode=market_type,
                    dataset_meta=dataset_meta,
                    params_json=params_json,
                    metrics=metrics_obj,
                    equity_rows=equity_rows,
                    trade_rows=trade_rows,
                    started_utc=started_utc,
                    finished_utc=finished_utc,
                    status=status,
                )
            else:
                persist_backtest_to_db(
                    conn=conn,
                    backtest_id=run_id,
                    run_id=run_id,
                    mode=market_type,
                    dataset_meta=dataset_meta,
                    cfg=bt_config,
                    params_json=params_json,
                    metrics=metrics_obj,
                    equity_rows=equity_rows,
                    trade_rows=trade_rows,
                    started_utc=started_utc,
                    finished_utc=finished_utc,
                    status="DONE",
                )

            upsert_run(
                conn,
                run_id=run_id,
                kind="backtest",
                mode=market_type,
                created_utc=created_utc,
                config_hash=cfg_hash,
                seed=seed,
                status=(
                    "DONE"
                    if market_type != "futures"
                    else ("LIQUIDATED" if bool(extra.get("liquidated", False)) else "DONE")
                ),
                run_dir=str(run_dir),
                run_json_path=run_json_path,
                run_log_path=per_run_log,
                global_log_path=global_log,
            )
            conn.commit()

        run_meta: Dict[str, Any] = {
            "run_id": run_id,
            "created_utc": created_utc,
            "kind": "backtest",
            "requested_mode": requested_mode,
            "market_type": market_type,
            "dataset_id": dataset_id,
            "dataset_dir": str(dataset_dir),
            "strategy": strategy_norm,
            "strategy_params": strategy_params,
            "backtest_config": json.loads(params_json),
            "metrics": metrics,
            "equity_csv": str(equity_csv_path),
            "trades_csv": str(trades_csv_path),
            "metrics_json": str(metrics_json_path),
            "config_hash": cfg_hash,
            "db_path": str(db_path),
            "seed_report": seed_report,
            "log_paths": log_paths,
            "status": (
                "DONE"
                if market_type != "futures"
                else ("LIQUIDATED" if bool(extra.get("liquidated", False)) else "DONE")
            ),
        }
        (run_dir / "run.json").write_text(
            json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print(
            f"OK: backtest {run_id} {run_meta['status']} (market_type={market_type}, strategy={strategy_norm})"
        )
        return 0

    except Exception as e:  # noqa: BLE001
        logger.exception("Backtest failed")
        finished_utc = datetime.now(timezone.utc).isoformat()

        with connect(db_path) as conn:
            upsert_run(
                conn,
                run_id=run_id,
                kind="backtest",
                mode=market_type,
                created_utc=created_utc,
                config_hash=cfg_hash,
                seed=seed,
                status="FAILED",
                run_dir=str(run_dir),
                run_json_path=run_json_path,
                run_log_path=per_run_log,
                global_log_path=global_log,
            )
            conn.commit()

        run_meta = {
            "run_id": run_id,
            "created_utc": created_utc,
            "kind": "backtest",
            "requested_mode": requested_mode,
            "market_type": market_type,
            "dataset_id": dataset_id,
            "dataset_dir": str(dataset_dir),
            "strategy": strategy_norm,
            "strategy_params": strategy_params,
            "backtest_config": json.loads(params_json),
            "config_hash": cfg_hash,
            "db_path": str(db_path),
            "seed_report": seed_report,
            "log_paths": log_paths,
            "status": "FAILED",
            "error": repr(e),
            "started_utc": started_utc,
            "finished_utc": finished_utc,
        }
        (run_dir / "run.json").write_text(
            json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print(f"ERROR: backtest failed (run_id={run_id}). See {run_dir}/run.log")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
