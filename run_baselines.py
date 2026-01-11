from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from autonomous_rl_trading_bot.common.config import load_config
from autonomous_rl_trading_bot.common.db import connect, migrate
from autonomous_rl_trading_bot.common.logging import configure_logging
from autonomous_rl_trading_bot.common.paths import artifacts_dir, ensure_artifact_tree
from autonomous_rl_trading_bot.common.reproducibility import set_global_seed
from autonomous_rl_trading_bot.evaluation import (
    BacktestConfig,
    compute_expanded_metrics,
    load_dataset,
    make_strategy,
    plot_equity_and_drawdown,
    plot_trades_over_price,
    run_futures_backtest,
    run_spot_backtest,
)
from autonomous_rl_trading_bot.evaluation.reporting import write_equity_csv, write_json, write_trades_csv


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


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

    candidates: List[tuple[Path, float]] = []
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
) -> Path:
    """Return dataset_dir."""
    if dataset_path:
        p = Path(dataset_path)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.is_file() and p.name.lower() == "dataset.npz":
            p = p.parent
        return p

    if dataset_id:
        return artifacts_dir() / "datasets" / dataset_id

    return _latest_dataset_dir(desired_market_type)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 5: Run baseline strategies on TEST split.")
    parser.add_argument("--mode", default=None, help="spot/futures. Used for defaults.")
    parser.add_argument("--dataset-id", default=None, help="Dataset ID")
    parser.add_argument("--dataset-path", default=None, help="Path to dataset directory")
    
    parser.add_argument("--initial-cash", type=float, default=None)
    parser.add_argument("--taker-fee-rate", type=float, default=None)
    parser.add_argument("--slippage-bps", type=float, default=None)
    
    # SMA params
    parser.add_argument("--sma-fast", type=int, default=10, help="SMA fast window")
    parser.add_argument("--sma-slow", type=int, default=30, help="SMA slow window")
    
    # RSI params
    parser.add_argument("--rsi-period", type=int, default=14, help="RSI period")
    parser.add_argument("--rsi-low", type=float, default=30.0, help="RSI low threshold")
    parser.add_argument("--rsi-high", type=float, default=70.0, help="RSI high threshold")
    
    # Futures-only
    parser.add_argument("--leverage", type=float, default=None)
    parser.add_argument("--maintenance-margin-rate", type=float, default=None)
    
    args = parser.parse_args(argv)

    ensure_artifact_tree()

    loaded = load_config(mode=args.mode)
    cfg = loaded.config

    requested_mode = str(args.mode or cfg["mode"]["id"]).strip().lower()
    if requested_mode not in ("spot", "futures"):
        requested_mode = "spot"

    db_path = migrate(cfg)

    bt_cfg = (cfg.get("evaluation", {}) or {}).get("backtest", {}) or {}
    
    initial_cash = float(
        args.initial_cash if args.initial_cash is not None else bt_cfg.get("initial_cash", 1000.0)
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

    # Load dataset with TEST split
    dataset_dir = _resolve_dataset_dir(args.dataset_id, args.dataset_path, requested_mode)
    dataset_meta, arrays = load_dataset(dataset_dir, split="test")
    dataset_id = str(dataset_meta.get("dataset_id") or dataset_dir.name)
    
    dataset_market_type = str(dataset_meta.get("market_type") or requested_mode).strip().lower()
    if dataset_market_type not in ("spot", "futures"):
        dataset_market_type = requested_mode
    
    market_type = dataset_market_type
    
    # Create backtest config with same cost model
    bt_config = BacktestConfig(
        initial_cash=initial_cash,
        order_size_quote=0.0,  # All-in
        taker_fee_rate=taker_fee_rate,
        slippage_bps=slippage_bps,
        leverage=leverage,
        maintenance_margin_rate=maintenance_margin_rate,
        allow_short=True,
        stop_on_liquidation=True,
    )
    
    seed = int(cfg["run"]["seed"])
    set_global_seed(seed)
    
    # Create output directory
    timestamp = _utc_ts()
    run_id = f"{timestamp}_baselines_{dataset_id}"
    output_dir = artifacts_dir() / "baselines" / dataset_id / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_level = (cfg.get("logging", {}) or {}).get("level", "INFO")
    logger = configure_logging(level=log_level, console=True, file_paths=None, run_id=run_id)
    
    logger.info("Step 5: Running baseline strategies on TEST split")
    logger.info("dataset_id=%s market_type=%s", dataset_id, market_type)
    
    # Define strategies
    strategies = [
        ("buy_and_hold", "Buy & Hold", {}),
        ("sma_crossover", f"SMA Crossover ({args.sma_fast}/{args.sma_slow})", {
            "fast": args.sma_fast,
            "slow": args.sma_slow,
        }),
        ("rsi_reversion", f"RSI Mean-Reversion ({args.rsi_period}, {args.rsi_low}/{args.rsi_high})", {
            "period": args.rsi_period,
            "low": args.rsi_low,
            "high": args.rsi_high,
        }),
    ]
    
    results: List[Dict[str, Any]] = []
    
    for strategy_name, strategy_display, strategy_params in strategies:
        logger.info("Running strategy: %s", strategy_display)
        
        strategy = make_strategy(strategy_name, params=strategy_params)
        
        # Run backtest
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
        
        # Extract data for expanded metrics
        equity = [float(r["equity"]) for r in equity_rows]
        drawdown = [float(r["drawdown"]) for r in equity_rows]
        exposure = [float(r.get("exposure", 0.0)) for r in equity_rows]
        open_time_ms = [int(r["open_time_ms"]) for r in equity_rows]
        prices = [float(r["price"]) for r in equity_rows]
        
        interval = str(dataset_meta.get("interval") or "")
        from autonomous_rl_trading_bot.common.timeframes import interval_to_ms
        interval_ms = interval_to_ms(interval) if interval else int(open_time_ms[1] - open_time_ms[0]) if len(open_time_ms) > 1 else 60_000
        
        # Compute expanded metrics
        expanded = compute_expanded_metrics(
            equity=equity,
            drawdown=drawdown,
            exposure=exposure,
            trades=trade_rows,
            interval_ms=interval_ms,
        )
        
        # Save outputs for this strategy
        strategy_dir = output_dir / strategy_name
        strategy_dir.mkdir(parents=True, exist_ok=True)
        
        # Write CSV files
        write_equity_csv(strategy_dir / "equity.csv", equity_rows)
        write_trades_csv(strategy_dir / "trades.csv", trade_rows)
        
        # Write metrics
        metrics_dict = {
            **metrics_obj.to_dict(),
            **expanded.to_dict(),
        }
        write_json(strategy_dir / "metrics.json", metrics_dict)
        
        # Generate plots
        plot_equity_and_drawdown(
            equity=equity,
            drawdown=drawdown,
            open_time_ms=open_time_ms,
            output_path=strategy_dir / "equity_drawdown.png",
            title=f"{strategy_display} - Equity & Drawdown",
        )
        
        plot_trades_over_price(
            prices=prices,
            open_time_ms=open_time_ms,
            trades=trade_rows,
            output_path=strategy_dir / "trades_price.png",
            title=f"{strategy_display} - Trades Over Price",
        )
        
        # Store results for comparison
        results.append({
            "strategy": strategy_name,
            "display": strategy_display,
            "metrics": metrics_dict,
            "output_dir": str(strategy_dir),
        })
        
        logger.info("Completed: %s (Sharpe=%.3f, Return=%.2f%%)", 
                   strategy_display,
                   expanded.sharpe or 0.0,
                   expanded.total_return * 100.0)
    
    # Generate comparison summary
    comparison_path = output_dir / "baseline_comparison.md"
    _write_comparison_summary(comparison_path, results)
    
    logger.info("All baselines completed. Output: %s", output_dir)
    print(f"OK: Baseline comparison saved to {comparison_path}")
    return 0


def _write_comparison_summary(output_path: Path, results: List[Dict[str, Any]]) -> None:
    """Write baseline comparison markdown summary."""
    lines = [
        "# Baseline Strategy Comparison",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        "",
        "Strategies ranked by Sharpe ratio and total return.",
        "",
    ]
    
    # Sort by Sharpe (descending)
    sorted_by_sharpe = sorted(
        results,
        key=lambda x: x["metrics"].get("sharpe") or float("-inf"),
        reverse=True,
    )
    
    lines.append("### Ranked by Sharpe Ratio")
    lines.append("")
    lines.append("| Rank | Strategy | Sharpe | Sortino | Total Return | Max DD | Win Rate | Profit Factor |")
    lines.append("|------|----------|--------|---------|--------------|--------|----------|---------------|")
    
    for rank, result in enumerate(sorted_by_sharpe, 1):
        m = result["metrics"]
        sharpe = m.get("sharpe")
        sortino = m.get("sortino")
        total_ret = m.get("total_return", 0.0)
        max_dd = m.get("max_drawdown", 0.0)
        win_rate = m.get("win_rate")
        profit_factor = m.get("profit_factor")
        
        sharpe_str = f"{sharpe:.3f}" if sharpe is not None else "N/A"
        sortino_str = f"{sortino:.3f}" if sortino is not None else "N/A"
        ret_str = f"{total_ret * 100:.2f}%"
        dd_str = f"{max_dd * 100:.2f}%"
        wr_str = f"{win_rate * 100:.1f}%" if win_rate is not None else "N/A"
        pf_str = f"{profit_factor:.2f}" if profit_factor is not None else "N/A"
        
        lines.append(f"| {rank} | {result['display']} | {sharpe_str} | {sortino_str} | {ret_str} | {dd_str} | {wr_str} | {pf_str} |")
    
    # Sort by total return
    sorted_by_return = sorted(
        results,
        key=lambda x: x["metrics"].get("total_return", 0.0),
        reverse=True,
    )
    
    lines.append("")
    lines.append("### Ranked by Total Return")
    lines.append("")
    lines.append("| Rank | Strategy | Total Return | Sharpe | Max DD | Trade Count | Avg Trade |")
    lines.append("|------|----------|--------------|--------|--------|-------------|-----------|")
    
    for rank, result in enumerate(sorted_by_return, 1):
        m = result["metrics"]
        total_ret = m.get("total_return", 0.0)
        sharpe = m.get("sharpe")
        max_dd = m.get("max_drawdown", 0.0)
        trade_count = m.get("trade_count", 0)
        avg_trade = m.get("avg_trade")
        
        ret_str = f"{total_ret * 100:.2f}%"
        sharpe_str = f"{sharpe:.3f}" if sharpe is not None else "N/A"
        dd_str = f"{max_dd * 100:.2f}%"
        avg_str = f"{avg_trade:.2f}" if avg_trade is not None else "N/A"
        
        lines.append(f"| {rank} | {result['display']} | {ret_str} | {sharpe_str} | {dd_str} | {trade_count} | {avg_str} |")
    
    # Detailed metrics per strategy
    lines.append("")
    lines.append("## Detailed Metrics")
    lines.append("")
    
    for result in results:
        m = result["metrics"]
        lines.append(f"### {result['display']}")
        lines.append("")
        lines.append(f"- **Sharpe Ratio**: {m.get('sharpe', 'N/A')}")
        lines.append(f"- **Sortino Ratio**: {m.get('sortino', 'N/A')}")
        lines.append(f"- **Calmar Ratio**: {m.get('calmar', 'N/A')}")
        lines.append(f"- **Total Return**: {m.get('total_return', 0.0) * 100:.2f}%")
        lines.append(f"- **Max Drawdown**: {m.get('max_drawdown', 0.0) * 100:.2f}%")
        lines.append(f"- **Win Rate**: {m.get('win_rate', 0.0) * 100:.1f}%" if m.get('win_rate') is not None else "- **Win Rate**: N/A")
        lines.append(f"- **Profit Factor**: {m.get('profit_factor', 'N/A')}")
        lines.append(f"- **Avg Trade**: {m.get('avg_trade', 'N/A')}")
        lines.append(f"- **Exposure Avg**: {m.get('exposure_avg', 0.0) * 100:.1f}%")
        lines.append(f"- **Turnover**: {m.get('turnover', 0.0):.2f}x")
        lines.append(f"- **Trade Count**: {m.get('trade_count', 0)}")
        lines.append(f"- **Output Directory**: `{result['output_dir']}`")
        lines.append("")
    
    output_path.write_text("\n".join(lines), encoding="utf-8")

