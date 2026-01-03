from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def write_run_summary(
    output_path: Path,
    *,
    run_id: str,
    dataset_id: str,
    dataset_meta: Dict[str, Any],
    split_used: Optional[str],
    config_snapshot: Dict[str, Any],
    metrics: Dict[str, Any],
    strategy_name: str,
    strategy_params: Dict[str, Any],
) -> None:
    """
    Write run_summary.md with config snapshot, dataset info, and key metrics.
    
    Args:
        output_path: Path to save run_summary.md
        run_id: Backtest run ID
        dataset_id: Dataset ID used
        dataset_meta: Dataset metadata
        split_used: Split used (train/val/test or None)
        config_snapshot: Backtest configuration snapshot
        metrics: Full metrics dictionary
        strategy_name: Strategy name
        strategy_params: Strategy parameters
    """
    lines = [
        "# Backtest Run Summary",
        "",
        f"**Run ID**: `{run_id}`",
        f"**Generated**: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Configuration",
        "",
        f"- **Strategy**: {strategy_name}",
        f"- **Strategy Parameters**:",
    ]
    
    for key, value in strategy_params.items():
        lines.append(f"  - `{key}`: {value}")
    
    lines.extend([
        "",
        "### Backtest Config",
        "",
        "```json",
        json.dumps(config_snapshot, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Dataset",
        "",
        f"- **Dataset ID**: `{dataset_id}`",
        f"- **Market Type**: {dataset_meta.get('market_type', 'N/A')}",
        f"- **Symbol**: {dataset_meta.get('symbol', 'N/A')}",
        f"- **Interval**: {dataset_meta.get('interval', 'N/A')}",
    ])
    
    if split_used:
        lines.append(f"- **Split Used**: `{split_used}`")
        splits = dataset_meta.get("splits", {})
        if split_used in splits:
            split_info = splits[split_used]
            lines.append(f"  - Start Index: {split_info.get('start_idx', 'N/A')}")
            lines.append(f"  - End Index: {split_info.get('end_idx', 'N/A')}")
            if "start_time_ms" in split_info and split_info["start_time_ms"] is not None:
                try:
                    start_dt = datetime.fromtimestamp(split_info["start_time_ms"] / 1000, tz=timezone.utc)
                    lines.append(f"  - Start Time: {start_dt.isoformat()}")
                except (ValueError, TypeError, OSError):
                    pass
            if "end_time_ms" in split_info and split_info["end_time_ms"] is not None:
                try:
                    end_dt = datetime.fromtimestamp(split_info["end_time_ms"] / 1000, tz=timezone.utc)
                    lines.append(f"  - End Time: {end_dt.isoformat()}")
                except (ValueError, TypeError, OSError):
                    pass
    
    lines.extend([
        "",
        "## Key Metrics",
        "",
    ])
    
    # Format metrics nicely
    def _format_metric(key: str, value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            if "return" in key.lower() or "rate" in key.lower():
                return f"{value * 100:.2f}%"
            elif "sharpe" in key.lower() or "sortino" in key.lower() or "calmar" in key.lower() or "factor" in key.lower():
                return f"{value:.3f}"
            elif "duration" in key.lower():
                return f"{value:.1f} days"
            else:
                return f"{value:.2f}"
        return str(value)
    
    key_metrics = [
        ("Total Return", "total_return"),
        ("Annualized Return", "annualized_return"),
        ("Annualized Volatility", "annualized_volatility"),
        ("Sharpe Ratio", "sharpe"),
        ("Sharpe CI (95%)", lambda m: f"[{_format_metric('', m.get('sharpe_ci_lower'))}, {_format_metric('', m.get('sharpe_ci_upper'))}]" if m.get('sharpe_ci_lower') is not None else "N/A"),
        ("Sortino Ratio", "sortino"),
        ("Calmar Ratio", "calmar"),
        ("Max Drawdown", "max_drawdown"),
        ("Max Drawdown Duration", "max_drawdown_duration_days"),
        ("CAGR", "cagr"),
        ("Win Rate", "win_rate"),
        ("Profit Factor", "profit_factor"),
        ("Avg Trade PnL", "avg_trade_pnl"),
        ("Trade Count", "trade_count"),
        ("Turnover", "turnover"),
        ("Exposure Avg", "exposure_avg"),
        ("Exposure Max", "exposure_max"),
        ("Fee Total", "fee_total"),
        ("Slippage Total", "slippage_total"),
    ]
    
    for label, key_or_func in key_metrics:
        if callable(key_or_func):
            value_str = key_or_func(metrics)
        else:
            value = metrics.get(key_or_func)
            value_str = _format_metric(key_or_func, value)
        lines.append(f"- **{label}**: {value_str}")
    
    lines.extend([
        "",
        "## Files",
        "",
        "- `equity.csv` - Equity and drawdown curves",
        "- `trades.csv` - Full trade ledger with PnL",
        "- `metrics.json` - Complete metrics",
        "- `plots/equity_drawdown.png` - Equity and drawdown visualization",
        "- `plots/price_trades.png` - Price chart with trade markers",
        "",
    ])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

