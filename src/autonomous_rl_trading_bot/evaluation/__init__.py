"""Evaluation / offline analysis utilities (backtesting, metrics, reporting)."""

from .backtester import (
    BacktestConfig,
    load_dataset,
    persist_backtest_to_db,
    persist_futures_backtest_to_db,
    run_futures_backtest,
    run_spot_backtest,
)
from .baselines import Strategy, make_strategy
from .expanded_metrics import ExpandedMetrics, compute_expanded_metrics
from .metrics import BacktestMetrics, compute_metrics
from .plotting import plot_equity_and_drawdown, plot_price_with_trades, plot_trades_over_price

__all__ = [
    "BacktestConfig",
    "BacktestMetrics",
    "ExpandedMetrics",
    "Strategy",
    "compute_metrics",
    "compute_expanded_metrics",
    "load_dataset",
    "make_strategy",
    "persist_backtest_to_db",
    "persist_futures_backtest_to_db",
    "plot_equity_and_drawdown",
    "plot_price_with_trades",
    "plot_trades_over_price",
    "run_spot_backtest",
    "run_futures_backtest",
    "write_run_summary",
]
