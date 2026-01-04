"""Live (paper) trading utilities.

This package intentionally defaults to **paper trading** on public market data
(Binance klines via :mod:`autonomous_rl_trading_bot.exchange.binance_public`).

The design goals are:

* Safe-by-default execution (no API keys required).
* Deterministic, reproducible portfolio accounting (fees/slippage).
* Guardrails (kill switch file, max drawdown, max trade rate).
* Incremental persistence to SQLite for later analysis.
"""

from .live_runner import LiveRunner, LiveRunnerConfig

__all__ = ["LiveRunner", "LiveRunnerConfig"]

