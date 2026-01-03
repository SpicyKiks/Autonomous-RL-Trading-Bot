from __future__ import annotations

import os
from typing import Any, Optional

import ccxt

from autonomous_rl_trading_bot.data.exchange_client import CcxtClient, ExchangeFetchSpec
from autonomous_rl_trading_bot.data.ohlcv import OhlcvBar


def _truthy(name: str) -> bool:
    v = (os.getenv(name, "") or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def make_binance_futures_ccxt(*, api_key: Optional[str] = None, api_secret: Optional[str] = None) -> Any:
    # USD-M futures (most common)
    ex = ccxt.binanceusdm(
        {
            "enableRateLimit": True,
            "apiKey": api_key or os.getenv("BINANCE_API_KEY", ""),
            "secret": api_secret or os.getenv("BINANCE_API_SECRET", ""),
        }
    )
    use_testnet = _truthy("USE_TESTNET")
    if hasattr(ex, "set_sandbox_mode"):
        ex.set_sandbox_mode(use_testnet)
    return ex


class BinanceFuturesClient(CcxtClient):
    def __init__(self, *, api_key: Optional[str] = None, api_secret: Optional[str] = None) -> None:
        super().__init__(
            exchange=make_binance_futures_ccxt(api_key=api_key, api_secret=api_secret),
            exchange_id="binance",
            market_type="futures",
            ccxt_market_type="future",
        )

    def fetch_ohlcv(self, spec: ExchangeFetchSpec) -> list[OhlcvBar]:
        return super().fetch_ohlcv(spec)

