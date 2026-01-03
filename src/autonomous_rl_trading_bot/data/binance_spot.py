from __future__ import annotations

import os
from typing import Any, Optional

import ccxt

from autonomous_rl_trading_bot.data.exchange_client import CcxtClient, ExchangeFetchSpec
from autonomous_rl_trading_bot.data.ohlcv import MarketType, OhlcvBar


def _truthy(name: str) -> bool:
    v = (os.getenv(name, "") or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def make_binance_spot_ccxt(*, api_key: Optional[str] = None, api_secret: Optional[str] = None) -> Any:
    ex = ccxt.binance(
        {
            "enableRateLimit": True,
            "apiKey": api_key or os.getenv("BINANCE_API_KEY", ""),
            "secret": api_secret or os.getenv("BINANCE_API_SECRET", ""),
        }
    )
    # Testnet support (public OHLCV works either way, but keep consistent with your env flags)
    use_testnet = _truthy("USE_TESTNET")
    if hasattr(ex, "set_sandbox_mode"):
        ex.set_sandbox_mode(use_testnet)
    return ex


class BinanceSpotClient(CcxtClient):
    def __init__(self, *, api_key: Optional[str] = None, api_secret: Optional[str] = None) -> None:
        super().__init__(
            exchange=make_binance_spot_ccxt(api_key=api_key, api_secret=api_secret),
            exchange_id="binance",
            market_type="spot",
            ccxt_market_type="spot",
        )

    # signature preserved for clarity when used elsewhere
    def fetch_ohlcv(self, spec: ExchangeFetchSpec) -> list[OhlcvBar]:
        return super().fetch_ohlcv(spec)

