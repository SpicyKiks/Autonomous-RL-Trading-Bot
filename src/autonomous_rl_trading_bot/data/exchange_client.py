from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Optional

from autonomous_rl_trading_bot.data.ohlcv import MarketType, OhlcvBar, validate_bars


class NetworkDisabledError(RuntimeError):
    """Raised when code attempts to hit the network without explicit enablement."""


def _truthy_env(name: str) -> bool:
    v = (os.getenv(name, "") or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def network_allowed() -> bool:
    """Hard safety gate for ANY network calls.

    Enable explicitly with:
      - ALLOW_NETWORK=true   (recommended for dataset fetch)
    """
    return _truthy_env("ALLOW_NETWORK")


def require_network() -> None:
    if not network_allowed():
        raise NetworkDisabledError(
            "Network is disabled. Set ALLOW_NETWORK=true in your environment/.env to enable."
        )


def to_ccxt_symbol(symbol: str) -> str:
    """Accepts either:
      - 'BTC/USDT' (ccxt style) -> returned unchanged
      - 'BTCUSDT' (binance style) -> heuristically converted to 'BTC/USDT'

    Heuristic is deliberately simple but works for the common quote assets.
    If you use something non-standard, pass ccxt format explicitly.
    """
    s = (symbol or "").strip().upper()
    if "/" in s:
        return s

    common_quotes = ("USDT", "USDC", "BUSD", "BTC", "ETH", "BNB", "FDUSD", "DAI")
    for q in common_quotes:
        if s.endswith(q) and len(s) > len(q):
            base = s[: -len(q)]
            return f"{base}/{q}"

    # Fallback: best effort, but signal to user
    raise ValueError(f"Cannot infer ccxt symbol from {symbol!r}. Use ccxt format like 'BTC/USDT'.")


@dataclass(frozen=True, slots=True)
class ExchangeFetchSpec:
    symbol: str
    timeframe: str
    since_ms: Optional[int] = None
    limit: int = 1000


class ExchangeClient(ABC):
    """Abstract market-data client. (Public OHLCV only for now.)"""

    exchange_id: str
    market_type: MarketType

    @abstractmethod
    def fetch_ohlcv(self, spec: ExchangeFetchSpec) -> list[OhlcvBar]:
        """Return OHLCV bars in ascending open-time order."""
        raise NotImplementedError

    def close(self) -> None:
        """Optional cleanup hook."""
        return


# ---- CCXT implementation base ----

CcxtMarketType = Literal["spot", "future"]


class CcxtClient(ExchangeClient):
    """CCXT-backed implementation with network safety gate.

    This is safe-by-default: it refuses to fetch unless ALLOW_NETWORK=true.
    """

    def __init__(
        self,
        *,
        exchange: Any,
        exchange_id: str,
        market_type: MarketType,
        ccxt_market_type: CcxtMarketType,
    ) -> None:
        self._ex = exchange
        self.exchange_id = exchange_id
        self.market_type = market_type
        self._ccxt_market_type = ccxt_market_type
        self._markets_loaded = False

    def _ensure_markets(self) -> None:
        if self._markets_loaded:
            return
        # loading markets is a network call for most exchanges
        require_network()
        self._ex.load_markets()
        self._markets_loaded = True

    def fetch_ohlcv(self, spec: ExchangeFetchSpec) -> list[OhlcvBar]:
        require_network()
        self._ensure_markets()

        sym = to_ccxt_symbol(spec.symbol)
        # CCXT timeframe uses same strings like '1m', '5m', '1h', '1d'
        rows = self._ex.fetch_ohlcv(sym, timeframe=spec.timeframe, since=spec.since_ms, limit=spec.limit)
        bars = [OhlcvBar.from_ccxt_row(r) for r in rows]
        bars.sort(key=lambda b: b.ts_ms)

        # CCXT can sometimes return duplicate timestamps if exchange is weird.
        if bars:
            validate_bars(bars, interval=None, allow_gaps=True)

        return bars

    def close(self) -> None:
        try:
            # CCXT: some exchanges expose close()
            if hasattr(self._ex, "close"):
                self._ex.close()
        except Exception:
            # Don't explode during shutdown
            return

