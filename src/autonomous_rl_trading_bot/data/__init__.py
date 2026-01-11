from __future__ import annotations

from .ohlcv import MarketType, OhlcvBar, bars_from_ccxt, bars_to_frame, frame_to_bars, validate_bars
from .exchange_client import (
    ExchangeClient,
    ExchangeFetchSpec,
    NetworkDisabledError,
    network_allowed,
    require_network,
    to_ccxt_symbol,
)

# Lazy imports for CCXT clients to avoid import-time network dependencies
# These classes are only imported when explicitly requested, not at module import time


def _lazy_import_binance_spot():
    """Lazy import BinanceSpotClient to avoid ccxt import at module level."""
    try:
        from .binance_spot import BinanceSpotClient
        return BinanceSpotClient
    except ImportError as e:
        raise ImportError(
            "BinanceSpotClient requires ccxt. Install: pip install ccxt"
        ) from e


def _lazy_import_binance_futures():
    """Lazy import BinanceFuturesClient to avoid ccxt import at module level."""
    try:
        from .binance_futures import BinanceFuturesClient
        return BinanceFuturesClient
    except ImportError as e:
        raise ImportError(
            "BinanceFuturesClient requires ccxt. Install: pip install ccxt"
        ) from e


# Provide lazy accessors that import on demand
def get_binance_spot_client(*args, **kwargs):
    """Get BinanceSpotClient (lazy import)."""
    cls = _lazy_import_binance_spot()
    return cls(*args, **kwargs)


def get_binance_futures_client(*args, **kwargs):
    """Get BinanceFuturesClient (lazy import)."""
    cls = _lazy_import_binance_futures()
    return cls(*args, **kwargs)


# For backward compatibility, we can still expose the class names
# but they will raise ImportError if ccxt is not available
def __getattr__(name: str):
    """Lazy attribute access for CCXT clients."""
    if name == "BinanceSpotClient":
        return _lazy_import_binance_spot()
    if name == "BinanceFuturesClient":
        return _lazy_import_binance_futures()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MarketType",
    "OhlcvBar",
    "bars_from_ccxt",
    "bars_to_frame",
    "frame_to_bars",
    "validate_bars",
    "ExchangeClient",
    "ExchangeFetchSpec",
    "NetworkDisabledError",
    "network_allowed",
    "require_network",
    "to_ccxt_symbol",
    "BinanceSpotClient",  # Lazy import via __getattr__
    "BinanceFuturesClient",  # Lazy import via __getattr__
    "get_binance_spot_client",
    "get_binance_futures_client",
]