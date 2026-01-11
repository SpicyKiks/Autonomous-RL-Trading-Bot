from __future__ import annotations

from .dataset import load_dataset_npz, select_latest_dataset

# Lazy import for TradingEnv to avoid gymnasium import at module level
# This prevents import-time failures if gymnasium is not installed


def _lazy_import_trading_env():
    """Lazy import TradingEnv to avoid gymnasium import at module level."""
    try:
        from .env_trading import TradingEnv, TradingEnvConfig
        return TradingEnv, TradingEnvConfig
    except ImportError as e:
        raise ImportError(
            "TradingEnv requires gymnasium. Install: pip install gymnasium"
        ) from e


def __getattr__(name: str):
    """Lazy attribute access for gymnasium-dependent classes."""
    if name == "TradingEnv":
        TradingEnv, _ = _lazy_import_trading_env()
        return TradingEnv
    if name == "TradingEnvConfig":
        _, TradingEnvConfig = _lazy_import_trading_env()
        return TradingEnvConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "load_dataset_npz",
    "select_latest_dataset",
    "TradingEnv",  # Lazy import via __getattr__
    "TradingEnvConfig",  # Lazy import via __getattr__
]
