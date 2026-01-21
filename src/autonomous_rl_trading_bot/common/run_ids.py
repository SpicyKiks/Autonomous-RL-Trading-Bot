"""Centralized run ID generation utilities."""

from __future__ import annotations

from datetime import UTC, datetime

from .hashing import short_hash


def utc_timestamp_compact() -> str:
    """Generate UTC timestamp in compact format: YYYYMMDDTHHMMSSZ."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def utc_timestamp_micro() -> str:
    """Generate UTC timestamp with microseconds: YYYYMMDD_HHMMSS_ffffff."""
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")


def generate_run_id(
    kind: str,
    mode: str,
    *,
    dataset_id: str | None = None,
    algo: str | None = None,
    symbol: str | None = None,
    interval: str | None = None,
    cfg_hash: str | None = None,
    seed: int | None = None,
    window: int | None = None,
    timesteps: int | None = None,
) -> str:
    """
    Generate a standardized run ID.
    
    Format: {timestamp}_{mode}_{kind}_{tags}_{hash}
    
    Args:
        kind: Run kind (train, backtest, live, dataset, etc.)
        mode: Market mode (spot, futures)
        dataset_id: Optional dataset ID
        algo: Optional algorithm (ppo, dqn)
        symbol: Optional trading symbol
        interval: Optional timeframe interval
        cfg_hash: Optional config hash (for reproducibility)
        seed: Optional seed (for reproducibility)
        window: Optional window size
        timesteps: Optional timesteps
    
    Returns:
        Standardized run ID string
    """
    ts = utc_timestamp_compact()
    parts = [ts, mode, kind]
    
    # Add tags based on kind
    if kind == "train":
        if dataset_id:
            parts.append(dataset_id)
        if algo:
            parts.append(algo)
    elif kind == "live":
        if symbol and interval:
            parts.append(f"{symbol}_{interval}")
    elif kind == "backtest":
        if symbol and interval:
            parts.append(f"{symbol}_{interval}")
        if window is not None:
            parts.append(f"w{window}")
        if timesteps is not None:
            parts.append(f"t{timesteps}")
        if seed is not None:
            parts.append(f"s{seed}")
    elif kind == "dataset":
        if symbol and interval:
            parts.append(f"{symbol}_{interval}")
    
    # Add hash if provided
    if cfg_hash:
        parts.append(short_hash(cfg_hash, 8))
    
    return "_".join(parts)
