from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from autonomous_rl_trading_bot.features.indicators import log_returns, returns


def compute_features(
    candles: pd.DataFrame,
    *,
    timestamp_col: str = "open_time_ms",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> Tuple[np.ndarray, list[str]]:
    """
    Compute a deterministic feature matrix from candles.

    Args:
        candles: DataFrame with OHLCV columns
        timestamp_col: Name of timestamp column
        open_col: Name of open price column
        high_col: Name of high price column
        low_col: Name of low price column
        close_col: Name of close price column
        volume_col: Name of volume column

    Returns:
        Tuple of (feature_matrix, column_names)
        feature_matrix: (n_samples, n_features) numpy array
        column_names: List of feature names
    """
    close = candles[close_col].values.astype(np.float64)
    volume = candles[volume_col].values.astype(np.float64)

    # Compute features
    log_ret = log_returns(close)
    ret = returns(close)

    # Normalized close (first close = 1.0)
    first_close = float(close[0]) if len(close) > 0 else 1.0
    close_norm = close / max(first_close, 1e-12)

    # Normalized volume (log1p)
    vol_norm = np.log1p(np.maximum(volume, 0.0))

    # Stack features
    features = np.stack([log_ret, ret, close_norm, vol_norm], axis=1)
    columns = ["log_return", "return", "close_norm", "vol_norm"]

    return features.astype(np.float32), columns

