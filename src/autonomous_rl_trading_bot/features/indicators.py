from __future__ import annotations

import numpy as np


def returns(prices: np.ndarray) -> np.ndarray:
    """Compute simple returns: (p[t] / p[t-1]) - 1."""
    if len(prices) < 2:
        return np.zeros_like(prices)
    prev = prices[:-1]
    curr = prices[1:]
    return np.where(prev > 0, (curr / prev) - 1.0, 0.0)


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns: log(p[t] / p[t-1])."""
    if len(prices) < 2:
        return np.zeros_like(prices)
    prev = prices[:-1]
    curr = prices[1:]
    return np.where(prev > 0, np.log(np.maximum(curr, 1e-12) / prev), 0.0)


def sma(values: np.ndarray, window: int) -> np.ndarray:
    """Simple Moving Average."""
    if window < 1 or len(values) < window:
        return np.zeros_like(values)
    result = np.zeros_like(values)
    for i in range(window - 1, len(values)):
        result[i] = np.mean(values[i - window + 1 : i + 1])
    return result


def ema(values: np.ndarray, span: int) -> np.ndarray:
    """Exponential Moving Average."""
    if span < 1 or len(values) == 0:
        return np.zeros_like(values)
    alpha = 2.0 / (span + 1.0)
    result = np.zeros_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1]
    return result


def rsi(values: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index."""
    if period < 1 or len(values) < period + 1:
        return np.zeros_like(values)
    deltas = np.diff(values)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    result = np.zeros_like(values)
    # Simple average for first period
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))
    # Wilder's smoothing for subsequent periods
    for i in range(period + 1, len(values)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    return result


def zscore(values: np.ndarray, window: int) -> np.ndarray:
    """Z-score: (x - mean) / std over rolling window."""
    if window < 1 or len(values) < window:
        return np.zeros_like(values)
    result = np.zeros_like(values)
    for i in range(window - 1, len(values)):
        window_vals = values[i - window + 1 : i + 1]
        mean = np.mean(window_vals)
        std = np.std(window_vals)
        if std > 1e-12:
            result[i] = (values[i] - mean) / std
    return result


def clip(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Clip values to [lower, upper]."""
    return np.clip(values, float(lower), float(upper))


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range (ATR) - measures market volatility."""
    if period < 1 or len(high) < period + 1:
        return np.zeros_like(close)
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("high, low, close must have same length")
    
    # True Range: max(high-low, abs(high-close_prev), abs(low-close_prev))
    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    # ATR is EMA of TR
    return ema(tr, span=period)


def volatility(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling volatility (standard deviation of returns)."""
    if window < 1 or len(prices) < window + 1:
        return np.zeros_like(prices)
    
    # Compute returns
    ret = returns(prices)
    
    # Rolling standard deviation
    result = np.zeros_like(prices)
    for i in range(window - 1, len(ret)):
        window_vals = ret[i - window + 1 : i + 1]
        result[i] = np.std(window_vals)
    
    return result


def volume_delta(volume: np.ndarray, window: int = 20) -> np.ndarray:
    """Volume delta: change in volume relative to rolling average."""
    if window < 1 or len(volume) < window:
        return np.zeros_like(volume)
    
    # Rolling average volume
    avg_volume = sma(volume, window)
    
    # Delta: (current - average) / average
    result = np.zeros_like(volume)
    for i in range(window - 1, len(volume)):
        if avg_volume[i] > 1e-12:
            result[i] = (volume[i] - avg_volume[i]) / avg_volume[i]
    
    return result
