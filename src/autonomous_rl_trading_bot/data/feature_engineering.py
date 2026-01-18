from __future__ import annotations

import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute causal technical indicators on OHLCV.

    Requires columns:
      timestamp_ms, open, high, low, close, volume
    """
    d = df.copy()
    d = d.sort_values("timestamp_ms").reset_index(drop=True)

    close = d["close"].astype(float)
    high = d["high"].astype(float)
    low = d["low"].astype(float)
    vol = d["volume"].astype(float)

    d["log_return"] = np.log(close).diff()

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["tr"] = tr
    d["atr_14"] = tr.rolling(14, min_periods=14).mean()

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    period = 14
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0.0, np.nan))
    d["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    hist = macd - signal
    d["ema_12"] = ema12
    d["ema_26"] = ema26
    d["macd"] = macd
    d["macd_signal"] = signal
    d["macd_hist"] = hist

    d["volatility_30"] = d["log_return"].rolling(30, min_periods=30).std()
    d["volume_delta"] = vol.diff()

    d["datetime_utc"] = pd.to_datetime(d["timestamp_ms"], unit="ms", utc=True)
    return d


def causal_zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Causal expanding z-score:
      mean/std computed on history up to t-1 (shifted), so current row never influences its own scaling.
    """
    out = df.copy()
    for c in cols:
        x = out[c].astype(float)
        mu = x.expanding(min_periods=50).mean().shift(1)
        sd = x.expanding(min_periods=50).std(ddof=0).shift(1)
        z = (x - mu) / sd
        out[f"{c}_z"] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def build_state_windows(df: pd.DataFrame, state_cols: list[str], window: int) -> pd.DataFrame:
    """
    Build flattened state windows as list[float] per row.
    Row i contains data from [i-window+1 .. i] (inclusive).
    """
    if window < 2:
        raise ValueError("window must be >= 2")

    d = df.copy().reset_index(drop=True)
    mat = d[state_cols].astype(float).to_numpy()

    states: list[list[float]] = []
    timestamps: list[int] = []
    closes: list[float] = []
    next_lr: list[float] = []

    close = d["close"].astype(float).to_numpy()
    lr = np.log(close[1:] / close[:-1])
    lr = np.append(lr, np.nan)

    for i in range(window - 1, len(d)):
        w = mat[i - window + 1 : i + 1]
        states.append(w.reshape(-1).tolist())
        timestamps.append(int(d.loc[i, "timestamp_ms"]))
        closes.append(float(d.loc[i, "close"]))
        next_lr.append(float(lr[i]) if np.isfinite(lr[i]) else 0.0)

    out = pd.DataFrame(
        {
            "timestamp_ms": timestamps,
            "datetime_utc": pd.to_datetime(timestamps, unit="ms", utc=True),
            "close": closes,
            "next_log_return": next_lr,
            "state": states,
        }
    )
    return out
