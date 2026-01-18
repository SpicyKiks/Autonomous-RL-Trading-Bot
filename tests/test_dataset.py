from __future__ import annotations

import os
import math
import pandas as pd
import numpy as np

from autonomous_rl_trading_bot.data.feature_engineering import compute_features, causal_zscore


def test_dataset_integrity_and_no_nan_inf():
    symbol = "BTCUSDT"
    interval = "1m"
    stem = f"{symbol}_{interval}"

    dataset_path = os.path.join("data", "processed", f"{stem}_dataset.parquet")
    raw_path = os.path.join("data", "raw", f"{stem}.parquet")
    raw_csv = os.path.join("data", "raw", f"{stem}.csv")

    assert os.path.exists(dataset_path), f"Missing dataset parquet: {dataset_path}"

    ds = pd.read_parquet(dataset_path)
    assert len(ds) > 1000, "Dataset too small; download more history."

    for c in ["timestamp_ms", "datetime_utc", "close", "next_log_return", "state"]:
        assert c in ds.columns, f"Missing column: {c}"

    for c in ["close", "next_log_return"]:
        arr = ds[c].to_numpy(dtype=float)
        assert np.isfinite(arr).all(), f"Found NaN/inf in {c}"

    first = ds["state"].iloc[0]
    # Parquet may store as numpy array, convert to list for consistency
    if isinstance(first, np.ndarray):
        first = first.tolist()
    assert isinstance(first, (list, tuple, np.ndarray)), "state must be list-like"
    state_len = len(first)
    assert state_len > 0, "state_len must be > 0"

    for i in [0, len(ds) // 2, len(ds) - 1]:
        s = ds["state"].iloc[i]
        if isinstance(s, np.ndarray):
            s = s.tolist()
        s_arr = np.array(s, dtype=float)
        assert len(s_arr) == state_len, "Inconsistent state vector length"
        assert np.isfinite(s_arr).all(), "Found NaN/inf inside state vector"

    if os.path.exists(raw_path):
        raw = pd.read_parquet(raw_path)
    else:
        assert os.path.exists(raw_csv), "Need raw parquet or csv for leakage verification"
        raw = pd.read_csv(raw_csv)

    feats = compute_features(raw)

    required = [
        "log_return",
        "atr_14",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "volatility_30",
        "volume_delta",
        "volume",
    ]
    feats = feats.dropna(subset=required).reset_index(drop=True)
    feats = causal_zscore(feats, required)

    ts = int(ds["timestamp_ms"].iloc[0])
    row = feats.loc[feats["timestamp_ms"] == ts]
    assert len(row) == 1, "Timestamp alignment failed between features and dataset"
    row = row.iloc[0]

    state_cols = [f"{c}_z" for c in required]
    window = state_len // len(state_cols)
    assert window * len(state_cols) == state_len, "state_len not divisible by feature count"

    state_vec_raw = ds["state"].iloc[0]
    if isinstance(state_vec_raw, np.ndarray):
        state_vec = state_vec_raw.astype(float)
    else:
        state_vec = np.array(state_vec_raw, dtype=float)
    last_block = state_vec[-len(state_cols):]

    idx = state_cols.index("log_return_z")
    assert math.isfinite(row["log_return_z"])
    assert abs(last_block[idx] - float(row["log_return_z"])) < 1e-6, "Possible leakage/misalignment in window construction"
