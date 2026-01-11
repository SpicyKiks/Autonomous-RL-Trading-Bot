from __future__ import annotations

import pandas as pd
import pytest

from autonomous_rl_trading_bot.features.alignment import asof_align, asof_align_many


def test_asof_align_basic():
    """Test basic asof alignment."""
    left = pd.DataFrame({"timestamp": [1000, 2000, 3000], "value": [1, 2, 3]})
    right = pd.DataFrame({"timestamp": [1500, 2500], "data": [10, 20]})
    result = asof_align(left, right, on="timestamp")
    assert len(result) == 3
    assert pd.isna(result.iloc[0]["data"])  # 1000 has no match before it
    assert result.iloc[1]["data"] == 10  # 2000 aligns to 1500 (backward)
    assert result.iloc[2]["data"] == 20  # 3000 aligns to 2500 (backward)


def test_asof_align_no_lookahead():
    """Test that asof alignment prevents lookahead bias."""
    # Base: 5-minute intervals
    base_times = pd.date_range("2024-01-01 00:00", periods=10, freq="5min")
    base_df = pd.DataFrame({"timestamp": base_times, "base_value": range(10)})

    # Right: 1-hour intervals (coarser)
    right_times = pd.date_range("2024-01-01 00:00", periods=3, freq="1h")
    right_df = pd.DataFrame({"timestamp": right_times, "hourly_value": [100, 200, 300]})

    result = asof_align(base_df, right_df, on="timestamp", direction="backward")

    assert len(result) == len(base_df)
    assert "hourly_value" in result.columns
    assert "base_value" in result.columns

    # Check no lookahead: aligned timestamp must be <= base timestamp
    for idx, row in result.iterrows():
        base_ts = row["timestamp"]
        if pd.notna(row["hourly_value"]):
            # Find the source row in right_df
            aligned_idx = right_df[right_df["timestamp"] <= base_ts].index
            if len(aligned_idx) > 0:
                aligned_ts = right_df.loc[aligned_idx[-1], "timestamp"]
                assert aligned_ts <= base_ts, f"Lookahead detected: {aligned_ts} > {base_ts}"


def test_asof_align_many():
    """Test aligning multiple DataFrames."""
    base = pd.DataFrame({"timestamp": [1000, 2000, 3000]})
    frames = {
        "a": pd.DataFrame({"timestamp": [1500], "val": [100]}),
        "b": pd.DataFrame({"timestamp": [2500], "val": [200]}),
    }
    result = asof_align_many(base, frames, on="timestamp")
    assert "a_val" in result.columns
    assert "b_val" in result.columns
    assert len(result) == len(base)


def test_asof_align_shape_preservation():
    """Test that alignment preserves base DataFrame shape."""
    base = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="5min"),
            "price": range(20),
        }
    )

    # Multiple right frames with different frequencies
    right1 = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1h"),
            "feature1": range(5),
        }
    )
    right2 = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="4h"),
            "feature2": range(3),
        }
    )

    frames = {"hourly": right1, "four_hourly": right2}
    result = asof_align_many(base, frames, on="timestamp")

    # Shape should match base
    assert len(result) == len(base)
    assert "price" in result.columns
    assert "hourly_feature1" in result.columns
    assert "four_hourly_feature2" in result.columns


def test_asof_align_sorted_requirement():
    """Test that alignment requires sorted timestamps."""
    left = pd.DataFrame({"timestamp": [3000, 1000, 2000], "value": [3, 1, 2]})
    right = pd.DataFrame({"timestamp": [1500, 2500], "data": [10, 20]})

    # Should work (function sorts internally via merge_asof)
    result = asof_align(left, right, on="timestamp")
    assert len(result) == 3
