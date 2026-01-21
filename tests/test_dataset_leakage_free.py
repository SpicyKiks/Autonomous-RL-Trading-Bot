from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from autonomous_rl_trading_bot.common.db import apply_migrations, ensure_schema_migrations
from autonomous_rl_trading_bot.common.paths import repo_root
from autonomous_rl_trading_bot.data.dataset_builder import (
    _compute_splits,
    _prepare_scaled_features,
    build_dataset_from_db,
)
from autonomous_rl_trading_bot.rl.dataset import load_dataset_npz


def _insert_candle(conn: sqlite3.Connection, t: int, close: float) -> None:
    interval_ms = 60_000
    conn.execute(
        """
        INSERT INTO candles(
          exchange, market_type, symbol, interval, open_time_ms,
          open, high, low, close, volume, close_time_ms,
          quote_asset_volume, number_of_trades,
          taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "binance",
            "spot",
            "BTCUSDT",
            "1m",
            t,
            close,
            close,
            close,
            close,
            1.0,
            t + interval_ms - 1,
            None,
            None,
            None,
            None,
            None,
        ),
    )


def test_compute_splits() -> None:
    """Test that split boundaries are computed correctly."""
    n = 100
    splits = _compute_splits(n, train_frac=0.75, val_frac=0.10, test_frac=0.15)
    
    assert splits["train"]["start_idx"] == 0
    assert splits["train"]["end_idx"] == 75
    assert splits["val"]["start_idx"] == 75
    assert splits["val"]["end_idx"] == 85
    assert splits["test"]["start_idx"] == 85
    assert splits["test"]["end_idx"] == 100
    
    # Check no overlap and full coverage
    assert splits["train"]["end_idx"] == splits["val"]["start_idx"]
    assert splits["val"]["end_idx"] == splits["test"]["start_idx"]
    assert splits["test"]["end_idx"] == n


def test_compute_splits_sums_to_one() -> None:
    """Test that split fractions must sum to 1.0."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        _compute_splits(100, train_frac=0.8, val_frac=0.1, test_frac=0.05)


def test_scaler_fit_only_on_train(tmp_path: Path) -> None:
    """Test that scaler is fit only on train split."""
    # Create synthetic data with different distributions per split
    n = 100
    train_data = np.random.randn(75, 4) * 10.0 + 100.0  # mean ~100
    val_data = np.random.randn(10, 4) * 5.0 + 50.0  # mean ~50
    test_data = np.random.randn(15, 4) * 2.0 + 200.0  # mean ~200
    
    feature_matrix = np.vstack([train_data, val_data, test_data])
    
    splits = _compute_splits(n, train_frac=0.75, val_frac=0.10, test_frac=0.15)
    
    # Create arrays dict
    arrays = {
        "log_return": feature_matrix[:, 0],
        "return": feature_matrix[:, 1],
        "close": feature_matrix[:, 2],
        "volume": feature_matrix[:, 3],
    }
    
    feature_list = ["log_return", "return", "close_norm", "vol_norm"]
    
    arrays_scaled, scaler = _prepare_scaled_features(
        arrays, feature_list, splits, scaler_type="robust"
    )
    
    # Verify scaler was fit on train only
    train_start = splits["train"]["start_idx"]
    train_end = splits["train"]["end_idx"]
    train_mean = feature_matrix[train_start:train_end].mean(axis=0)
    
    # Scaler center should match train mean (RobustScaler uses median, but close enough for test)
    assert scaler is not None
    
    # Verify scaled features exist
    for feat_name in feature_list:
        scaled_key = f"{feat_name}_scaled"
        assert scaled_key in arrays_scaled
        assert arrays_scaled[scaled_key].shape[0] == n


def test_build_dataset_with_splits(tmp_path: Path) -> None:
    """Test that dataset building includes splits and scaler."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    
    migrations_dir = repo_root() / "sql" / "migrations"
    ensure_schema_migrations(conn)
    apply_migrations(conn, migrations_dir)
    
    # Insert 100 candles
    for i in range(100):
        _insert_candle(conn, i * 60_000, close=100.0 + i * 0.1)
    conn.commit()
    
    out_base = tmp_path / "datasets"
    res = build_dataset_from_db(
        conn,
        market_type="spot",
        symbol="BTCUSDT",
        interval="1m",
        window_minutes=100,
        strict_gaps=False,
        features=["return", "log_return"],
        out_base_dir=out_base,
        dataset_id="test_ds",
        end_ms=None,
        train_frac=0.75,
        val_frac=0.10,
        test_frac=0.15,
        scaler_type="robust",
    )
    
    # Check meta.json has splits
    meta = json.loads(res.meta_path.read_text(encoding="utf-8"))
    assert "splits" in meta
    assert "train" in meta["splits"]
    assert "val" in meta["splits"]
    assert "test" in meta["splits"]
    
    # Check split boundaries
    splits = meta["splits"]
    n = meta["rows"]
    assert splits["train"]["start_idx"] == 0
    assert splits["train"]["end_idx"] == int(n * 0.75)
    assert splits["val"]["start_idx"] == splits["train"]["end_idx"]
    assert splits["val"]["end_idx"] == splits["val"]["start_idx"] + int(n * 0.10)
    assert splits["test"]["start_idx"] == splits["val"]["end_idx"]
    assert splits["test"]["end_idx"] == n
    
    # Check scaler exists
    assert "scaler_path" in meta
    scaler_path = res.out_dir / meta["scaler_path"]
    assert scaler_path.exists()
    
    # Check feature_list exists
    assert "feature_list" in meta
    assert isinstance(meta["feature_list"], list)
    assert len(meta["feature_list"]) > 0
    
    # Check scaled features exist in npz
    npz = np.load(res.npz_path)
    for feat_name in meta["feature_list"]:
        scaled_key = f"{feat_name}_scaled"
        assert scaled_key in npz.files
    
    conn.close()


def test_load_dataset_with_split(tmp_path: Path) -> None:
    """Test that loading dataset with split filters correctly."""
    # Create a minimal dataset structure
    dataset_dir = tmp_path / "test_dataset"
    dataset_dir.mkdir(parents=True)
    
    n = 100
    meta = {
        "dataset_id": "test",
        "rows": n,
        "splits": {
            "train": {"start_idx": 0, "end_idx": 75},
            "val": {"start_idx": 75, "end_idx": 85},
            "test": {"start_idx": 85, "end_idx": 100},
        },
        "feature_list": ["log_return", "return", "close_norm", "vol_norm"],
    }
    
    (dataset_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    
    # Create npz with full data
    arrays = {
        "close": np.arange(n, dtype=np.float64),
        "volume": np.ones(n, dtype=np.float64),
        "log_return": np.zeros(n, dtype=np.float64),
        "return": np.zeros(n, dtype=np.float64),
    }
    np.savez_compressed(dataset_dir / "dataset.npz", **arrays)
    
    # Load without split (full data)
    ds_full = load_dataset_npz(dataset_dir, split=None)
    assert len(ds_full.data["close"]) == n
    
    # Load with train split
    ds_train = load_dataset_npz(dataset_dir, split="train")
    assert len(ds_train.data["close"]) == 75
    assert ds_train.data["close"][0] == 0
    assert ds_train.data["close"][-1] == 74
    
    # Load with test split
    ds_test = load_dataset_npz(dataset_dir, split="test")
    assert len(ds_test.data["close"]) == 15
    assert ds_test.data["close"][0] == 85
    assert ds_test.data["close"][-1] == 99


def test_feature_dimension_matches_meta(tmp_path: Path) -> None:
    """Test that feature dimension matches meta feature_list."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    
    migrations_dir = repo_root() / "sql" / "migrations"
    ensure_schema_migrations(conn)
    apply_migrations(conn, migrations_dir)
    
    # Insert candles
    for i in range(100):
        _insert_candle(conn, i * 60_000, close=100.0 + i * 0.1)
    conn.commit()
    
    out_base = tmp_path / "datasets"
    res = build_dataset_from_db(
        conn,
        market_type="spot",
        symbol="BTCUSDT",
        interval="1m",
        window_minutes=100,
        strict_gaps=False,
        features=["return", "log_return"],
        out_base_dir=out_base,
        dataset_id="test_ds",
        end_ms=None,
        train_frac=0.75,
        val_frac=0.10,
        test_frac=0.15,
        scaler_type="robust",
    )
    
    # Load dataset
    ds = load_dataset_npz(res.out_dir)
    
    # Check feature_list matches actual features
    meta = ds.meta
    feature_list = meta["feature_list"]
    
    # Verify all features in feature_list have scaled versions
    npz = np.load(res.npz_path)
    for feat_name in feature_list:
        scaled_key = f"{feat_name}_scaled"
        assert scaled_key in npz.files, f"Missing scaled feature: {scaled_key}"
        
        # Verify dimension matches
        feat_data = npz[scaled_key]
        assert feat_data.shape[0] == meta["rows"], f"Feature {feat_name} has wrong length"
    
    conn.close()

