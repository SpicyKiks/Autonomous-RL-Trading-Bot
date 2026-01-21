from __future__ import annotations

import os

import numpy as np
import pytest

from autonomous_rl_trading_bot.training.train_pipeline import (
    evaluate_ppo,
    load_dataset,
    split_dataset,
    train_ppo,
)


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create temporary directory for model output."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    return str(model_dir)


@pytest.fixture
def temp_report_dir(tmp_path):
    """Create temporary directory for report output."""
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True)
    return str(report_dir)


@pytest.fixture
def temp_tensorboard_dir(tmp_path):
    """Create temporary directory for TensorBoard logs."""
    tb_dir = tmp_path / "tensorboard"
    tb_dir.mkdir(parents=True)
    return str(tb_dir)


def test_train_smoke(temp_model_dir, temp_report_dir, temp_tensorboard_dir):
    """Test smoke test training creates model file."""
    dataset_path = os.path.join("data", "processed", "BTCUSDT_1m_dataset.parquet")
    
    if not os.path.exists(dataset_path):
        pytest.skip(f"Dataset not found: {dataset_path}. Run scripts/make_dataset.py first.")
    
    # Load dataset
    df = load_dataset("BTCUSDT", "1m")
    
    # Use small slice for smoke test
    if len(df) > 500:
        df = df.iloc[:500].copy()
    
    # Split
    train_df, test_df = split_dataset(df, train_split=0.8)
    
    # Train with minimal timesteps
    model_path = os.path.join(temp_model_dir, "ppo_trader.zip")
    
    train_ppo(
        train_df,
        timesteps=2000,
        seed=42,
        tensorboard_log_dir=temp_tensorboard_dir,
        model_out=model_path,
        taker_fee=0.0004,
        slippage_bps=1.0,
        risk_penalty=0.1,
        position_fraction=1.0,
    )
    
    # Verify model file exists
    assert os.path.exists(model_path), f"Model file not created: {model_path}"
    assert os.path.getsize(model_path) > 0, "Model file is empty"
    
    # Verify config file exists
    config_path = os.path.join(temp_model_dir, "ppo_trader_config.json")
    assert os.path.exists(config_path), f"Config file not created: {config_path}"
    
    # Evaluate
    report_path = os.path.join(temp_report_dir, "training_metrics.json")
    metrics = evaluate_ppo(
        model_path,
        test_df,
        seed=42,
        taker_fee=0.0004,
        slippage_bps=1.0,
        risk_penalty=0.1,
        position_fraction=1.0,
        report_out=report_path,
    )
    
    # Verify report exists
    assert os.path.exists(report_path), f"Report file not created: {report_path}"
    
    # Verify metrics structure
    assert "total_return" in metrics
    assert "sharpe" in metrics
    assert "max_drawdown" in metrics
    assert "trades" in metrics
    assert "win_rate" in metrics
    assert "avg_trade_pnl" in metrics
    
    # Verify metrics are finite
    assert np.isfinite(metrics["total_return"])
    assert np.isfinite(metrics["sharpe"])
    assert np.isfinite(metrics["max_drawdown"])
