from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from autonomous_rl_trading_bot.envs.trading_env import TradingEnv


@pytest.fixture
def sample_dataset():
    """Create a small sample dataset for testing."""
    n = 100
    state_len = 270  # 30 window * 9 features
    
    data = {
        "timestamp_ms": [1000000 + i * 60000 for i in range(n)],
        "close": [100.0 + i * 0.1 for i in range(n)],
        "next_log_return": [0.001 * (1 if i % 2 == 0 else -1) for i in range(n)],
        "state": [[0.0] * state_len for _ in range(n)],
    }
    
    df = pd.DataFrame(data)
    df["datetime_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    return df


def test_trading_env_reset(sample_dataset):
    """Test environment reset returns correct observation shape."""
    env = TradingEnv(sample_dataset, seed=42)
    obs, info = env.reset()
    
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert len(obs.shape) == 1
    # obs = [state_vector (270) + 4 account features] = 274
    assert obs.shape[0] == 270 + 4
    
    assert isinstance(info, dict)
    assert "current_idx" in info
    assert "equity" in info
    assert info["equity"] == 10000.0  # default initial_balance


def test_trading_env_step(sample_dataset):
    """Test environment step returns finite reward and no NaNs."""
    env = TradingEnv(sample_dataset, seed=42)
    obs, _ = env.reset()
    
    # Execute a few steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(reward, (float, np.floating))
        assert np.isfinite(reward), f"Reward not finite: {reward}"
        assert not np.isnan(reward), f"Reward is NaN: {reward}"
        
        assert isinstance(obs, np.ndarray)
        assert np.isfinite(obs).all(), f"Observation contains NaN/inf: {obs}"
        
        assert isinstance(info, dict)
        assert "equity" in info
        assert np.isfinite(info["equity"]), f"Equity not finite: {info['equity']}"
        
        if terminated:
            break


def test_trading_env_episode_terminates(sample_dataset):
    """Test episode terminates at end of dataset."""
    env = TradingEnv(sample_dataset, seed=42)
    obs, _ = env.reset()
    
    step_count = 0
    done = False
    
    while not done and step_count < len(sample_dataset) + 10:
        action = 0  # HOLD
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
    
    assert done, "Episode should terminate"
    assert step_count <= len(sample_dataset), "Should terminate by end of dataset"


def test_trading_env_actions_dont_crash(sample_dataset):
    """Test all actions execute without crashing."""
    env = TradingEnv(sample_dataset, seed=42)
    obs, _ = env.reset()
    
    # Test all actions
    for action in [0, 1, 2, 3]:
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(reward)
            assert np.isfinite(obs).all()
        except Exception as e:
            pytest.fail(f"Action {action} crashed: {e}")


def test_trading_env_with_real_dataset():
    """Test with real dataset if available."""
    dataset_path = os.path.join("data", "processed", "BTCUSDT_1m_dataset.parquet")
    
    if not os.path.exists(dataset_path):
        pytest.skip(f"Real dataset not found: {dataset_path}")
    
    df = pd.read_parquet(dataset_path)
    
    # Use small slice for testing
    if len(df) > 100:
        df = df.iloc[:100].copy()
    
    env = TradingEnv(df, seed=42)
    obs, info = env.reset()
    
    assert obs.shape[0] > 0
    assert np.isfinite(obs).all()
    
    # Run a few steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert np.isfinite(reward)
        assert np.isfinite(obs).all()
        
        if terminated:
            break
