from __future__ import annotations

import json
import os
import pytest
from pathlib import Path

from autonomous_rl_trading_bot.repro.runner import run_repro


def test_repro_smoke():
    """Smoke test for reproducibility pipeline."""
    # Skip if network access not allowed (for CI)
    if os.getenv("ALLOW_NETWORK") != "1":
        pytest.skip("Network access required (set ALLOW_NETWORK=1)")
    
    # Run minimal repro
    result = run_repro(
        symbol="BTCUSDT",
        interval="1m",
        days=2,  # Minimal data
        window=30,
        timesteps=2000,  # Minimal training
        mode="spot",
        seed=42,
        output_dir="reports/repro_test",
    )
    
    run_dir = Path(result["run_dir"])
    
    # Verify required files exist
    required_files = [
        "run_config.json",
        "training_metrics.json",
        "ppo/backtest_report.json",
        "sma/backtest_report.json",
        "comparison.json",
        "comparison.md",
    ]
    
    for file in required_files:
        file_path = run_dir / file
        assert file_path.exists(), f"Required file not found: {file_path}"
    
    # Verify run_config.json structure
    with open(run_dir / "run_config.json", "r") as f:
        config = json.load(f)
    
    assert "run_id" in config
    assert config["symbol"] == "BTCUSDT"
    assert config["interval"] == "1m"
    assert config["seed"] == 42
    
    # Verify comparison.json structure
    with open(run_dir / "comparison.json", "r") as f:
        comparison = json.load(f)
    
    assert "ppo" in comparison
    assert "sma" in comparison
    assert "winner" in comparison
    assert comparison["winner"] in ["ppo", "sma"]
    
    # Verify PPO backtest report
    with open(run_dir / "ppo" / "backtest_report.json", "r") as f:
        ppo_report = json.load(f)
    
    assert "total_return" in ppo_report
    assert "sharpe" in ppo_report
    assert ppo_report.get("policy") == "ppo"
    
    # Verify SMA backtest report
    with open(run_dir / "sma" / "backtest_report.json", "r") as f:
        sma_report = json.load(f)
    
    assert "total_return" in sma_report
    assert "sharpe" in sma_report
    assert sma_report.get("policy") == "sma"
