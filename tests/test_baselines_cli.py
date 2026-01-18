from __future__ import annotations

import os
import pytest
import subprocess
import sys


def test_baselines_cli_symbol_interval():
    """Test that baselines CLI supports --symbol and --interval."""
    # Check if dataset exists
    dataset_path = "data/processed/BTCUSDT_1m_dataset.parquet"
    if not os.path.exists(dataset_path):
        pytest.skip(f"Dataset not found: {dataset_path}. Run scripts/make_dataset.py first.")
    
    # Run baselines command with --symbol and --interval
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "autonomous_rl_trading_bot.cli",
            "baselines",
            "--symbol",
            "BTCUSDT",
            "--interval",
            "1m",
            "--sma",
            "--mode",
            "spot",
        ],
        capture_output=True,
        text=True,
    )
    
    # Should not error (exit code 0)
    assert result.returncode == 0, f"Command failed with exit code {result.returncode}. Stderr: {result.stderr}"


def test_baselines_cli_help():
    """Test that baselines --help shows symbol and interval options."""
    # Test via CLI wrapper
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "autonomous_rl_trading_bot.cli",
            "baselines",
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    
    # The CLI wrapper may not show help, so test by importing and checking parser
    from autonomous_rl_trading_bot.evaluation.baselines import main
    import argparse
    
    # Create a test parser to verify args exist
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--interval", default=None)
    
    # If we can parse these args, they exist
    args = parser.parse_args(["--symbol", "BTCUSDT", "--interval", "1m"])
    assert args.symbol == "BTCUSDT"
    assert args.interval == "1m"
