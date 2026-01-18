from __future__ import annotations

import json
import os
import pytest
from pathlib import Path

from autonomous_rl_trading_bot.backtest.runner import run_backtest


def test_baseline_backtest_sma():
    """Test that baseline backtest (SMA) generates correct artifacts."""
    # Check if dataset exists
    dataset_path = "data/processed/BTCUSDT_1m_dataset.parquet"
    if not os.path.exists(dataset_path):
        pytest.skip(f"Dataset not found: {dataset_path}. Run scripts/make_dataset.py first.")
    
    # Run SMA baseline backtest
    result = run_backtest(
        mode="spot",
        policy="sma",
        symbol="BTCUSDT",
        interval="1m",
        train_split=0.8,
        output_dir="reports",
    )
    
    # Verify output files exist
    report_json = Path(result["report_json"])
    trades_csv = Path(result["trades_csv"])
    equity_csv = Path(result["equity_csv"])
    
    assert report_json.exists(), f"Report JSON not created: {report_json}"
    assert trades_csv.exists(), f"Trades CSV not created: {trades_csv}"
    assert equity_csv.exists(), f"Equity CSV not created: {equity_csv}"
    
    # Verify report JSON has required keys
    with open(report_json, "r") as f:
        metrics = json.load(f)
    
    required_keys = [
        "total_return",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "num_trades",
        "win_rate",
        "avg_trade_pnl",
        "profit_factor",
        "final_equity",
        "initial_equity",
        "policy",
    ]
    
    for key in required_keys:
        assert key in metrics, f"Missing key in backtest_report.json: {key}"
    
    assert metrics["policy"] == "sma", f"Policy should be 'sma', got: {metrics['policy']}"
    
    # Verify trades CSV has rows (if any trades occurred)
    import pandas as pd
    trades_df = pd.read_csv(trades_csv)
    assert len(trades_df.columns) > 0, "Trades CSV should have columns"
    
    # Verify equity CSV has rows
    equity_df = pd.read_csv(equity_csv)
    assert len(equity_df) > 0, "Equity CSV should have rows"
    assert "drawdown" in equity_df.columns, "Equity CSV should have drawdown column"


def test_baseline_backtest_buyhold():
    """Test that baseline backtest (Buy & Hold) works."""
    dataset_path = "data/processed/BTCUSDT_1m_dataset.parquet"
    if not os.path.exists(dataset_path):
        pytest.skip(f"Dataset not found: {dataset_path}. Run scripts/make_dataset.py first.")
    
    result = run_backtest(
        mode="spot",
        policy="buyhold",
        symbol="BTCUSDT",
        interval="1m",
        train_split=0.8,
        output_dir="reports",
    )
    
    assert Path(result["report_json"]).exists()
    assert Path(result["trades_csv"]).exists()
    assert Path(result["equity_csv"]).exists()
    
    with open(result["report_json"], "r") as f:
        metrics = json.load(f)
    
    assert metrics["policy"] == "buyhold"
    assert "total_return" in metrics
    assert "num_trades" in metrics
