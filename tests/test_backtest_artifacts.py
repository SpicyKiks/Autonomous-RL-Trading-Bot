from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from autonomous_rl_trading_bot.backtest.runner import run_backtest


def test_backtest_artifacts_columns():
    """Test that backtest generates correct artifact columns."""
    # Check if model and dataset exist
    model_path = "models/ppo_trader.zip"
    dataset_path = "data/processed/BTCUSDT_1m_dataset.parquet"
    
    if not os.path.exists(model_path):
        pytest.skip(f"Model not found: {model_path}. Run 'arbt train --smoke-test' first.")
    if not os.path.exists(dataset_path):
        pytest.skip(f"Dataset not found: {dataset_path}. Run scripts/make_dataset.py first.")
    
    # Run backtest
    result = run_backtest(
        mode="spot",
        model_path=model_path,
        symbol="BTCUSDT",
        interval="1m",
        train_split=0.8,
        output_dir="reports",
    )
    
    # Verify trades.csv columns
    trades_csv = Path(result["trades_csv"])
    assert trades_csv.exists(), f"Trades CSV not created: {trades_csv}"
    
    trades_df = pd.read_csv(trades_csv)
    required_trade_cols = [
        "entry_timestamp_ms",
        "exit_timestamp_ms",
        "entry_price",
        "exit_price",
        "position",
        "qty",
        "gross_pnl",
        "fee_paid",
        "slippage_cost",
        "net_pnl",
        "pnl",  # Legacy alias
    ]
    
    for col in required_trade_cols:
        assert col in trades_df.columns, f"Missing column in trades.csv: {col}"
    
    # Verify equity_curve.csv columns
    equity_csv = Path(result["equity_csv"])
    assert equity_csv.exists(), f"Equity CSV not created: {equity_csv}"
    
    equity_df = pd.read_csv(equity_csv)
    required_equity_cols = ["timestamp_ms", "equity", "drawdown"]
    
    for col in required_equity_cols:
        assert col in equity_df.columns, f"Missing column in equity_curve.csv: {col}"
    
    # Verify backtest_report.json keys
    report_json = Path(result["report_json"])
    assert report_json.exists(), f"Report JSON not created: {report_json}"
    
    with open(report_json) as f:
        metrics = json.load(f)
    
    required_report_keys = [
        "total_fees",
        "total_slippage",
        "avg_trade_net_pnl",
    ]
    
    for key in required_report_keys:
        assert key in metrics, f"Missing key in backtest_report.json: {key}"
        assert isinstance(metrics[key], (int, float)), f"Key {key} is not numeric"
