from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from autonomous_rl_trading_bot.backtest.runner import run_backtest


def test_backtest_smoke():
    """Smoke test for backtest command."""
    # Check if model exists (from smoke test training)
    model_path = "models/ppo_trader.zip"
    if not os.path.exists(model_path):
        pytest.skip(f"Model not found: {model_path}. Run 'arbt train --smoke-test' first.")
    
    # Check if dataset exists
    dataset_path = "data/processed/BTCUSDT_1m_dataset.parquet"
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
    
    # Verify outputs exist
    report_json = Path(result["report_json"])
    trades_csv = Path(result["trades_csv"])
    equity_csv = Path(result["equity_csv"])
    
    assert report_json.exists(), f"Report JSON not created: {report_json}"
    assert trades_csv.exists(), f"Trades CSV not created: {trades_csv}"
    assert equity_csv.exists(), f"Equity CSV not created: {equity_csv}"
    
    # Verify JSON contains required keys
    with open(report_json) as f:
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
    ]
    
    for key in required_keys:
        assert key in metrics, f"Missing metric key: {key}"
        assert isinstance(metrics[key], (int, float)), f"Metric {key} is not numeric"
    
    # Verify CSV files have content
    assert trades_csv.stat().st_size > 0, "Trades CSV is empty"
    assert equity_csv.stat().st_size > 0, "Equity CSV is empty"
    
    # Verify result structure
    assert "metrics" in result
    assert "report_json" in result
    assert "trades_csv" in result
    assert "equity_csv" in result
    assert "num_trades" in result
