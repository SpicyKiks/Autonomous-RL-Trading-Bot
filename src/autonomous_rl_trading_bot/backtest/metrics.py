from __future__ import annotations

import math
from typing import Any

import numpy as np


def compute_backtest_metrics(
    equity_curve: list[float],
    trades: list[dict[str, Any]],
    initial_equity: float,
) -> dict[str, float]:
    """
    Compute comprehensive backtest metrics from equity curve and trades.
    
    Args:
        equity_curve: List of equity values over time
        trades: List of trade dictionaries with 'pnl' field
        initial_equity: Starting equity value
    
    Returns:
        Dictionary of computed metrics
    """
    equity_array = np.array(equity_curve, dtype=float)
    
    # Total return
    final_equity = equity_array[-1]
    total_return = (final_equity / initial_equity) - 1.0 if initial_equity > 0 else 0.0
    
    # Log returns for Sharpe/Sortino
    log_returns = np.diff(np.log(np.maximum(equity_array, 1e-12)))
    
    # Sharpe ratio (annualized, assuming 1m data: 252 * 24 * 60 periods per year)
    sharpe = 0.0
    if len(log_returns) > 1 and np.std(log_returns) > 1e-12:
        sharpe = np.mean(log_returns) / np.std(log_returns) * math.sqrt(252 * 24 * 60)
    
    # Sortino ratio (downside deviation only)
    sortino = 0.0
    if len(log_returns) > 1:
        downside_returns = log_returns[log_returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 1e-12:
            sortino = np.mean(log_returns) / np.std(downside_returns) * math.sqrt(252 * 24 * 60)
    
    # Max drawdown
    peak = np.maximum.accumulate(equity_array)
    drawdown = (peak - equity_array) / peak
    max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
    
    # Calmar ratio (return / max_drawdown)
    calmar = total_return / max_drawdown if max_drawdown > 1e-12 else 0.0
    
    # Trade statistics
    num_trades = len(trades)
    win_count = sum(1 for t in trades if t.get("pnl", 0.0) > 0)
    loss_count = sum(1 for t in trades if t.get("pnl", 0.0) < 0)
    win_rate = win_count / num_trades if num_trades > 0 else 0.0
    
    # Average trade PnL (use net_pnl if available, fallback to pnl)
    total_pnl = sum(t.get("net_pnl", t.get("pnl", 0.0)) for t in trades)
    avg_trade_pnl = total_pnl / num_trades if num_trades > 0 else 0.0
    
    # Average trade net PnL (explicitly use net_pnl)
    avg_trade_net_pnl = 0.0
    if num_trades > 0:
        total_net_pnl = sum(t.get("net_pnl", t.get("pnl", 0.0)) for t in trades)
        avg_trade_net_pnl = total_net_pnl / num_trades
    
    # Profit factor (gross profit / gross loss) - use net_pnl
    net_pnl_values = [t.get("net_pnl", t.get("pnl", 0.0)) for t in trades]
    gross_profit = sum(pnl for pnl in net_pnl_values if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in net_pnl_values if pnl < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else 0.0
    
    # Calculate total fees and slippage from trades
    total_fees = sum(t.get("fee_paid", 0.0) for t in trades)
    total_slippage = sum(t.get("slippage_cost", 0.0) for t in trades)
    
    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_drawdown),
        "calmar": float(calmar),
        "num_trades": int(num_trades),
        "win_rate": float(win_rate),
        "avg_trade_pnl": float(avg_trade_pnl),
        "avg_trade_net_pnl": float(avg_trade_net_pnl),
        "profit_factor": float(profit_factor),
        "final_equity": float(final_equity),
        "initial_equity": float(initial_equity),
        "total_fees": float(total_fees),
        "total_slippage": float(total_slippage),
    }
