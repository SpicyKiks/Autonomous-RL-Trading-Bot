from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExpandedMetrics:
    """Expanded metrics for baseline comparison."""
    sharpe: float | None
    sortino: float | None
    calmar: float | None
    max_drawdown: float
    win_rate: float | None
    profit_factor: float | None
    avg_trade: float | None
    exposure_avg: float
    turnover: float
    total_return: float
    trade_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade": self.avg_trade,
            "exposure_avg": self.exposure_avg,
            "turnover": self.turnover,
            "total_return": self.total_return,
            "trade_count": self.trade_count,
        }


def _safe_div(a: float, b: float) -> float:
    if b == 0.0:
        return 0.0
    return a / b


def _annualization_factor(interval_ms: int) -> float:
    """Approximate sqrt(steps per year) for Sharpe/Sortino."""
    if interval_ms <= 0:
        return 0.0
    seconds_per_step = interval_ms / 1000.0
    steps_per_year = (365.0 * 24.0 * 60.0 * 60.0) / seconds_per_step
    return math.sqrt(steps_per_year)


def compute_expanded_metrics(
    *,
    equity: list[float],
    drawdown: list[float],
    exposure: list[float],
    trades: list[dict[str, Any]],
    interval_ms: int,
) -> ExpandedMetrics:
    """
    Compute expanded metrics including Sharpe, Sortino, Calmar, win rate, profit factor, etc.
    
    Args:
        equity: Equity curve over time
        drawdown: Drawdown curve over time
        exposure: Exposure curve over time (position value / equity)
        trades: List of trade dictionaries with 'equity_after' or 'realized_pnl'
        interval_ms: Interval in milliseconds for annualization
    
    Returns:
        ExpandedMetrics object
    """
    if not equity:
        raise ValueError("equity curve is empty")
    
    initial_cash = float(equity[0])
    final_equity = float(equity[-1])
    total_return = _safe_div(final_equity, initial_cash) - 1.0
    max_dd = float(max(drawdown)) if drawdown else 0.0
    
    # Compute Sharpe and Sortino from equity log-returns
    sharpe: float | None = None
    sortino: float | None = None
    if len(equity) >= 3:
        lr: list[float] = []
        for i in range(1, len(equity)):
            e0 = float(equity[i - 1])
            e1 = float(equity[i])
            if e0 > 0.0 and e1 > 0.0:
                lr.append(math.log(e1 / e0))
            else:
                lr.append(0.0)
        
        if lr:
            mean_lr = sum(lr) / len(lr)
            var_lr = sum((x - mean_lr) ** 2 for x in lr) / (len(lr) - 1) if len(lr) > 1 else 0.0
            std_lr = math.sqrt(max(0.0, var_lr))
            
            # Downside deviation for Sortino (only negative returns)
            downside_returns = [x for x in lr if x < 0.0]
            if downside_returns:
                mean_downside = sum(downside_returns) / len(downside_returns)
                var_downside = sum((x - mean_downside) ** 2 for x in downside_returns) / (len(downside_returns) - 1) if len(downside_returns) > 1 else 0.0
                std_downside = math.sqrt(max(0.0, var_downside))
            else:
                std_downside = 0.0
            
            ann_factor = _annualization_factor(interval_ms)
            if std_lr > 0.0:
                sharpe = (mean_lr / std_lr) * ann_factor
            if std_downside > 0.0:
                sortino = (mean_lr / std_downside) * ann_factor
    
    # Calmar ratio = CAGR / max_drawdown
    calmar: float | None = None
    if max_dd > 0.0 and len(equity) >= 2:
        # Approximate CAGR from total return and time
        # For simplicity, assume we can estimate from equity curve length
        # In practice, use actual timestamps if available
        if total_return > -1.0:  # Avoid log of negative
            # Rough estimate: assume backtest covers reasonable time period
            # This is approximate - for exact Calmar, use actual time duration
            cagr_approx = total_return  # Simplified
            calmar = _safe_div(cagr_approx, max_dd)
    
    # Win rate and profit factor from trades
    win_rate: float | None = None
    profit_factor: float | None = None
    avg_trade: float | None = None
    
    if trades:
        # Extract trade PnL
        trade_pnls: list[float] = []
        prev_equity = initial_cash
        
        for trade in trades:
            # Futures trades have realized_pnl, spot trades we compute from equity change
            if "realized_pnl" in trade:
                pnl = float(trade["realized_pnl"])
            else:
                # For spot: compute PnL from equity change
                curr_equity = float(trade.get("equity_after", prev_equity))
                pnl = curr_equity - prev_equity
                prev_equity = curr_equity
            
            trade_pnls.append(pnl)
        
        if trade_pnls:
            wins = [pnl for pnl in trade_pnls if pnl > 0.0]
            losses = [abs(pnl) for pnl in trade_pnls if pnl < 0.0]
            
            win_rate = _safe_div(len(wins), len(trade_pnls))
            
            total_profit = sum(wins) if wins else 0.0
            total_loss = sum(losses) if losses else 0.0
            profit_factor = _safe_div(total_profit, total_loss) if total_loss > 0.0 else (float("inf") if total_profit > 0.0 else None)
            
            avg_trade = sum(trade_pnls) / len(trade_pnls)
    
    # Average exposure
    exposure_avg = float(sum(exposure) / len(exposure)) if exposure else 0.0
    
    # Turnover = total notional traded / initial capital
    turnover = 0.0
    if trades and initial_cash > 0.0:
        total_notional = sum(float(t.get("notional", 0.0)) for t in trades)
        turnover = _safe_div(total_notional, initial_cash)
    
    return ExpandedMetrics(
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade=avg_trade,
        exposure_avg=exposure_avg,
        turnover=turnover,
        total_return=total_return,
        trade_count=len(trades),
    )

