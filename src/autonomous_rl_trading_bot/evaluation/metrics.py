from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class BacktestMetrics:
    initial_cash: float
    final_equity: float
    total_return: float
    max_drawdown: float
    trade_count: int
    fee_total: float
    slippage_total: float
    sharpe: Optional[float]
    cagr: Optional[float]
    # Expanded metrics
    annualized_return: Optional[float]
    annualized_volatility: Optional[float]
    sortino: Optional[float]
    calmar: Optional[float]
    max_drawdown_duration_days: Optional[float]
    win_rate: Optional[float]
    profit_factor: Optional[float]
    avg_trade_pnl: Optional[float]
    turnover: Optional[float]
    exposure_avg: Optional[float]
    exposure_max: Optional[float]
    sharpe_ci_lower: Optional[float]
    sharpe_ci_upper: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_cash": self.initial_cash,
            "final_equity": self.final_equity,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "trade_count": self.trade_count,
            "fee_total": self.fee_total,
            "slippage_total": self.slippage_total,
            "sharpe": self.sharpe,
            "cagr": self.cagr,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_pnl": self.avg_trade_pnl,
            "turnover": self.turnover,
            "exposure_avg": self.exposure_avg,
            "exposure_max": self.exposure_max,
            "sharpe_ci_lower": self.sharpe_ci_lower,
            "sharpe_ci_upper": self.sharpe_ci_upper,
        }


def _safe_div(a: float, b: float) -> float:
    if b == 0.0:
        return 0.0
    return a / b


def _annualization_factor(interval_ms: int) -> float:
    # Approximate steps per year for Sharpe. Works for minute/hour/day.
    if interval_ms <= 0:
        return 0.0
    seconds_per_step = interval_ms / 1000.0
    steps_per_year = (365.0 * 24.0 * 60.0 * 60.0) / seconds_per_step
    return math.sqrt(steps_per_year)


def _compute_max_drawdown_duration(
    open_time_ms: List[int],
    drawdown: List[float],
    max_dd: float,
) -> Optional[float]:
    """Compute maximum drawdown duration in days."""
    if not drawdown or max_dd <= 0.0:
        return None
    
    # Find all periods where drawdown >= 90% of max DD
    threshold = max_dd * 0.9
    in_dd = False
    dd_start_ms: Optional[int] = None
    max_duration_days = 0.0
    
    for i, (ms, dd) in enumerate(zip(open_time_ms, drawdown)):
        if dd >= threshold:
            if not in_dd:
                in_dd = True
                dd_start_ms = ms
        else:
            if in_dd and dd_start_ms is not None:
                duration_ms = ms - dd_start_ms
                duration_days = duration_ms / (24 * 60 * 60 * 1000.0)
                max_duration_days = max(max_duration_days, duration_days)
                in_dd = False
                dd_start_ms = None
    
    # Handle case where drawdown persists to end
    if in_dd and dd_start_ms is not None and len(open_time_ms) > 0:
        duration_ms = open_time_ms[-1] - dd_start_ms
        duration_days = duration_ms / (24 * 60 * 60 * 1000.0)
        max_duration_days = max(max_duration_days, duration_days)
    
    return max_duration_days if max_duration_days > 0.0 else None


def _bootstrap_sharpe_ci(
    returns: List[float],
    interval_ms: int,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Bootstrap confidence interval for Sharpe ratio using block bootstrap.
    
    Args:
        returns: List of log returns
        interval_ms: Interval in milliseconds for annualization
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0.95 for 95% CI)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (lower_bound, upper_bound) or (None, None) if insufficient data
    """
    if len(returns) < 10:
        return None, None
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    ann_factor = _annualization_factor(interval_ms)
    sharpe_samples: List[float] = []
    
    # Block bootstrap: use daily blocks (approximate)
    # For simplicity, use fixed block size of ~daily returns
    block_size = max(1, int(24 * 60 * 60 * 1000 / interval_ms)) if interval_ms > 0 else 10
    block_size = min(block_size, len(returns) // 2)  # Don't make blocks too large
    
    for _ in range(n_bootstrap):
        # Sample blocks with replacement
        bootstrap_returns: List[float] = []
        while len(bootstrap_returns) < len(returns):
            start_idx = random.randint(0, len(returns) - block_size)
            block = returns[start_idx : start_idx + block_size]
            bootstrap_returns.extend(block)
        
        bootstrap_returns = bootstrap_returns[:len(returns)]
        
        if len(bootstrap_returns) >= 2:
            mean_ret = sum(bootstrap_returns) / len(bootstrap_returns)
            var_ret = sum((x - mean_ret) ** 2 for x in bootstrap_returns) / (len(bootstrap_returns) - 1)
            std_ret = math.sqrt(max(0.0, var_ret))
            
            if std_ret > 0.0:
                sharpe = (mean_ret / std_ret) * ann_factor
                sharpe_samples.append(sharpe)
    
    if len(sharpe_samples) < 10:
        return None, None
    
    sharpe_samples.sort()
    alpha = 1.0 - confidence
    lower_idx = int(len(sharpe_samples) * (alpha / 2))
    upper_idx = int(len(sharpe_samples) * (1 - alpha / 2))
    
    return sharpe_samples[lower_idx], sharpe_samples[upper_idx]


def compute_metrics(
    *,
    open_time_ms: List[int],
    equity: List[float],
    drawdown: List[float],
    trade_count: int,
    fee_total: float,
    slippage_total: float,
    interval_ms: int,
    trades: Optional[List[Dict[str, Any]]] = None,
    exposure: Optional[List[float]] = None,
    seed: Optional[int] = None,
) -> BacktestMetrics:
    if not equity:
        raise ValueError("equity curve is empty")
    if len(open_time_ms) != len(equity) or len(drawdown) != len(equity):
        raise ValueError("equity/drawdown/time lengths mismatch")

    initial_cash = float(equity[0])
    final_equity = float(equity[-1])
    total_return = _safe_div(final_equity, initial_cash) - 1.0
    max_dd = float(max(drawdown)) if drawdown else 0.0

    # Equity log-returns for Sharpe/Sortino
    lr: List[float] = []
    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    annualized_return: Optional[float] = None
    annualized_volatility: Optional[float] = None
    sharpe_ci_lower: Optional[float] = None
    sharpe_ci_upper: Optional[float] = None
    
    if len(equity) >= 3:
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
            
            # Downside deviation for Sortino
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
                annualized_return = mean_lr * ann_factor * ann_factor  # Convert to annualized return
                annualized_volatility = std_lr * ann_factor
                
                # Bootstrap Sharpe CI
                sharpe_ci_lower, sharpe_ci_upper = _bootstrap_sharpe_ci(lr, interval_ms, seed=seed)
            
            if std_downside > 0.0:
                sortino = (mean_lr / std_downside) * ann_factor

    # CAGR (use ms timestamps)
    cagr: Optional[float] = None
    if open_time_ms and len(open_time_ms) >= 2:
        dur_days = (open_time_ms[-1] - open_time_ms[0]) / 86_400_000.0
        if dur_days > 0.0 and initial_cash > 0.0 and final_equity > 0.0:
            years = dur_days / 365.0
            if years >= (1.0 / 365.0):
                ratio = final_equity / initial_cash
                cagr = math.exp(math.log(ratio) / years) - 1.0
    
    # Calmar ratio
    calmar: Optional[float] = None
    if max_dd > 0.0 and cagr is not None:
        calmar = _safe_div(cagr, max_dd)
    
    # Max drawdown duration
    max_dd_duration = _compute_max_drawdown_duration(open_time_ms, drawdown, max_dd)
    
    # Trade metrics
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    avg_trade_pnl: Optional[float] = None
    turnover: Optional[float] = None
    
    if trades:
        trade_pnls: List[float] = []
        prev_equity = initial_cash
        
        for trade in trades:
            if "realized_pnl" in trade:
                pnl = float(trade["realized_pnl"])
            else:
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
            
            avg_trade_pnl = sum(trade_pnls) / len(trade_pnls)
            
            # Turnover
            total_notional = sum(float(t.get("notional", 0.0)) for t in trades)
            turnover = _safe_div(total_notional, initial_cash)
    
    # Exposure stats
    exposure_avg: Optional[float] = None
    exposure_max: Optional[float] = None
    if exposure:
        exposure_avg = float(sum(exposure) / len(exposure))
        exposure_max = float(max(exposure))

    return BacktestMetrics(
        initial_cash=initial_cash,
        final_equity=final_equity,
        total_return=total_return,
        max_drawdown=max_dd,
        trade_count=int(trade_count),
        fee_total=float(fee_total),
        slippage_total=float(slippage_total),
        sharpe=sharpe,
        cagr=cagr,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sortino=sortino,
        calmar=calmar,
        max_drawdown_duration_days=max_dd_duration,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_pnl=avg_trade_pnl,
        turnover=turnover,
        exposure_avg=exposure_avg,
        exposure_max=exposure_max,
        sharpe_ci_lower=sharpe_ci_lower,
        sharpe_ci_upper=sharpe_ci_upper,
    )
