from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class EquityMetrics:
    final_equity: float
    total_return: float
    max_drawdown: float
    sharpe: float


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.maximum(peak, 1e-12)
    return float(np.max(dd))


def sharpe_ratio(returns: np.ndarray, eps: float = 1e-12) -> float:
    # per-step returns assumed; sharpe here is unannualized, just a quality indicator for demo.
    if returns.size < 2:
        return 0.0
    mu = float(np.mean(returns))
    sd = float(np.std(returns))
    if sd < eps:
        return 0.0
    return mu / sd


def compute_equity_metrics(equity: np.ndarray) -> EquityMetrics:
    if equity.size == 0:
        return EquityMetrics(final_equity=0.0, total_return=0.0, max_drawdown=0.0, sharpe=0.0)

    final_eq = float(equity[-1])
    total_ret = float(final_eq / max(float(equity[0]), 1e-12) - 1.0)

    rets = np.zeros_like(equity, dtype=np.float64)
    rets[1:] = (equity[1:] / np.maximum(equity[:-1], 1e-12)) - 1.0

    mdd = max_drawdown(equity)
    sh = sharpe_ratio(rets[1:])
    return EquityMetrics(final_equity=final_eq, total_return=total_ret, max_drawdown=mdd, sharpe=sh)

