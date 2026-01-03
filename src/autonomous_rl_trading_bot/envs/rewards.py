from __future__ import annotations

import math


def log_equity_return(equity_prev: float, equity_next: float, eps: float = 1e-12) -> float:
    """Stable log-return reward."""
    a = max(float(equity_prev), eps)
    b = max(float(equity_next), eps)
    return float(math.log(b / a))


def simple_pnl(equity_prev: float, equity_next: float) -> float:
    return float(equity_next - equity_prev)

