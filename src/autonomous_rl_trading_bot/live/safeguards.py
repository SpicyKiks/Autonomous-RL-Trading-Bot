from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path


def _now_s() -> float:
    return time.time()


@dataclass
class SafeguardsConfig:
    # If this file exists, the runner will close positions and stop.
    kill_switch_path: str = ""

    # Max drawdown from peak equity (fraction). e.g. 0.2 = 20%.
    max_drawdown: float = 0.25

    # Trade rate limiting
    min_seconds_between_trades: float = 15.0
    max_trades_per_hour: int = 30


class Safeguards:
    def __init__(self, cfg: SafeguardsConfig):
        self.cfg = cfg
        self._last_trade_s: float = 0.0
        self._trade_times_s: list[float] = []

    def kill_switch_triggered(self) -> bool:
        p = (self.cfg.kill_switch_path or "").strip()
        if not p:
            return False
        p = os.path.expandvars(p)
        return Path(p).exists()

    def allow_trade(self) -> bool:
        """Rate-limit trades (defensive against runaway policies)."""
        now = _now_s()

        if self._last_trade_s and (now - self._last_trade_s) < float(
            self.cfg.min_seconds_between_trades
        ):
            return False

        horizon = now - 3600.0
        self._trade_times_s = [t for t in self._trade_times_s if t >= horizon]
        if int(self.cfg.max_trades_per_hour) > 0 and len(self._trade_times_s) >= int(
            self.cfg.max_trades_per_hour
        ):
            return False

        return True

    def record_trade(self) -> None:
        now = _now_s()
        self._last_trade_s = now
        self._trade_times_s.append(now)

    def drawdown_breached(self, *, equity: float, peak_equity: float) -> bool:
        if peak_equity <= 0.0:
            return False
        dd = max(0.0, 1.0 - (equity / peak_equity))
        return dd >= float(self.cfg.max_drawdown)

    def summary(self) -> dict:
        return {
            "kill_switch_path": self.cfg.kill_switch_path,
            "max_drawdown": float(self.cfg.max_drawdown),
            "min_seconds_between_trades": float(self.cfg.min_seconds_between_trades),
            "max_trades_per_hour": int(self.cfg.max_trades_per_hour),
        }

