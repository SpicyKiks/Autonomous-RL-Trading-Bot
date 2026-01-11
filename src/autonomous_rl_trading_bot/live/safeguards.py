from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path


def _now_s() -> float:
    return time.time()


@dataclass
class SafeguardsConfig:
    kill_switch_path: str = ""

    # drawdown kill
    max_drawdown: float = 0.25

    # rate limiting
    min_seconds_between_trades: float = 15.0
    max_trades_per_hour: int = 30
    max_orders_per_minute: int = 5

    # exposure + leverage
    max_exposure_pct: float = 1.0
    max_leverage: float = 3.0

    # execution quality
    max_slippage_bps: float = 50.0
    stale_candle_max_ms: int = 120_000

    # optional stop-loss
    stop_loss_pct: float = 0.0


class Safeguards:
    def __init__(self, cfg: SafeguardsConfig):
        self.cfg = cfg
        self._last_trade_s: float = 0.0
        self._trade_times_s: list[float] = []
        self._killed: bool = False
        self._kill_reason: str = ""

    def kill_switch_triggered(self) -> bool:
        if self._killed:
            return True
        p = (self.cfg.kill_switch_path or "").strip()
        if not p:
            return False
        p = os.path.expandvars(p)
        return Path(p).exists()

    def trigger_kill(self, reason: str) -> None:
        self._killed = True
        self._kill_reason = (reason or "").strip()[:500]
        p = (self.cfg.kill_switch_path or "").strip()
        if p:
            try:
                Path(os.path.expandvars(p)).write_text(self._kill_reason or "killed")
            except Exception:
                pass

    def allow_trade(self) -> bool:
        now = _now_s()

        if self._last_trade_s and (now - self._last_trade_s) < float(self.cfg.min_seconds_between_trades):
            return False

        horizon = now - 3600.0
        self._trade_times_s = [t for t in self._trade_times_s if t >= horizon]
        if int(self.cfg.max_trades_per_hour) > 0 and len(self._trade_times_s) >= int(self.cfg.max_trades_per_hour):
            return False

        if int(self.cfg.max_orders_per_minute) > 0:
            minute_horizon = now - 60.0
            last_minute = [t for t in self._trade_times_s if t >= minute_horizon]
            if len(last_minute) >= int(self.cfg.max_orders_per_minute):
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

    def candle_is_stale(self, *, candle_close_ms: int, now_ms: int) -> bool:
        max_ms = int(self.cfg.stale_candle_max_ms)
        if max_ms <= 0:
            return False
        return (now_ms - int(candle_close_ms)) > max_ms

    def slippage_breached(self, *, expected_price: float, fill_price: float) -> bool:
        if expected_price <= 0 or fill_price <= 0:
            return False
        bps = abs(fill_price - expected_price) / expected_price * 10_000.0
        return bps > float(self.cfg.max_slippage_bps)

    def summary(self) -> dict:
        return {
            "kill_switch_path": self.cfg.kill_switch_path,
            "max_drawdown": float(self.cfg.max_drawdown),
            "min_seconds_between_trades": float(self.cfg.min_seconds_between_trades),
            "max_trades_per_hour": int(self.cfg.max_trades_per_hour),
            "max_orders_per_minute": int(self.cfg.max_orders_per_minute),
            "max_exposure_pct": float(self.cfg.max_exposure_pct),
            "max_leverage": float(self.cfg.max_leverage),
            "max_slippage_bps": float(self.cfg.max_slippage_bps),
            "stale_candle_max_ms": int(self.cfg.stale_candle_max_ms),
            "stop_loss_pct": float(self.cfg.stop_loss_pct),
            "killed": bool(self._killed),
            "kill_reason": self._kill_reason,
        }
