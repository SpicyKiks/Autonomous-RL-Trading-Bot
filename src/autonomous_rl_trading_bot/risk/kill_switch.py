from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KillSwitchConfig:
    max_drawdown: float = 0.3  # 30%
    min_equity: float = 1e-9


class KillSwitch:
    """Stateless checks for hard-stop conditions."""

    def __init__(self, cfg: KillSwitchConfig = KillSwitchConfig()) -> None:
        self.cfg = cfg

    def check(self, *, equity: float, peak_equity: float) -> tuple[bool, str]:
        eq = float(equity)
        peak = float(peak_equity)

        if eq <= float(self.cfg.min_equity):
            return True, "min_equity"

        if peak > 0.0:
            dd = max(0.0, 1.0 - (eq / peak))
            if dd >= float(self.cfg.max_drawdown):
                return True, "max_drawdown"

        return False, ""

