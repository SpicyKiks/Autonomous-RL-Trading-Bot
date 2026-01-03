from __future__ import annotations

from dataclasses import dataclass

from autonomous_rl_trading_bot.common.types import AccountSnapshot, OrderRequest, RiskDecision
from autonomous_rl_trading_bot.risk.kill_switch import KillSwitch, KillSwitchConfig
from autonomous_rl_trading_bot.risk.risk_manager import RiskContext, RiskManager


@dataclass(frozen=True, slots=True)
class SpotRiskConfig:
    """Hard risk limits for spot execution."""
    max_drawdown: float = 0.3
    min_equity: float = 1e-9
    max_order_quote: float = 0.0  # 0 => no cap
    max_exposure: float = 1.0


class SpotRiskManager(RiskManager):
    """Simple spot risk gate."""

    def __init__(self, cfg: SpotRiskConfig = SpotRiskConfig()) -> None:
        self.cfg = cfg
        self._kill = KillSwitch(KillSwitchConfig(max_drawdown=cfg.max_drawdown, min_equity=cfg.min_equity))
        self._peak_equity: float = 0.0

    def evaluate(
        self,
        *,
        account: AccountSnapshot,
        proposed_order: OrderRequest,
        ctx: RiskContext,
    ) -> RiskDecision:
        eq = float(account.equity)
        self._peak_equity = max(self._peak_equity, eq) if self._peak_equity > 0.0 else eq

        killed, reason = self._kill.check(equity=eq, peak_equity=self._peak_equity)
        if killed:
            return RiskDecision(verdict="block", reason=f"kill_switch:{reason}")

        if float(self.cfg.max_order_quote) > 0.0 and ctx.last_price is not None:
            px = float(ctx.last_price)
            if px > 0.0:
                qty = float(proposed_order.qty)
                unit = proposed_order.qty_unit.lower().strip()
                if unit == "quote":
                    notional = abs(qty)
                elif unit == "base":
                    notional = abs(qty) * px
                else:
                    notional = abs(qty) * px

                if notional > float(self.cfg.max_order_quote):
                    return RiskDecision(verdict="block", reason="max_order_quote")

        if ctx.last_price is not None and float(self.cfg.max_exposure) < 1.0:
            px = float(ctx.last_price)
            if px > 0.0:
                pos_val = max(0.0, eq - float(account.available_cash))
                exposure = 0.0 if eq <= 0.0 else pos_val / eq
                if exposure > float(self.cfg.max_exposure) + 1e-9 and proposed_order.side.lower().strip() == "buy":
                    return RiskDecision(verdict="block", reason="max_exposure")

        return RiskDecision(verdict="allow", reason="")

