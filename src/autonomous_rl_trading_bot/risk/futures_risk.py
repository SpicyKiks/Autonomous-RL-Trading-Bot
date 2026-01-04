from __future__ import annotations

from dataclasses import dataclass

from autonomous_rl_trading_bot.common.types import AccountSnapshot, OrderRequest, RiskDecision
from autonomous_rl_trading_bot.risk.kill_switch import KillSwitch, KillSwitchConfig
from autonomous_rl_trading_bot.risk.risk_manager import RiskContext, RiskManager


@dataclass(frozen=True, slots=True)
class FuturesRiskConfig:
    allow_short: bool = True
    max_drawdown: float = 0.3
    min_equity: float = 1e-9

    # Hard caps
    max_leverage_used: float = 20.0          # leverage_used = position_notional / equity
    max_order_quote: float = 0.0             # 0 => disabled
    max_position_notional_quote: float = 0.0 # 0 => disabled


class FuturesRiskManager(RiskManager):
    def __init__(self, cfg: FuturesRiskConfig = FuturesRiskConfig()) -> None:
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

        # short restriction (only blocks "opening a short" when flat)
        if (not self.cfg.allow_short) and proposed_order.side == "sell" and abs(float(account.leverage)) < 1e-12:
            # NOTE: account.leverage in our snapshots is leverage_used; flat => 0
            return RiskDecision(verdict="block", reason="short_disabled")

        # leverage cap (leverage_used is computed by broker/env snapshots)
        if float(self.cfg.max_leverage_used) > 0.0 and float(account.leverage) > float(self.cfg.max_leverage_used):
            return RiskDecision(verdict="block", reason="max_leverage_used")

        # order notional caps need price
        if ctx.last_price is not None and float(ctx.last_price) > 0.0:
            px = float(ctx.last_price)
            qty = float(proposed_order.qty)
            unit = proposed_order.qty_unit.lower().strip()

            if unit == "quote":
                notional = abs(qty)
            else:
                notional = abs(qty) * px

            if float(self.cfg.max_order_quote) > 0.0 and notional > float(self.cfg.max_order_quote):
                return RiskDecision(verdict="block", reason="max_order_quote")

            if float(self.cfg.max_position_notional_quote) > 0.0:
                # approximate: current position_notional ~= leverage_used * equity
                current_pos_notional = max(0.0, float(account.leverage) * max(eq, 0.0))
                if current_pos_notional + notional > float(self.cfg.max_position_notional_quote):
                    return RiskDecision(verdict="block", reason="max_position_notional_quote")

        return RiskDecision(verdict="allow", reason="")

