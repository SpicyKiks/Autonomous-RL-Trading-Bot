from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from autonomous_rl_trading_bot.broker.base_broker import BrokerAdapter
from autonomous_rl_trading_bot.common.types import (
    AccountSnapshot,
    Fill,
    OrderAck,
    OrderRequest,
    Position,
)
from autonomous_rl_trading_bot.envs.fees_slippage import apply_slippage, taker_fee


@dataclass(frozen=True, slots=True)
class PaperFuturesConfig:
    initial_cash: float = 1000.0
    leverage: float = 3.0
    maintenance_margin_rate: float = 0.005
    allow_short: bool = True
    taker_fee_rate: float = 0.001
    slippage_bps: float = 5.0


@dataclass
class PaperFuturesState:
    margin_balance: float
    qty: float = 0.0
    entry_price: float = 0.0
    peak_equity: float = 0.0


class PaperFuturesBroker(BrokerAdapter):
    """
    Minimal paper USD-M futures broker (single symbol, single net position).

    qty > 0 => long
    qty < 0 => short

    Margin model:
      used_margin = |qty|*price / leverage
      equity      = margin_balance + qty*(price-entry)
      liquidation when equity <= maintenance_margin_rate * |qty|*price
    """

    def __init__(self, *, symbol: str, cfg: PaperFuturesConfig = PaperFuturesConfig()) -> None:
        self.symbol = symbol.upper()
        self.cfg = cfg

        self._state = PaperFuturesState(
            margin_balance=float(cfg.initial_cash),
            qty=0.0,
            entry_price=0.0,
            peak_equity=float(cfg.initial_cash),
        )
        self._last_price: Optional[float] = None
        self._last_ts_ms: Optional[int] = None
        self._fills: List[Fill] = []
        self._order_seq: int = 0

    def update_market_price(self, *, symbol: str, price: float, ts_ms: int) -> None:
        if symbol.upper() != self.symbol:
            return
        p = float(price)
        if p <= 0.0:
            return
        self._last_price = p
        self._last_ts_ms = int(ts_ms)

        eq = self._equity(p)
        self._state.peak_equity = max(self._state.peak_equity, eq)

    # ---------- math helpers ----------

    def _unrealized_pnl(self, price: float) -> float:
        if abs(self._state.qty) <= 1e-18:
            return 0.0
        return float(self._state.qty * (float(price) - float(self._state.entry_price)))

    def _equity(self, price: float) -> float:
        return float(self._state.margin_balance + self._unrealized_pnl(price))

    def _pos_notional(self, price: float) -> float:
        return float(abs(self._state.qty) * float(price))

    def _used_margin(self, price: float) -> float:
        return float(self._pos_notional(price) / max(1e-12, float(self.cfg.leverage)))

    def _maintenance_margin(self, price: float) -> float:
        return float(self._pos_notional(price) * max(0.0, float(self.cfg.maintenance_margin_rate)))

    # ---------- BrokerAdapter interface ----------

    def get_account_snapshot(self) -> AccountSnapshot:
        px = float(self._last_price) if self._last_price is not None else 0.0
        eq = self._equity(px) if px > 0.0 else float(self._state.margin_balance)
        used = self._used_margin(px) if px > 0.0 else 0.0
        lev_used = (self._pos_notional(px) / eq) if (px > 0.0 and eq > 0.0) else 0.0

        return AccountSnapshot(
            ts_ms=int(self._last_ts_ms or 0),
            equity=float(eq),
            available_cash=float(self._state.margin_balance),
            unrealized_pnl=float(self._unrealized_pnl(px)),
            used_margin=float(used),
            leverage=float(lev_used),
        )

    def get_open_positions(self, *, symbol: Optional[str] = None) -> list[Position]:
        if symbol is not None and symbol.upper() != self.symbol:
            return []
        if abs(self._state.qty) <= 1e-18:
            return []
        side = "buy" if self._state.qty > 0 else "sell"
        return [
            Position(
                symbol=self.symbol,
                side=side,  # buy=long, sell=short
                qty=float(abs(self._state.qty)),
                entry_price=float(self._state.entry_price),
            )
        ]

    def submit_order(self, req: OrderRequest) -> OrderAck:
        if req.symbol.upper() != self.symbol:
            return OrderAck(order_id="", status="rejected", reason=f"PaperFuturesBroker only supports {self.symbol}")

        if self._last_price is None or self._last_price <= 0.0:
            return OrderAck(order_id="", status="rejected", reason="No market price set; call update_market_price() first")

        if req.order_type.lower().strip() != "market":
            return OrderAck(order_id="", status="rejected", reason="PaperFuturesBroker supports market orders only")

        px = float(self._last_price)
        ts_ms = int(self._last_ts_ms or 0)
        side = req.side.lower().strip()
        qty = float(req.qty)
        qty_unit = req.qty_unit.lower().strip()

        # short permission
        if (not self.cfg.allow_short) and side == "sell" and abs(self._state.qty) <= 1e-18:
            return OrderAck(order_id="", status="rejected", reason="short_disabled")

        self._order_seq += 1
        order_id = f"paper_futures_{self._order_seq}"

        # convert qty to contracts
        fill = apply_slippage(px, side, float(self.cfg.slippage_bps))

        if qty_unit == "quote":
            desired_contracts = qty / max(1e-12, fill)
        else:
            # base/contracts treated the same
            desired_contracts = qty

        desired_contracts = max(0.0, desired_contracts)
        if desired_contracts <= 1e-12:
            return OrderAck(order_id=order_id, status="rejected", reason="qty_too_small")

        # translate side into delta qty
        delta = desired_contracts if side == "buy" else -desired_contracts

        # reduce_only clamp
        if req.reduce_only:
            if abs(self._state.qty) <= 1e-18:
                return OrderAck(order_id=order_id, status="rejected", reason="reduce_only_flat")
            # must reduce existing position, not increase or flip
            if (self._state.qty > 0 and delta > 0) or (self._state.qty < 0 and delta < 0):
                return OrderAck(order_id=order_id, status="rejected", reason="reduce_only_wrong_side")
            # clamp to not flip
            if abs(delta) > abs(self._state.qty):
                delta = -self._state.qty

        # try execute to target = current + delta
        filled_delta, fee_paid, msg = self._execute_to_target_qty(self._state.qty + delta)

        if abs(filled_delta) <= 1e-12:
            return OrderAck(order_id=order_id, status="rejected", reason=msg or "blocked")

        self._fills.append(
            Fill(
                order_id=order_id,
                ts_ms=ts_ms,
                price=float(fill),
                qty=float(abs(filled_delta)),
                fee_paid=float(fee_paid),
                fee_asset="QUOTE",
            )
        )

        return OrderAck(order_id=order_id, status="filled")

    def cancel_order(self, order_id: str) -> bool:
        return False

    def iter_fills(self, *, since_ts_ms: Optional[int] = None) -> Iterable[Fill]:
        if since_ts_ms is None:
            yield from list(self._fills)
            return
        t = int(since_ts_ms)
        for f in self._fills:
            if int(f.ts_ms) >= t:
                yield f

    # ---------- execution core ----------

    def _execute_to_target_qty(self, target_qty: float) -> tuple[float, float, str]:
        """
        Execute one market move from current qty -> target_qty with fee/slippage and margin clamp.
        Returns: (filled_delta, fee_paid, msg)
        """
        assert self._last_price is not None and self._last_price > 0.0
        px = float(self._last_price)

        qty0 = float(self._state.qty)
        target_qty = float(target_qty)

        if abs(target_qty - qty0) <= 1e-12:
            return 0.0, 0.0, "no_change"

        delta = target_qty - qty0
        side = "buy" if delta > 0 else "sell"
        fill = apply_slippage(px, side, float(self.cfg.slippage_bps))

        equity_pre = self._equity(px)

        notional_desired = abs(delta) * fill
        fee_desired = taker_fee(notional_desired, float(self.cfg.taker_fee_rate))

        max_total_abs = max(
            0.0, (equity_pre - fee_desired) * float(self.cfg.leverage) / max(1e-12, fill)
        )

        if abs(target_qty) > max_total_abs + 1e-12:
            target_qty = float((1.0 if target_qty >= 0 else -1.0) * max_total_abs)
            delta = target_qty - qty0
            if abs(delta) <= 1e-12:
                return 0.0, 0.0, "margin_clamped_to_zero"
            notional_desired = abs(delta) * fill
            fee_desired = taker_fee(notional_desired, float(self.cfg.taker_fee_rate))

        # realized pnl for closed portion
        realized = 0.0
        entry0 = float(self._state.entry_price)
        if abs(qty0) > 1e-18 and (delta != 0.0) and ((delta > 0) != (qty0 > 0)):
            closed_abs = min(abs(qty0), abs(delta))
            closed_signed = (1.0 if qty0 > 0 else -1.0) * closed_abs
            realized = float(closed_signed * (fill - entry0))

        self._state.margin_balance = float(self._state.margin_balance + realized - fee_desired)

        qty1 = float(qty0 + delta)
        if abs(qty1) <= 1e-12:
            qty1 = 0.0
            entry1 = 0.0
        else:
            if abs(qty0) <= 1e-18 or ((qty0 > 0) == (delta > 0)):
                if abs(qty0) <= 1e-18:
                    entry1 = fill
                else:
                    entry1 = (abs(qty0) * entry0 + abs(delta) * fill) / max(1e-12, abs(qty1))
            else:
                entry1 = entry0

        self._state.qty = float(qty1)
        self._state.entry_price = float(entry1)

        # liquidation sanity (broker doesn't auto-liquidate, but you can query this)
        msg = "ok"
        mm = self._maintenance_margin(px)
        eq = self._equity(px)
        if abs(self._state.qty) > 1e-18 and eq <= mm:
            msg = "warning:below_maintenance_margin"

        return float(delta), float(fee_desired), msg

