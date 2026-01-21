from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from autonomous_rl_trading_bot.broker.base_broker import BrokerAdapter
from autonomous_rl_trading_bot.common.types import (
    AccountSnapshot,
    Fill,
    OrderAck,
    OrderRequest,
    Position,
)
from autonomous_rl_trading_bot.envs.fees_slippage import execute_spot_buy, execute_spot_sell


@dataclass
class PaperSpotConfig:
    initial_cash: float = 1000.0
    taker_fee_rate: float = 0.001
    slippage_bps: float = 5.0


@dataclass
class PaperSpotState:
    cash: float
    qty_base: float = 0.0
    avg_entry_price: float = 0.0
    peak_equity: float = 0.0


class PaperSpotBroker(BrokerAdapter):
    """Deterministic paper broker for spot.

    - Maintains an internal cash + base inventory state.
    - Executes MARKET orders immediately at last seen market price with slippage + taker fees.
    """

    def __init__(self, *, symbol: str, cfg: PaperSpotConfig = PaperSpotConfig()) -> None:
        self.symbol = symbol.upper()
        self.cfg = cfg

        self._state = PaperSpotState(
            cash=float(cfg.initial_cash),
            qty_base=0.0,
            avg_entry_price=0.0,
            peak_equity=float(cfg.initial_cash),
        )
        self._last_price: float | None = None
        self._last_ts_ms: int | None = None
        self._fills: list[Fill] = []
        self._order_seq: int = 0

    # extra helper for feeding prices
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

    # BrokerAdapter interface
    def get_account_snapshot(self) -> AccountSnapshot:
        price = float(self._last_price or 0.0)
        equity = self._equity(price) if price > 0.0 else float(self._state.cash)
        u_pnl = 0.0
        if self._state.qty_base > 0.0 and price > 0.0 and self._state.avg_entry_price > 0.0:
            u_pnl = (price - float(self._state.avg_entry_price)) * float(self._state.qty_base)

        return AccountSnapshot(
            ts_ms=int(self._last_ts_ms or 0),
            equity=float(equity),
            available_cash=float(self._state.cash),
            unrealized_pnl=float(u_pnl),
            used_margin=0.0,
            leverage=0.0,
        )

    def get_open_positions(self, *, symbol: str | None = None) -> list[Position]:
        if symbol is not None and symbol.upper() != self.symbol:
            return []
        if self._state.qty_base <= 0.0:
            return []
        return [
            Position(
                symbol=self.symbol,
                side="buy",
                qty=float(self._state.qty_base),
                entry_price=float(self._state.avg_entry_price or 0.0),
            )
        ]

    def submit_order(self, req: OrderRequest) -> OrderAck:
        if req.symbol.upper() != self.symbol:
            return OrderAck(order_id="", status="rejected", reason=f"PaperSpotBroker only supports {self.symbol}")

        if self._last_price is None:
            return OrderAck(order_id="", status="rejected", reason="No market price set; call update_market_price() first")

        if req.order_type.lower().strip() != "market":
            return OrderAck(order_id="", status="rejected", reason="PaperSpotBroker supports market orders only")

        price = float(self._last_price)
        ts_ms = int(self._last_ts_ms or 0)
        side = req.side.lower().strip()
        qty = float(req.qty)
        qty_unit = req.qty_unit.lower().strip()
        if qty_unit not in ("quote", "base"):
            return OrderAck(order_id="", status="rejected", reason=f"Invalid qty_unit for spot: {req.qty_unit!r}")

        self._order_seq += 1
        order_id = f"paper_spot_{self._order_seq}"

        if side == "buy":
            order_size_quote = qty if qty_unit == "quote" else qty * price
            r = execute_spot_buy(
                cash=self._state.cash,
                qty_base=self._state.qty_base,
                avg_entry=self._state.avg_entry_price,
                mid_price=price,
                order_size_quote=order_size_quote,
                taker_fee_rate=float(self.cfg.taker_fee_rate),
                slippage_bps=float(self.cfg.slippage_bps),
            )
            if r is None:
                return OrderAck(order_id=order_id, status="rejected", reason="BUY rejected (insufficient cash or invalid price)")

            self._state.cash = r.cash_after
            self._state.qty_base = r.qty_after
            self._state.avg_entry_price = r.avg_entry_after

            self._fills.append(
                Fill(
                    order_id=order_id,
                    ts_ms=ts_ms,
                    price=float(r.fill_price),
                    qty=float(r.qty_base),
                    fee_paid=float(r.fee_quote),
                    fee_asset="QUOTE",
                )
            )
            return OrderAck(order_id=order_id, status="filled")

        if side == "sell":
            close_all = bool(req.reduce_only) and qty <= 0.0
            order_size_quote = (qty * price) if qty_unit == "base" else qty

            r = execute_spot_sell(
                cash=self._state.cash,
                qty_base=self._state.qty_base,
                avg_entry=self._state.avg_entry_price,
                mid_price=price,
                order_size_quote=order_size_quote,
                taker_fee_rate=float(self.cfg.taker_fee_rate),
                slippage_bps=float(self.cfg.slippage_bps),
                close_all=close_all or qty <= 0.0,
            )
            if r is None:
                return OrderAck(order_id=order_id, status="rejected", reason="SELL rejected (insufficient inventory or invalid price)")

            self._state.cash = r.cash_after
            self._state.qty_base = r.qty_after
            self._state.avg_entry_price = r.avg_entry_after

            self._fills.append(
                Fill(
                    order_id=order_id,
                    ts_ms=ts_ms,
                    price=float(r.fill_price),
                    qty=float(r.qty_base),
                    fee_paid=float(r.fee_quote),
                    fee_asset="QUOTE",
                )
            )
            return OrderAck(order_id=order_id, status="filled")

        return OrderAck(order_id=order_id, status="rejected", reason=f"Unsupported side: {req.side!r}")

    def cancel_order(self, order_id: str) -> bool:
        return False

    def iter_fills(self, *, since_ts_ms: int | None = None) -> Iterable[Fill]:
        if since_ts_ms is None:
            yield from list(self._fills)
            return
        t = int(since_ts_ms)
        for f in self._fills:
            if int(f.ts_ms) >= t:
                yield f

    def _equity(self, price: float) -> float:
        return float(self._state.cash + self._state.qty_base * float(price))

