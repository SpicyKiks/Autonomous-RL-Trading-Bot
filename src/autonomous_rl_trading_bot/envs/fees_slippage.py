from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Side = Literal["buy", "sell"]


def slippage_frac(slippage_bps: float) -> float:
    return max(0.0, float(slippage_bps) / 10_000.0)


def apply_slippage(price: float, side: Side, slippage_bps: float) -> float:
    p = float(price)
    if p <= 0.0:
        return p
    s = slippage_frac(slippage_bps)
    if side == "buy":
        return p * (1.0 + s)
    return p * (1.0 - s)


def taker_fee(notional: float, taker_fee_rate: float) -> float:
    return max(0.0, float(notional) * max(0.0, float(taker_fee_rate)))


@dataclass
class SpotExecResult:
    side: Side
    qty_base: float
    fill_price: float
    notional_quote: float
    fee_quote: float
    slippage_cost_quote: float
    cash_after: float
    qty_after: float
    avg_entry_after: float


def execute_spot_buy(
    *,
    cash: float,
    qty_base: float,
    avg_entry: float,
    mid_price: float,
    order_size_quote: float,
    taker_fee_rate: float,
    slippage_bps: float,
) -> SpotExecResult | None:
    """Market BUY on spot.

    order_size_quote:
      - <=0 : all-in
      - >0  : spend up to this amount (quote)
    """
    if cash <= 0.0 or mid_price <= 0.0:
        return None

    fill_price = apply_slippage(mid_price, "buy", slippage_bps)

    desired_notional = cash if order_size_quote <= 0.0 else min(cash, float(order_size_quote))
    fee_rate = max(0.0, float(taker_fee_rate))

    # Need to ensure notional + fee <= cash
    notional = min(desired_notional, cash / (1.0 + fee_rate))
    if notional <= 0.0:
        return None

    fee = taker_fee(notional, fee_rate)
    qty_add = notional / fill_price
    slippage_cost = max(0.0, (fill_price - mid_price) * qty_add)

    new_qty = qty_base + qty_add
    if new_qty > 0.0:
        if qty_base <= 0.0:
            new_avg = fill_price
        else:
            new_avg = (qty_base * avg_entry + qty_add * fill_price) / new_qty
    else:
        new_avg = 0.0

    cash_after = cash - notional - fee

    return SpotExecResult(
        side="buy",
        qty_base=float(qty_add),
        fill_price=float(fill_price),
        notional_quote=float(notional),
        fee_quote=float(fee),
        slippage_cost_quote=float(slippage_cost),
        cash_after=float(cash_after),
        qty_after=float(new_qty),
        avg_entry_after=float(new_avg),
    )


def execute_spot_sell(
    *,
    cash: float,
    qty_base: float,
    avg_entry: float,
    mid_price: float,
    order_size_quote: float,
    taker_fee_rate: float,
    slippage_bps: float,
    close_all: bool,
) -> SpotExecResult | None:
    """Market SELL on spot.

    If close_all True => sell entire qty_base.
    Else uses order_size_quote:
      - <=0 : sell all
      - >0  : sell up to this notional (quote)
    """
    if qty_base <= 0.0 or mid_price <= 0.0:
        return None

    fill_price = apply_slippage(mid_price, "sell", slippage_bps)
    fee_rate = max(0.0, float(taker_fee_rate))

    if close_all or order_size_quote <= 0.0:
        qty_to_sell = qty_base
    else:
        desired_notional = float(order_size_quote)
        qty_to_sell = min(qty_base, desired_notional / fill_price)

    if qty_to_sell <= 0.0:
        return None

    notional = qty_to_sell * fill_price
    fee = taker_fee(notional, fee_rate)
    slippage_cost = max(0.0, (mid_price - fill_price) * qty_to_sell)

    cash_after = cash + notional - fee
    qty_after = qty_base - qty_to_sell
    if qty_after <= 1e-12:
        qty_after = 0.0
        avg_after = 0.0
    else:
        avg_after = float(avg_entry)

    return SpotExecResult(
        side="sell",
        qty_base=float(qty_to_sell),
        fill_price=float(fill_price),
        notional_quote=float(notional),
        fee_quote=float(fee),
        slippage_cost_quote=float(slippage_cost),
        cash_after=float(cash_after),
        qty_after=float(qty_after),
        avg_entry_after=float(avg_after),
    )

