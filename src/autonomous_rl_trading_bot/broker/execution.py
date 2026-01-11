from __future__ import annotations

from typing import Optional

from autonomous_rl_trading_bot.common.types import OrderQtyUnit, OrderRequest, Side


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x to [lo, hi]."""
    return max(lo, min(hi, float(x)))


def target_qty_from_fraction(
    equity: float,
    price: float,
    fraction: float,
    market_type: str,
    leverage: float = 1.0,
) -> float:
    """
    Compute target quantity from a position fraction.

    Args:
        equity: Current equity
        price: Current price
        fraction: Target position fraction (-1 to +1, or 0 to +1 for spot)
        market_type: "spot" or "futures"
        leverage: Leverage multiplier (futures only)

    Returns:
        Target quantity (signed for futures, >=0 for spot)
    """
    equity = float(equity)
    price = float(price)
    fraction = float(fraction)
    leverage = float(leverage)

    if price <= 0.0 or equity <= 0.0:
        return 0.0

    mt = (market_type or "").strip().lower()
    if mt == "spot":
        fraction = max(0.0, min(1.0, fraction))
        notional = equity * fraction
        return notional / price

    # Futures: fraction can be negative (short)
    fraction = clamp(fraction, -1.0, 1.0)
    notional = equity * abs(fraction) * leverage
    qty = notional / price
    return qty if fraction >= 0 else -qty


def order_for_target_fraction(
    symbol: str,
    current_qty: float,
    equity: float,
    price: float,
    target_fraction: float,
    market_type: str,
    qty_unit: OrderQtyUnit,
    leverage: float = 1.0,
    reduce_only: bool = False,
) -> Optional[OrderRequest]:
    """
    Generate an OrderRequest to move from current_qty to target_fraction.

    Args:
        symbol: Trading symbol
        current_qty: Current position quantity (signed for futures, >=0 for spot)
        equity: Current equity
        price: Current price
        target_fraction: Target position fraction
        market_type: "spot" or "futures"
        qty_unit: Unit for order quantity
        leverage: Leverage (futures only)
        reduce_only: If True, only allow reducing position

    Returns:
        OrderRequest if delta is non-zero, None otherwise
    """
    target_qty = target_qty_from_fraction(equity, price, target_fraction, market_type, leverage)

    delta = target_qty - float(current_qty)
    if abs(delta) < 1e-12:
        return None

    mt = (market_type or "").strip().lower()
    if mt == "spot":
        if delta > 0:
            side: Side = "buy"
            order_qty = delta
        else:
            side = "sell"
            order_qty = abs(delta)
    else:
        # Futures: delta sign determines side
        if delta > 0:
            side = "buy"
            order_qty = delta
        else:
            side = "sell"
            order_qty = abs(delta)

    if reduce_only:
        # Check if this reduces position
        if mt == "spot":
            if delta > 0:
                return None  # Can't reduce by buying
        else:
            if (delta > 0 and current_qty < 0) or (delta < 0 and current_qty > 0):
                return None  # Wrong direction for reduction

    return OrderRequest(
        symbol=symbol,
        side=side,
        order_type="market",
        qty=float(order_qty),
        qty_unit=qty_unit,
        price=None,
        reduce_only=reduce_only,
    )

