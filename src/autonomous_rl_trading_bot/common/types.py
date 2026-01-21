from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---- Generic trading primitives (mode-agnostic) ----

MarketType = Literal["spot", "futures"]
Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]

# Quantity unit for an order:
# - base: spot base asset units (e.g., BTC)
# - quote: spot quote currency units (e.g., USDT)
# - contracts: futures contract quantity
OrderQtyUnit = Literal["base", "quote", "contracts"]


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """Unified order request used by paper + live broker adapters."""
    symbol: str
    side: Side
    order_type: OrderType
    qty: float
    qty_unit: OrderQtyUnit
    price: float | None = None
    reduce_only: bool = False
    client_order_id: str | None = None


OrderStatus = Literal["new", "partially_filled", "filled", "canceled", "rejected"]


@dataclass(frozen=True, slots=True)
class OrderAck:
    order_id: str
    status: OrderStatus
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: str
    ts_ms: int
    price: float
    qty: float
    fee_paid: float
    fee_asset: str


@dataclass(frozen=True, slots=True)
class Position:
    """Mode-agnostic view of a position."""
    symbol: str
    side: Side
    qty: float
    entry_price: float


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """Minimal account snapshot used for risk gating + monitoring."""
    ts_ms: int
    equity: float
    available_cash: float
    unrealized_pnl: float = 0.0
    used_margin: float = 0.0
    leverage: float = 0.0


# ---- Risk interface primitives ----

RiskVerdict = Literal["allow", "block", "override"]


@dataclass(frozen=True, slots=True)
class RiskDecision:
    verdict: RiskVerdict
    reason: str = ""
    # If verdict == "override", the risk layer may rewrite the order.
    override_order: OrderRequest | None = None

