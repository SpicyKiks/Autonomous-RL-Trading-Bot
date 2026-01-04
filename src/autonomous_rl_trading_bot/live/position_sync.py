from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LivePosition:
    """Minimal position snapshot used by the live runner.

    For spot: qty is base asset quantity (>=0).
    For futures: qty is signed contracts quantity (can be <0 for short).
    """

    qty: float
    entry_price: float  # futures avg entry; for spot informational only


class PositionSync:
    """Placeholder for broker/exchange position syncing.

    In paper trading, the live runner is the source of truth and this class is a no-op.
    In a real-money extension (e.g., via CCXT), this can poll open positions from the
    exchange and reconcile state.
    """

    def __init__(self) -> None:
        self._last: Optional[LivePosition] = None

    def set_local(self, pos: LivePosition) -> None:
        self._last = pos

    def get(self) -> Optional[LivePosition]:
        return self._last

