from __future__ import annotations

from dataclasses import dataclass

from autonomous_rl_trading_bot.broker.base_broker import BrokerAdapter


@dataclass
class LivePosition:
    qty: float
    entry_price: float


class PositionSync:
    """
    Position sync helper.

    - Paper mode: runner is source of truth, uses set_local().
    - Exchange mode: can poll broker each step using sync_from_broker().
    """

    def __init__(self, broker: BrokerAdapter | None = None, symbol: str | None = None, market_type: str = "spot"):
        self._broker = broker
        self._symbol = symbol
        self._market_type = market_type
        self._last: LivePosition | None = None

    def set_local(self, pos: LivePosition) -> None:
        self._last = pos

    def get(self) -> LivePosition | None:
        return self._last

    def sync_from_broker(self) -> LivePosition | None:
        if self._broker is None or not self._symbol:
            return None

        positions = list(self._broker.get_open_positions(symbol=self._symbol))

        if self._market_type == "spot":
            # Spot returns base inventory as Position(symbol="BTC/USDT", qty=...)
            p = positions[0] if positions else None
            pos = LivePosition(qty=(p.qty if p else 0.0), entry_price=(p.entry_price if p else 0.0))
        else:
            p = positions[0] if positions else None
            signed_qty = 0.0
            if p:
                signed_qty = p.qty if p.side == "buy" else -p.qty
            pos = LivePosition(qty=signed_qty, entry_price=(p.entry_price if p else 0.0))

        self._last = pos
        return pos
