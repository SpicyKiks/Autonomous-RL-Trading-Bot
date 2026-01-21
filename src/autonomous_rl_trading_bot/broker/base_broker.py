from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from autonomous_rl_trading_bot.common.types import (
    AccountSnapshot,
    Fill,
    OrderAck,
    OrderRequest,
    Position,
)


class BrokerAdapter(ABC):
    """Unified broker interface.

    Implementations:
      - paper: deterministic fills for offline backtests and demos
      - live: exchange API orders + reconciliation
    """

    @abstractmethod
    def get_account_snapshot(self) -> AccountSnapshot:
        raise NotImplementedError

    @abstractmethod
    def get_open_positions(self, *, symbol: str | None = None) -> list[Position]:
        raise NotImplementedError

    @abstractmethod
    def submit_order(self, order: OrderRequest) -> OrderAck:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def iter_fills(self, *, since_ts_ms: int | None = None) -> Iterable[Fill]:
        """Yield fills since a timestamp. (Implementations may return a list.)"""
        raise NotImplementedError

    def close(self) -> None:
        """Optional cleanup hook."""
        return

