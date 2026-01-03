from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from autonomous_rl_trading_bot.common.types import AccountSnapshot, OrderRequest, RiskDecision


@dataclass(frozen=True, slots=True)
class RiskContext:
    """Extra information the risk layer can use."""
    mode_id: str
    symbol: str
    ts_ms: int
    last_price: Optional[float] = None


class RiskManager(ABC):
    """Risk gate used mainly for live/paper execution loops."""

    @abstractmethod
    def evaluate(
        self,
        *,
        account: AccountSnapshot,
        proposed_order: OrderRequest,
        ctx: RiskContext,
    ) -> RiskDecision:
        raise NotImplementedError

