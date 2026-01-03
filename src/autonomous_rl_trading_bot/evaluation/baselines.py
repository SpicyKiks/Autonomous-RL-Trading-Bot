from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Protocol


class Strategy(Protocol):
    """Deterministic strategy mapping (t, price) -> action int.

    Spot actions (engine interprets):
      0=Hold, 1=Buy, 2=Sell, 3=Close

    Futures actions (engine interprets):
      0=Hold, 1=Buy/Increase Long (or reduce short), 2=Sell/Increase Short (or reduce long), 3=Close
    """

    def act(self, t: int, price: float) -> int: ...


def _norm_name(name: str) -> str:
    return (name or "").strip().lower().replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class BuyAndHoldStrategy:
    """Buy/Long at first step. Engine force-closes at the end."""

    def act(self, t: int, price: float) -> int:
        if t == 0:
            return 1
        return 0


@dataclass
class SMACrossoverStrategy:
    """SMA fast/slow crossover.

    Emits:
      - BUY (1) on bullish crossover (fast crosses above slow)
      - SELL (2) on bearish crossover (fast crosses below slow)
    """

    fast: int
    slow: int

    _fast_q: Deque[float] = field(default_factory=lambda: deque())
    _slow_q: Deque[float] = field(default_factory=lambda: deque())
    _fast_sum: float = 0.0
    _slow_sum: float = 0.0
    _prev_diff: Optional[float] = None

    def __post_init__(self) -> None:
        if self.fast <= 0 or self.slow <= 0:
            raise ValueError("SMA periods must be > 0")
        if self.fast >= self.slow:
            raise ValueError("SMA requires fast < slow")
        object.__setattr__(self, "_fast_q", deque(maxlen=self.fast))
        object.__setattr__(self, "_slow_q", deque(maxlen=self.slow))

    def act(self, t: int, price: float) -> int:
        # Update fast window
        if len(self._fast_q) == self._fast_q.maxlen:
            self._fast_sum -= self._fast_q[0]
        self._fast_q.append(price)
        self._fast_sum += price

        # Update slow window
        if len(self._slow_q) == self._slow_q.maxlen:
            self._slow_sum -= self._slow_q[0]
        self._slow_q.append(price)
        self._slow_sum += price

        if len(self._fast_q) < self.fast or len(self._slow_q) < self.slow:
            return 0

        fast_sma = self._fast_sum / self.fast
        slow_sma = self._slow_sum / self.slow
        diff = fast_sma - slow_sma

        action = 0
        if self._prev_diff is not None:
            if self._prev_diff <= 0.0 and diff > 0.0:
                action = 1  # bullish cross => buy
            elif self._prev_diff >= 0.0 and diff < 0.0:
                action = 2  # bearish cross => sell
        self._prev_diff = diff
        return action


@dataclass
class EMACrossoverStrategy:
    """EMA fast/slow crossover.

    Uses standard EMA with alpha=2/(period+1).
    Emits BUY on bullish cross, SELL on bearish cross.
    """

    fast: int
    slow: int

    _ema_fast: Optional[float] = None
    _ema_slow: Optional[float] = None
    _prev_diff: Optional[float] = None

    def __post_init__(self) -> None:
        if self.fast <= 0 or self.slow <= 0:
            raise ValueError("EMA periods must be > 0")
        if self.fast >= self.slow:
            raise ValueError("EMA requires fast < slow")

    def act(self, t: int, price: float) -> int:
        a_fast = 2.0 / (self.fast + 1.0)
        a_slow = 2.0 / (self.slow + 1.0)

        if self._ema_fast is None:
            self._ema_fast = price
        else:
            self._ema_fast = (a_fast * price) + (1.0 - a_fast) * self._ema_fast

        if self._ema_slow is None:
            self._ema_slow = price
        else:
            self._ema_slow = (a_slow * price) + (1.0 - a_slow) * self._ema_slow

        diff = self._ema_fast - self._ema_slow
        action = 0
        if self._prev_diff is not None:
            if self._prev_diff <= 0.0 and diff > 0.0:
                action = 1
            elif self._prev_diff >= 0.0 and diff < 0.0:
                action = 2
        self._prev_diff = diff
        return action


@dataclass
class RSIReversionStrategy:
    """RSI mean-reversion using Wilder smoothing.

    - BUY when RSI < low
    - SELL when RSI > high
    Uses a latch so it fires once per excursion.
    """

    period: int
    low: float
    high: float

    _prev_price: Optional[float] = None
    _avg_gain: Optional[float] = None
    _avg_loss: Optional[float] = None
    _seed_gains: Deque[float] = field(default_factory=lambda: deque())
    _seed_losses: Deque[float] = field(default_factory=lambda: deque())
    _armed_buy: bool = True
    _armed_sell: bool = True

    def __post_init__(self) -> None:
        if self.period <= 1:
            raise ValueError("RSI period must be > 1")
        if not (0.0 < self.low < 100.0 and 0.0 < self.high < 100.0):
            raise ValueError("RSI thresholds must be in (0,100)")
        if self.low >= self.high:
            raise ValueError("RSI requires low < high")
        object.__setattr__(self, "_seed_gains", deque(maxlen=self.period))
        object.__setattr__(self, "_seed_losses", deque(maxlen=self.period))

    def _compute_rsi(self) -> Optional[float]:
        if self._avg_gain is None or self._avg_loss is None:
            return None
        if self._avg_loss == 0.0:
            return 100.0
        rs = self._avg_gain / self._avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def act(self, t: int, price: float) -> int:
        if self._prev_price is None:
            self._prev_price = price
            return 0

        change = price - self._prev_price
        self._prev_price = price

        gain = max(0.0, change)
        loss = max(0.0, -change)

        # Seed the Wilder averages for the first `period` steps
        if self._avg_gain is None or self._avg_loss is None:
            self._seed_gains.append(gain)
            self._seed_losses.append(loss)
            if len(self._seed_gains) < self.period:
                return 0
            self._avg_gain = sum(self._seed_gains) / self.period
            self._avg_loss = sum(self._seed_losses) / self.period
        else:
            # Wilder smoothing
            self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
            self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period

        rsi = self._compute_rsi()
        if rsi is None:
            return 0

        # Rearm when back in neutral zone
        if self.low <= rsi <= self.high:
            self._armed_buy = True
            self._armed_sell = True
            return 0

        if rsi < self.low and self._armed_buy:
            self._armed_buy = False
            self._armed_sell = True
            return 1

        if rsi > self.high and self._armed_sell:
            self._armed_sell = False
            self._armed_buy = True
            return 2

        return 0


def make_strategy(name: str, params: Optional[Dict[str, float | int]] = None) -> Strategy:
    """Create a strategy by name with optional params dict.

    Supported:
      - buy_and_hold
      - sma_crossover (fast, slow)
      - ema_crossover (fast, slow)
      - rsi_reversion (period, low, high)
    """
    n = _norm_name(name)
    p = params or {}

    if n in ("buy_and_hold", "buyhold", "buy_hold", "buy_and_hold_strategy"):
        return BuyAndHoldStrategy()

    if n in ("sma_crossover", "sma", "sma_cross"):
        fast = int(p.get("fast", 10))
        slow = int(p.get("slow", 30))
        return SMACrossoverStrategy(fast=fast, slow=slow)

    if n in ("ema_crossover", "ema", "ema_cross"):
        fast = int(p.get("fast", 12))
        slow = int(p.get("slow", 26))
        return EMACrossoverStrategy(fast=fast, slow=slow)

    if n in ("rsi_reversion", "rsi", "rsi_mean_reversion", "rsi_mr"):
        period = int(p.get("period", 14))
        low = float(p.get("low", 30.0))
        high = float(p.get("high", 70.0))
        return RSIReversionStrategy(period=period, low=low, high=high)

    raise ValueError(f"Unknown strategy: {name}")
