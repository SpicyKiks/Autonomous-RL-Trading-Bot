from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from autonomous_rl_trading_bot.common.timeframes import interval_to_ms

MarketType = Literal["spot", "futures"]


@dataclass(frozen=True, slots=True)
class OhlcvBar:
    """Canonical OHLCV bar used everywhere in v2 code.

    Timestamp convention:
      - ts_ms = bar OPEN time in milliseconds since epoch (UTC)
    """

    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_ccxt_row(self) -> list[float]:
        # CCXT OHLCV format: [timestamp, open, high, low, close, volume]
        return [float(self.ts_ms), self.open, self.high, self.low, self.close, self.volume]

    @staticmethod
    def from_ccxt_row(row: Sequence[float]) -> OhlcvBar:
        if len(row) < 6:
            raise ValueError(f"Invalid CCXT OHLCV row length={len(row)} row={row!r}")
        ts_ms = int(row[0])
        return OhlcvBar(
            ts_ms=ts_ms,
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )


def bars_from_ccxt(rows: Sequence[Sequence[float]]) -> list[OhlcvBar]:
    return [OhlcvBar.from_ccxt_row(r) for r in rows]


def bars_to_frame(bars: Iterable[OhlcvBar]) -> pd.DataFrame:
    """Convert bars -> DataFrame indexed by UTC datetime (open time)."""
    rows = list(bars)
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).set_index(
            pd.DatetimeIndex([], name="ts")
        )

    df = pd.DataFrame(
        {
            "ts_ms": [b.ts_ms for b in rows],
            "open": [b.open for b in rows],
            "high": [b.high for b in rows],
            "low": [b.low for b in rows],
            "close": [b.close for b in rows],
            "volume": [b.volume for b in rows],
        }
    )
    ts = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.drop(columns=["ts_ms"])
    df.index = ts
    df.index.name = "ts"
    return df.sort_index()


def frame_to_bars(df: pd.DataFrame) -> list[OhlcvBar]:
    """Convert DataFrame -> bars.

    Accepts either:
      - UTC datetime index (recommended), OR
      - column 'ts_ms'
    """
    if df.empty:
        return []

    if "ts_ms" in df.columns:
        ts_ms = df["ts_ms"].astype("int64").tolist()
    else:
        if df.index.tz is None:
            # Assume UTC if naive; better than silently local time.
            idx = df.index.tz_localize("UTC")
        else:
            idx = df.index.tz_convert("UTC")
        ts_ms = (idx.view("int64") // 1_000_000).astype("int64").tolist()

    required = ["open", "high", "low", "close", "volume"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"DataFrame missing required column: {c}")

    out: list[OhlcvBar] = []
    for i in range(len(df)):
        out.append(
            OhlcvBar(
                ts_ms=int(ts_ms[i]),
                open=float(df["open"].iloc[i]),
                high=float(df["high"].iloc[i]),
                low=float(df["low"].iloc[i]),
                close=float(df["close"].iloc[i]),
                volume=float(df["volume"].iloc[i]),
            )
        )
    out.sort(key=lambda b: b.ts_ms)
    return out


def validate_bars(
    bars: Sequence[OhlcvBar],
    *,
    interval: str | None = None,
    allow_gaps: bool = True,
) -> None:
    """Validate monotonicity/duplicates; optionally validate spacing by interval."""
    if not bars:
        return

    # Monotonic + no duplicates
    ts = [b.ts_ms for b in bars]
    for i in range(1, len(ts)):
        if ts[i] <= ts[i - 1]:
            raise ValueError(
                f"Bars must be strictly increasing by ts_ms. "
                f"Found ts[{i-1}]={ts[i-1]} ts[{i}]={ts[i]}"
            )

    if interval is None:
        return

    step = interval_to_ms(interval)
    if allow_gaps:
        return

    # Strict: every step must match exactly
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i - 1]
        if dt != step:
            raise ValueError(
                f"Gap detected (strict). interval={interval} step_ms={step} "
                f"at i={i} prev={ts[i-1]} cur={ts[i]} dt={dt}"
            )

