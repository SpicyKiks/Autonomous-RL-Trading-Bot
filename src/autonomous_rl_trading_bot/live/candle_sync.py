from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autonomous_rl_trading_bot.common.timeframes import interval_to_ms
from autonomous_rl_trading_bot.exchange.binance_public import Candle, fetch_klines


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class CandleSyncConfig:
    market_type: str
    symbol: str
    interval: str
    poll_seconds: float = 2.0
    # Binance returns the currently-forming candle; we only trade on CLOSED candles.
    # If `require_settled_ms` > 0, we additionally require now_ms >= close_time_ms + require_settled_ms.
    require_settled_ms: int = 250
    kill_switch_path: str = ""  # Optional kill switch path to check during sleep


class CandleSync:
    """Polls the exchange for the latest CLOSED candle and yields it once per interval."""

    def __init__(self, cfg: dict[str, Any], c: CandleSyncConfig):
        self._cfg = cfg
        self._c = c
        self._interval_ms = interval_to_ms(c.interval)
        self._last_open_time_ms: int | None = None

    @property
    def interval_ms(self) -> int:
        return int(self._interval_ms)

    def _select_latest_closed(self, candles: list[Candle]) -> Candle | None:
        if not candles:
            return None

        now = _now_ms()
        # candles sorted ascending by open_time_ms from fetch_klines
        # prefer last candle if it's already closed
        for cand in reversed(candles[-3:]):
            close_ok = now >= int(cand.close_time_ms) + int(self._c.require_settled_ms)
            if close_ok:
                return cand
        return None

    def poll_latest_closed(self) -> Candle | None:
        """Return the latest CLOSED candle (may be same as last returned)."""
        candles = fetch_klines(
            self._cfg,
            market_type=self._c.market_type,
            symbol=self._c.symbol,
            interval=self._c.interval,
            limit=3,
        )
        return self._select_latest_closed(candles)

    def wait_next(self) -> Candle:
        """Block until a NEW closed candle is available, then return it.
        
        Checks kill switch during sleep to respond to stop requests quickly.
        """
        while True:
            c = self.poll_latest_closed()
            if c is not None:
                ot = int(c.open_time_ms)
                if self._last_open_time_ms is None or ot > self._last_open_time_ms:
                    self._last_open_time_ms = ot
                    return c
            
            # Check kill switch during sleep (break sleep into chunks)
            poll_sec = float(self._c.poll_seconds)
            kill_switch_path = (self._c.kill_switch_path or "").strip()
            if kill_switch_path:
                # Sleep in small chunks and check kill switch
                chunk_sec = min(0.5, poll_sec)  # Check every 0.5s max
                elapsed = 0.0
                while elapsed < poll_sec:
                    time.sleep(chunk_sec)
                    elapsed += chunk_sec
                    # Check kill switch
                    p = os.path.expandvars(kill_switch_path)
                    if Path(p).exists():
                        raise RuntimeError("Kill switch triggered during candle wait")
            else:
                time.sleep(poll_sec)

    def prime(self) -> tuple[Candle, int]:
        """Wait for the most recent closed candle and prime internal cursor.

        Returns (candle, now_ms).
        """
        c = self.wait_next()
        return c, _now_ms()

