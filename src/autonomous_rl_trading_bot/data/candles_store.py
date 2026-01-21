from __future__ import annotations

import csv
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from autonomous_rl_trading_bot.exchange.binance_public import Candle


def insert_candles(conn: sqlite3.Connection, candles: Iterable[Candle]) -> int:
    rows = list(candles)
    if not rows:
        return 0

    conn.executemany(
        """
        INSERT OR IGNORE INTO candles(
          exchange, market_type, symbol, interval, open_time_ms,
          open, high, low, close, volume, close_time_ms,
          quote_asset_volume, number_of_trades,
          taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        [
            (
                c.exchange,
                c.market_type,
                c.symbol,
                c.interval,
                c.open_time_ms,
                c.open,
                c.high,
                c.low,
                c.close,
                c.volume,
                c.close_time_ms,
                c.quote_asset_volume,
                c.number_of_trades,
                c.taker_buy_base_asset_volume,
                c.taker_buy_quote_asset_volume,
                c.ignore,
            )
            for c in rows
        ],
    )
    # SQLite doesn't give exact inserted count for OR IGNORE easily; we return attempted rows
    return len(rows)


def write_candles_csv(path: Path, candles: list[Candle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "exchange",
                "market_type",
                "symbol",
                "interval",
                "open_time_ms",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time_ms",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ]
        )
        for c in candles:
            w.writerow(
                [
                    c.exchange,
                    c.market_type,
                    c.symbol,
                    c.interval,
                    c.open_time_ms,
                    c.open,
                    c.high,
                    c.low,
                    c.close,
                    c.volume,
                    c.close_time_ms,
                    c.quote_asset_volume,
                    c.number_of_trades,
                    c.taker_buy_base_asset_volume,
                    c.taker_buy_quote_asset_volume,
                    c.ignore,
                ]
            )
