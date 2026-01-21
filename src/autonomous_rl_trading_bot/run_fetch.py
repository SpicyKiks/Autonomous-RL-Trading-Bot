from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path
from typing import Any

import ccxt  # type: ignore

from autonomous_rl_trading_bot.common.config import load_config
from autonomous_rl_trading_bot.common.db import migrate
from autonomous_rl_trading_bot.common.logging import get_logger

log = get_logger("arbt")


# Binance max OHLCV per call is typically 1000
EXCHANGE_LIMIT_DEFAULT = 1000


def _interval_to_ms(interval: str) -> int:
    # supports "1m", "5m", "1h", "1d"
    unit = interval[-1]
    n = int(interval[:-1])
    if unit == "m":
        return n * 60_000
    if unit == "h":
        return n * 3_600_000
    if unit == "d":
        return n * 86_400_000
    raise ValueError(f"Unsupported interval: {interval}")


def _ensure_candles_schema(conn: sqlite3.Connection) -> None:
    """
    Ensure `candles` exists with the schema your project already uses
    (open_time_ms/close_time_ms, not `timestamp`).
    """
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS candles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exchange TEXT NOT NULL,
            market_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            open_time_ms INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            close_time_ms INTEGER,
            quote_asset_volume REAL,
            number_of_trades INTEGER,
            taker_buy_base_asset_volume REAL,
            taker_buy_quote_asset_volume REAL,
            ignore REAL,
            UNIQUE(exchange, market_type, symbol, interval, open_time_ms)
        )
        """
    )
    conn.commit()


def _insert_candles(
    conn: sqlite3.Connection,
    exchange: str,
    market_type: str,
    symbol: str,
    interval: str,
    rows: list[list[Any]],
) -> int:
    """
    rows are ccxt OHLCV: [timestamp_ms, open, high, low, close, volume]
    We map to your DB schema and UPSERT (ignore duplicates).
    """
    cur = conn.cursor()
    inserted = 0

    to_insert = []
    for r in rows:
        open_time_ms = int(r[0])
        o, h, l, c, v = float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])
        close_time_ms = open_time_ms + _interval_to_ms(interval) - 1
        # Optional binance extras are unknown from ccxt OHLCV, store NULL
        to_insert.append(
            (
                exchange,
                market_type,
                symbol,
                interval,
                open_time_ms,
                o,
                h,
                l,
                c,
                v,
                close_time_ms,
                None,
                None,
                None,
                None,
                None,
            )
        )

    cur.executemany(
        """
        INSERT OR IGNORE INTO candles (
            exchange, market_type, symbol, interval,
            open_time_ms, open, high, low, close, volume,
            close_time_ms,
            quote_asset_volume, number_of_trades,
            taker_buy_base_asset_volume, taker_buy_quote_asset_volume,
            ignore
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        to_insert,
    )
    inserted = cur.rowcount if cur.rowcount is not None else 0
    conn.commit()
    return inserted


def _count_available(conn: sqlite3.Connection, exchange: str, market_type: str, symbol: str, interval: str) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(*) FROM candles
        WHERE exchange=? AND market_type=? AND symbol=? AND interval=?
        """,
        (exchange, market_type, symbol, interval),
    )
    return int(cur.fetchone()[0])


def _minmax_open_time(conn: sqlite3.Connection, exchange: str, market_type: str, symbol: str, interval: str) -> tuple[int | None, int | None]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT MIN(open_time_ms), MAX(open_time_ms) FROM candles
        WHERE exchange=? AND market_type=? AND symbol=? AND interval=?
        """,
        (exchange, market_type, symbol, interval),
    )
    a, b = cur.fetchone()
    return (int(a) if a is not None else None, int(b) if b is not None else None)


def _make_exchange(exchange_id: str) -> Any:
    ex_cls = getattr(ccxt, exchange_id)
    ex = ex_cls({"enableRateLimit": True})
    return ex


def fetch_candles_paged(
    exchange_id: str,
    market_type: str,
    symbol: str,
    interval: str,
    minutes: int,
    limit: int = EXCHANGE_LIMIT_DEFAULT,
) -> list[list[Any]]:
    """
    Fetch candles from (now - minutes*interval) forward to now, in pages of `limit`.
    Returns all rows in chronological order (as best effort).
    """
    ex = _make_exchange(exchange_id)

    now_ms = ex.milliseconds()
    interval_ms = _interval_to_ms(interval)
    total_ms = minutes * interval_ms
    start_ms = now_ms - total_ms

    # Fetch from start_ms forward, paging by last_open_time + interval_ms
    since_ms = start_ms
    all_rows: list[list[Any]] = []
    safety = 0

    while since_ms < now_ms:
        safety += 1
        if safety > 50_000:
            # something is wrong (no progression)
            break

        log.info(f"Requesting candles | since_ms={since_ms} | limit={limit}")
        rows = ex.fetch_ohlcv(symbol, timeframe=interval, since=since_ms, limit=limit)

        if not rows:
            break

        # Ensure sorted
        rows = sorted(rows, key=lambda r: r[0])

        # Keep only within window
        rows = [r for r in rows if start_ms <= int(r[0]) <= now_ms]

        if rows:
            all_rows.extend(rows)

        last_ts = int(rows[-1][0]) if rows else int(rows[-1][0])  # type: ignore
        next_since = last_ts + interval_ms

        if next_since <= since_ms:
            # no forward progress; stop
            break

        since_ms = next_since

        # avoid rate-limit hammering
        time.sleep(0.2)

    # Deduplicate by timestamp (ccxt can overlap pages)
    seen = set()
    deduped = []
    for r in sorted(all_rows, key=lambda x: x[0]):
        ts = int(r[0])
        if ts in seen:
            continue
        seen.add(ts)
        deduped.append(r)

    return deduped


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arbt dataset fetch")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval", required=True)
    p.add_argument("--minutes", type=int, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    loaded = load_config()
    cfg = loaded.config

    exchange_id = cfg.get("exchange", {}).get("id", "binance")
    market_type = cfg.get("mode", {}).get("id", "spot")

    db_path = Path(migrate(cfg))
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        _ensure_candles_schema(conn)

        log.info(
            f"Fetch start | exchange={exchange_id} | market_type={market_type} | symbol={args.symbol} | interval={args.interval} | minutes={args.minutes}"
        )

        rows = fetch_candles_paged(
            exchange_id=exchange_id,
            market_type=market_type,
            symbol=args.symbol,
            interval=args.interval,
            minutes=args.minutes,
            limit=EXCHANGE_LIMIT_DEFAULT,
        )

        inserted = _insert_candles(
            conn,
            exchange=exchange_id,
            market_type=market_type,
            symbol=args.symbol,
            interval=args.interval,
            rows=rows,
        )

        mn, mx = _minmax_open_time(conn, exchange_id, market_type, args.symbol, args.interval)

        log.info(f"Fetch done | inserted={inserted} | db={db_path}")
        log.info(f"DB range | min_open_time_ms={mn} | max_open_time_ms={mx}")

        print(f"OK: fetched {len(rows)} candles (inserted {inserted}) into {db_path}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
