#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple

import ccxt

from autonomous_rl_trading_bot.common.config import load_config
from autonomous_rl_trading_bot.common.db import connect, migrate
from autonomous_rl_trading_bot.common.logging import configure_logging
from autonomous_rl_trading_bot.common.paths import artifacts_dir, ensure_artifact_tree


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _interval_to_ms(interval: str) -> int:
    s = interval.strip().lower()
    if s.endswith("m"):
        return int(s[:-1]) * 60_000
    if s.endswith("h"):
        return int(s[:-1]) * 3_600_000
    if s.endswith("d"):
        return int(s[:-1]) * 86_400_000
    raise ValueError(f"Unsupported interval: {interval}")


def _market_type_from_mode(mode: str) -> str:
    m = (mode or "").strip().lower()
    return "futures" if m == "futures" else "spot"


def _normalize_symbol(symbol: str) -> str:
    # Accept BTC/USDT, BTCUSDT, etc.
    s = (symbol or "").strip().upper()
    if "/" in s:
        return s.replace("/", "")
    return s


def _ensure_candles_schema(conn) -> None:
    """
    Your DB already has the Binance-style schema:
      open_time_ms, close_time_ms, open, high, low, close, volume, ...
    This function ensures:
      - candles table exists
      - unique index exists
    It does NOT require a 'timestamp' column (because your schema doesn't have it).
    """
    cur = conn.cursor()

    # Ensure table exists with the schema that matches what you already have.
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
          ignore REAL
        )
        """
    )

    # Unique index for upserts
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_candles_unique
        ON candles(exchange, market_type, symbol, interval, open_time_ms)
        """
    )

    conn.commit()


def _upsert_candles(
    conn,
    exchange: str,
    market_type: str,
    symbol: str,
    interval: str,
    rows: Iterable[Tuple[int, float, float, float, float, float]],
) -> int:
    """
    rows: (open_time_ms, open, high, low, close, volume)
    """
    cur = conn.cursor()
    interval_ms = _interval_to_ms(interval)

    data = []
    for (ts, o, h, l, c, v) in rows:
        open_time_ms = int(ts)
        close_time_ms = open_time_ms + interval_ms - 1

        data.append(
            (
                exchange,
                market_type,
                symbol,
                interval,
                open_time_ms,
                float(o),
                float(h),
                float(l),
                float(c),
                float(v),
                close_time_ms,
                None,   # quote_asset_volume
                None,   # number_of_trades
                None,   # taker_buy_base_asset_volume
                None,   # taker_buy_quote_asset_volume
                None,   # ignore
            )
        )

    cur.executemany(
        """
        INSERT INTO candles (
          exchange, market_type, symbol, interval,
          open_time_ms, open, high, low, close, volume,
          close_time_ms, quote_asset_volume, number_of_trades,
          taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(exchange, market_type, symbol, interval, open_time_ms)
        DO UPDATE SET
          open=excluded.open,
          high=excluded.high,
          low=excluded.low,
          close=excluded.close,
          volume=excluded.volume,
          close_time_ms=excluded.close_time_ms
        """,
        data,
    )

    conn.commit()
    return len(data)


def _make_exchange(exchange_id: str, market_type: str):
    ex_id = (exchange_id or "binance").strip().lower()
    klass = getattr(ccxt, ex_id, None)
    if klass is None:
        raise ValueError(f"Unsupported exchange: {exchange_id}")

    ex = klass({"enableRateLimit": True})

    # futures on binance is a different ccxt class
    if ex_id == "binance" and market_type == "futures":
        ex = ccxt.binanceusdm({"enableRateLimit": True})

    return ex


@dataclass
class FetchArgs:
    mode: Optional[str]
    exchange: str
    symbol: str
    interval: str
    minutes: int


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch market candles into SQLite DB.")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--minutes", type=int, default=600)

    args_ns = parser.parse_args(argv)
    args = FetchArgs(
        mode=args_ns.mode,
        exchange=args_ns.exchange,
        symbol=args_ns.symbol,
        interval=args_ns.interval,
        minutes=int(args_ns.minutes),
    )

    ensure_artifact_tree()

    loaded = load_config(mode=args.mode)
    cfg = loaded.config

    mode = cfg["mode"]["id"]
    market_type = _market_type_from_mode(mode)
    symbol = _normalize_symbol(args.symbol)
    interval = args.interval.strip()

    if args.minutes <= 0:
        raise SystemExit("--minutes must be > 0")

    db_path = migrate(cfg)

    run_dir = artifacts_dir() / "runs" / f"fetch_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(
        level="INFO",
        console=True,
        file_paths=[run_dir / "run.log"],
        run_id="arbt",
    )

    logger.info("Fetch start | utc=%s | exchange=%s | market_type=%s | symbol=%s | interval=%s | minutes=%d",
                _utc_now_iso(), args.exchange, market_type, symbol, interval, args.minutes)

    with connect(db_path) as conn:
        _ensure_candles_schema(conn)

    ex = _make_exchange(args.exchange, market_type)

    # We want N minutes of 1m => N candles
    # For higher intervals, minutes still means "how many minutes back" window, not candles count.
    # Keep it simple: target candle count = minutes / interval_minutes (ceil).
    interval_ms = _interval_to_ms(interval)
    target_ms = args.minutes * 60_000
    target_candles = max(1, (target_ms + interval_ms - 1) // interval_ms)

    # Binance limits: usually 1000 per call, safe default.
    limit = int(min(1000, target_candles))

    now_ms = int(time.time() * 1000)
    since_ms = now_ms - int(target_candles * interval_ms)

    logger.info("Requesting candles | since_ms=%d | limit=%d", since_ms, limit)

    ohlcv = ex.fetch_ohlcv(
        symbol=f"{symbol[:-4]}/{symbol[-4:]}" if symbol.endswith("USDT") else symbol,
        timeframe=interval,
        since=since_ms,
        limit=limit,
    )

    if not ohlcv:
        raise SystemExit("No candles returned from exchange API.")

    # ccxt: [timestamp, open, high, low, close, volume]
    rows = [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in ohlcv]

    with connect(db_path) as conn:
        inserted = _upsert_candles(
            conn,
            exchange=args.exchange,
            market_type=market_type,
            symbol=symbol,
            interval=interval,
            rows=rows,
        )

    logger.info("Fetch done | inserted=%d | db=%s", inserted, db_path)
    print(f"OK: fetched {inserted} candles into {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
