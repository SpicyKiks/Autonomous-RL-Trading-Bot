#!/usr/bin/env python3
"""
Download OHLCV from Binance Futures via CCXT and save as CSV + Parquet.

Example:
  python scripts/download_binance_futures.py --symbol BTCUSDT --interval 1m --days 30
Outputs:
  data/raw/BTCUSDT_1m.csv
  data/raw/BTCUSDT_1m.parquet
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import UTC, datetime, timedelta

import ccxt
import pandas as pd


def _normalize_symbol(symbol: str, exchange_name: str = "binanceusdm") -> str:
    s = symbol.strip().upper()
    # For binanceusdm (USD-M futures), use format without slash: BTCUSDT
    if exchange_name.lower() == "binanceusdm":
        if "/" in s:
            return s.replace("/", "")
        return s
    # For regular binance, use format with slash: BTC/USDT
    if "/" in s:
        return s
    if s.endswith("USDT") and len(s) > 4:
        base = s[:-4]
        return f"{base}/USDT"
    return s


def _exchange(exchange_name: str) -> ccxt.Exchange:
    name = exchange_name.strip().lower()
    if name == "binanceusdm":
        return ccxt.binanceusdm({"enableRateLimit": True})
    if name == "binance":
        return ccxt.binance(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
        )
    raise ValueError(f"Unsupported exchange_name={exchange_name}. Use binanceusdm or binance.")


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def download_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1500,
) -> pd.DataFrame:
    all_rows: list[list[float]] = []
    since = start_ms

    exchange.load_markets()

    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break

        for row in ohlcv:
            ts = int(row[0])
            if ts < start_ms:
                continue
            if ts > end_ms:
                break
            all_rows.append(row)

        last_ts = int(ohlcv[-1][0])
        next_since = last_ts + 1

        if next_since <= since or last_ts >= end_ms:
            break

        since = next_since
        time.sleep(max(exchange.rateLimit / 1000.0, 0.05))

    if not all_rows:
        raise RuntimeError("No OHLCV rows downloaded. Check symbol/timeframe/date range.")

    df = pd.DataFrame(all_rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms").reset_index(drop=True)
    df["datetime_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, help="e.g. BTCUSDT or BTC/USDT")
    ap.add_argument("--interval", default="1m", help="CCXT timeframe, e.g. 1m, 5m, 1h")
    ap.add_argument("--days", type=int, default=30, help="How many days back from now (UTC)")
    ap.add_argument("--exchange", default="binanceusdm", help="binanceusdm (USD-M futures) or binance")
    ap.add_argument("--out-dir", default="data/raw", help="Output directory")
    args = ap.parse_args()

    symbol_in = args.symbol.strip()
    symbol_ccxt = _normalize_symbol(symbol_in, args.exchange)

    now = datetime.now(UTC)
    start = now - timedelta(days=args.days)
    start_ms = _to_ms(start)
    end_ms = _to_ms(now)

    ex = _exchange(args.exchange)

    df = download_ohlcv(ex, symbol_ccxt, args.interval, start_ms, end_ms)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"{symbol_in.upper()}_{args.interval}"
    csv_path = os.path.join(args.out_dir, f"{stem}.csv")
    pq_path = os.path.join(args.out_dir, f"{stem}.parquet")

    df.to_csv(csv_path, index=False)
    df.to_parquet(pq_path, index=False)

    print(f"[OK] Downloaded {len(df):,} rows")
    print(f"[OK] CSV    : {csv_path}")
    print(f"[OK] Parquet: {pq_path}")
    print(f"[UTC] Range : {df['datetime_utc'].iloc[0]} -> {df['datetime_utc'].iloc[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
