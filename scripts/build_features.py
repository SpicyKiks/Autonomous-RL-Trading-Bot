#!/usr/bin/env python3
"""
Feature build + validation run.

Example:
  python scripts/build_features.py --symbol BTCUSDT --interval 1m
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from autonomous_rl_trading_bot.data.feature_engineering import compute_features


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    stem = f"{args.symbol.upper()}_{args.interval}"
    pq_path = os.path.join(args.raw_dir, f"{stem}.parquet")
    csv_path = os.path.join(args.raw_dir, f"{stem}.csv")

    if os.path.exists(pq_path):
        df = pd.read_parquet(pq_path)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Missing raw data. Expected {pq_path} or {csv_path}")

    feats = compute_features(df)

    must_have = ["atr_14", "rsi_14", "ema_12", "ema_26", "macd", "volatility_30", "volume_delta"]
    missing = [c for c in must_have if c not in feats.columns]
    if missing:
        raise RuntimeError(f"Missing computed features: {missing}")

    print("[OK] Feature columns present:", ", ".join(must_have))
    print(feats[["datetime_utc", "close"] + must_have].tail(5).to_string(index=False))

    os.makedirs(args.out_dir, exist_ok=True)
    out_pq = os.path.join(args.out_dir, f"{stem}_features.parquet")
    feats.to_parquet(out_pq, index=False)
    print(f"[OK] Saved features parquet: {out_pq}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
