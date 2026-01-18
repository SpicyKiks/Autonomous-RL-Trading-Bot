#!/usr/bin/env python3
"""
Build final dataset parquet with:
- causal indicators
- causal normalization
- windowed flattened state vectors

Example:
  python scripts/make_dataset.py --symbol BTCUSDT --interval 1m --window 30

Output:
  data/processed/BTCUSDT_1m_dataset.parquet
"""

from __future__ import annotations

import argparse
import os
import pandas as pd

from autonomous_rl_trading_bot.data.feature_engineering import (
    compute_features,
    causal_zscore,
    build_state_windows,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    stem = f"{args.symbol.upper()}_{args.interval}"
    raw_pq = os.path.join(args.raw_dir, f"{stem}.parquet")
    raw_csv = os.path.join(args.raw_dir, f"{stem}.csv")

    if os.path.exists(raw_pq):
        df = pd.read_parquet(raw_pq)
    elif os.path.exists(raw_csv):
        df = pd.read_csv(raw_csv)
    else:
        raise FileNotFoundError(f"Missing raw data. Expected {raw_pq} or {raw_csv}")

    feats = compute_features(df)

    required = [
        "log_return",
        "atr_14",
        "rsi_14",
        "ema_12",
        "ema_26",
        "macd",
        "macd_signal",
        "macd_hist",
        "volatility_30",
        "volume_delta",
        "volume",
    ]
    feats = feats.dropna(subset=required).reset_index(drop=True)

    to_norm = [
        "log_return",
        "atr_14",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "volatility_30",
        "volume_delta",
        "volume",
    ]
    feats = causal_zscore(feats, to_norm)
    state_cols = [f"{c}_z" for c in to_norm]

    dataset = build_state_windows(feats, state_cols=state_cols, window=args.window)

    os.makedirs(args.out_dir, exist_ok=True)
    out_pq = os.path.join(args.out_dir, f"{stem}_dataset.parquet")
    dataset.to_parquet(out_pq, index=False)

    state_len = len(dataset["state"].iloc[0])
    print(f"[OK] Built dataset rows: {len(dataset):,}")
    print(f"[OK] Window: {args.window}, state_len: {state_len} (window * features = {args.window} * {len(state_cols)})")
    print(f"[OK] Saved: {out_pq}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
