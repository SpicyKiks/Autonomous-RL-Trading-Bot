from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Strategy interface (THIS is what backtester.py is trying to import)
# ------------------------------------------------------------

class Strategy:
    """
    Minimal baseline strategy interface.

    backtester.py imports Strategy from here.
    We keep it intentionally simple and stable:
      - reset() optional
      - generate_positions(df) returns a pd.Series of positions in {-1,0,1}
    """

    name: str = "strategy"

    def reset(self) -> None:
        pass

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        """
        Given OHLCV dataframe, return a position series:
          +1 = long, 0 = flat, -1 = short
        Must have same length as df and aligned with df.index.
        """
        raise NotImplementedError


# ------------------------------------------------------------
# Baseline strategies
# ------------------------------------------------------------

@dataclass
class BuyAndHold(Strategy):
    name: str = "buy_and_hold"
    allow_short: bool = False

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        pos = pd.Series(1, index=df.index, dtype=np.int8)
        if not self.allow_short:
            # always long
            return pos
        return pos


@dataclass
class SMACrossover(Strategy):
    name: str = "sma_crossover"
    fast: int = 20
    slow: int = 100
    allow_short: bool = True

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        close = _require_col(df, "close").astype(float)

        fast_ma = close.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = close.rolling(self.slow, min_periods=self.slow).mean()

        # signal: fast above slow => long, below => short/flat
        long_mask = fast_ma > slow_ma
        short_mask = fast_ma < slow_ma

        pos = pd.Series(0, index=df.index, dtype=np.int8)
        pos[long_mask] = 1
        if self.allow_short:
            pos[short_mask] = -1
        else:
            pos[short_mask] = 0

        # avoid NaN warmup leading to bogus positions
        pos = pos.fillna(0).astype(np.int8)
        return pos


@dataclass
class RSIReversion(Strategy):
    name: str = "rsi_reversion"
    period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    allow_short: bool = True

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        close = _require_col(df, "close").astype(float)
        rsi = _rsi(close, period=self.period)

        pos = pd.Series(0, index=df.index, dtype=np.int8)
        # mean reversion idea:
        # oversold -> long
        # overbought -> short/flat
        pos[rsi < self.oversold] = 1
        if self.allow_short:
            pos[rsi > self.overbought] = -1
        else:
            pos[rsi > self.overbought] = 0

        pos = pos.fillna(0).astype(np.int8)
        return pos


# ------------------------------------------------------------
# CLI runner (so `arbt baselines` actually works)
# ------------------------------------------------------------

def run_baselines(
    dataset_id: Optional[str],
    mode: str,
    out_csv: Optional[str],
    strategies: List[Strategy],
) -> int:
    """
    Loads dataset (via rl.dataset.Dataset) and outputs a CSV of positions for each baseline.
    This is intentionally lightweight and independent from the RL trainer.

    If you want it to run backtests, integrate with evaluation.backtester.py later.
    """
    from autonomous_rl_trading_bot.rl.dataset import Dataset  # local import to avoid circulars

    if dataset_id is None:
        # try to use "latest" dataset by scanning artifacts/datasets
        dataset_id = _find_latest_dataset_id(mode=mode)

    ds = Dataset.load(dataset_id, market_type=mode, split="full")
    df = ds.to_dataframe()

    # normalize expected cols
    if "close" not in df.columns and "Close" in df.columns:
        df = df.rename(columns={"Close": "close"})

    out: Dict[str, pd.Series] = {}
    for strat in strategies:
        strat.reset()
        out[strat.name] = strat.generate_positions(df)

    out_df = pd.DataFrame(out, index=df.index)

    if out_csv:
        out_df.to_csv(out_csv, index=True)
        print(f"OK: wrote baseline positions to {out_csv}")
    else:
        # print a compact preview
        print(out_df.tail(20).to_string())

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="arbt baselines", description="Run baseline strategies on a dataset.")
    p.add_argument("--mode", choices=["spot", "futures"], default="spot", help="market mode (default: spot)")
    p.add_argument("--dataset-id", default=None, help="dataset id under artifacts/datasets (default: latest)")
    p.add_argument("--out", default=None, help="optional CSV output path")

    # choose strategies
    p.add_argument("--buyhold", action="store_true", help="include buy-and-hold")
    p.add_argument("--sma", action="store_true", help="include SMA crossover")
    p.add_argument("--rsi", action="store_true", help="include RSI reversion")
    p.add_argument("--fast", type=int, default=20, help="SMA fast window")
    p.add_argument("--slow", type=int, default=100, help="SMA slow window")
    p.add_argument("--no-short", action="store_true", help="disable shorts (flat instead)")

    args = p.parse_args(argv)

    allow_short = not args.no_short

    # default: run all if none specified
    selected_any = args.buyhold or args.sma or args.rsi
    strategies: List[Strategy] = []

    if args.buyhold or not selected_any:
        strategies.append(BuyAndHold(allow_short=allow_short))
    if args.sma or not selected_any:
        strategies.append(SMACrossover(fast=args.fast, slow=args.slow, allow_short=allow_short))
    if args.rsi or not selected_any:
        strategies.append(RSIReversion(allow_short=allow_short))

    return run_baselines(
        dataset_id=args.dataset_id,
        mode=args.mode,
        out_csv=args.out,
        strategies=strategies,
    )


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _require_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Expected column '{col}' in dataframe. Available: {list(df.columns)}")
    return df[col]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    rs = avg_gain / (avg_loss.replace(0.0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _find_latest_dataset_id(mode: str) -> str:
    """
    Find latest dataset folder under:
      <repo>/artifacts/datasets
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]  # .../src/autonomous_rl_trading_bot/evaluation/baselines.py -> repo root
    ds_dir = root / "artifacts" / "datasets"

    if not ds_dir.exists():
        raise FileNotFoundError(f"Could not find datasets dir: {ds_dir}")

    # filter by mode substring if present in folder name
    candidates = [p for p in ds_dir.iterdir() if p.is_dir() and f"_{mode}_" in p.name]
    if not candidates:
        # fall back to any dataset
        candidates = [p for p in ds_dir.iterdir() if p.is_dir()]

    if not candidates:
        raise FileNotFoundError(f"No datasets found under {ds_dir}")

    # latest by name (timestamp prefix)
    latest = sorted(candidates, key=lambda p: p.name)[-1]
    return latest.name
def make_strategy(name: str, **kwargs) -> Strategy:
    """
    Factory expected by evaluation/__init__.py and possibly backtester.
    """
    key = (name or "").strip().lower()

    if key in {"buy_and_hold", "buyhold", "hold"}:
        allow_short = bool(kwargs.get("allow_short", True))
        return BuyAndHold(allow_short=allow_short)

    if key in {"sma_crossover", "sma", "ma_crossover"}:
        fast = int(kwargs.get("fast", 20))
        slow = int(kwargs.get("slow", 100))
        allow_short = bool(kwargs.get("allow_short", True))
        return SMACrossover(fast=fast, slow=slow, allow_short=allow_short)

    if key in {"rsi_reversion", "rsi"}:
        period = int(kwargs.get("period", 14))
        oversold = float(kwargs.get("oversold", 30.0))
        overbought = float(kwargs.get("overbought", 70.0))
        allow_short = bool(kwargs.get("allow_short", True))
        return RSIReversion(
            period=period,
            oversold=oversold,
            overbought=overbought,
            allow_short=allow_short,
        )

    raise ValueError(
        f"Unknown strategy '{name}'. Supported: buy_and_hold, sma_crossover, rsi_reversion"
    )


__all__ = [
    "Strategy",
    "make_strategy",
    "BuyAndHold",
    "SMACrossover",
    "RSIReversion",
    "run_baselines",
    "main",
]
