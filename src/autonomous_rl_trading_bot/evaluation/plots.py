from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

# Force non-interactive backend for headless/CI environments (prevents Tk errors)
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _extract_equity_data(
    equity_data: list[dict[str, Any]] | pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract equity values and timestamps from various input formats.
    """
    if isinstance(equity_data, pd.DataFrame):
        # Find equity column
        equity_col = None
        for col in ["equity", "equity_value", "value"]:
            if col in equity_data.columns:
                equity_col = col
                break

        # Find timestamp column
        ts_col = None
        for col in ["open_time_ms", "timestamp", "time_ms", "ts"]:
            if col in equity_data.columns:
                ts_col = col
                break

        if equity_col is None:
            raise ValueError("Could not find equity column in DataFrame")

        equity_values = equity_data[equity_col].to_numpy(dtype=float)

        if ts_col is None:
            timestamps = np.arange(len(equity_data))
        else:
            timestamps = equity_data[ts_col].to_numpy(dtype=float)

        return timestamps, equity_values

    # List of dicts
    if not equity_data:
        return np.array([]), np.array([])

    # Find columns
    equity_keys = ["equity", "equity_value", "value"]
    ts_keys = ["open_time_ms", "timestamp", "time_ms", "ts", "t", "step"]

    first = equity_data[0]
    equity_key = next((k for k in equity_keys if k in first), None)
    ts_key = next((k for k in ts_keys if k in first), None)

    if equity_key is None:
        raise ValueError("Could not find equity key in equity rows")

    eq = np.array([float(r.get(equity_key, 0.0)) for r in equity_data], dtype=float)

    if ts_key is None:
        ts = np.arange(len(equity_data), dtype=float)
    else:
        ts = np.array([float(r.get(ts_key, i)) for i, r in enumerate(equity_data)], dtype=float)

    return ts, eq


def _compute_drawdown(equity: np.ndarray) -> np.ndarray:
    if equity.size == 0:
        return equity
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.maximum(peak, 1e-12)
    return dd


def plot_equity_curve(
    equity_data: list[dict[str, Any]] | pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Equity Curve",
) -> Path:
    ts, eq = _extract_equity_data(equity_data)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts, eq)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_drawdown(
    equity_data: list[dict[str, Any]] | pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Drawdown",
) -> Path:
    ts, eq = _extract_equity_data(equity_data)
    dd = _compute_drawdown(eq)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts, dd)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_trades_pnl_histogram(
    trades: list[dict[str, Any]] | pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Trade PnL Histogram",
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(trades, pd.DataFrame):
        rows = trades.to_dict(orient="records")
    else:
        rows = list(trades)

    pnl = []
    for r in rows:
        if "realized_pnl" in r:
            try:
                pnl.append(float(r["realized_pnl"]))
            except Exception:
                pass
        elif "pnl" in r:
            try:
                pnl.append(float(r["pnl"]))
            except Exception:
                pass

    fig, ax = plt.subplots(figsize=(12, 6))
    if pnl:
        ax.hist(pnl, bins=30)
    ax.set_title(title)
    ax.set_xlabel("PnL")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_feature_importance_bar(
    feature_importance: Mapping[str, float],
    out_path: Path,
    *,
    title: str = "Feature Importance",
    top_k: int = 20,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    names = [k for k, _ in items]
    vals = [float(v) for _, v in items]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(list(reversed(names)), list(reversed(vals)))
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
