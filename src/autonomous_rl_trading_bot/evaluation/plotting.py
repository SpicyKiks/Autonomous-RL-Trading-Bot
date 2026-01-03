from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_equity_and_drawdown(
    equity: List[float],
    drawdown: List[float],
    open_time_ms: List[int],
    output_path: Path,
    title: str = "Equity & Drawdown",
) -> None:
    """
    Plot equity curve and drawdown on dual y-axes.
    
    Args:
        equity: Equity values over time
        drawdown: Drawdown values over time (0-1)
        open_time_ms: Timestamps in milliseconds
        output_path: Path to save the plot
        title: Plot title
    """
    if not equity or len(equity) != len(drawdown) or len(equity) != len(open_time_ms):
        raise ValueError("All inputs must have same length")
    
    # Convert timestamps to days since start for x-axis
    start_ms = open_time_ms[0]
    days_since_start = [(ms - start_ms) / (24 * 60 * 60 * 1000) for ms in open_time_ms]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot equity on left y-axis
    color1 = "tab:blue"
    ax1.set_xlabel("Days Since Start")
    ax1.set_ylabel("Equity", color=color1)
    ax1.plot(days_since_start, equity, color=color1, linewidth=1.5, label="Equity")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Plot drawdown on right y-axis
    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Drawdown", color=color2)
    ax2.plot(days_since_start, drawdown, color=color2, linewidth=1.5, label="Drawdown", alpha=0.7)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim([0, max(1.0, max(drawdown) * 1.1)])
    
    plt.title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_trades_over_price(
    prices: List[float],
    open_time_ms: List[int],
    trades: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Trades Over Price",
) -> None:
    """
    Plot price chart with buy/sell markers.
    
    Args:
        prices: Price values over time
        open_time_ms: Timestamps in milliseconds
        trades: List of trade dicts with 'open_time_ms', 'side', 'price'
        output_path: Path to save the plot
        title: Plot title
    """
    if not prices or len(prices) != len(open_time_ms):
        raise ValueError("prices and open_time_ms must have same length")
    
    # Convert timestamps to days since start
    start_ms = open_time_ms[0]
    days_since_start = [(ms - start_ms) / (24 * 60 * 60 * 1000) for ms in open_time_ms]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot price
    ax.plot(days_since_start, prices, color="black", linewidth=1.0, alpha=0.7, label="Price")
    
    # Plot trades
    buy_times = []
    buy_prices = []
    sell_times = []
    sell_prices = []
    
    for trade in trades:
        trade_ms = int(trade.get("open_time_ms", 0))
        trade_price = float(trade.get("price", 0.0))
        side = str(trade.get("side", "")).upper()
        
        if trade_ms >= start_ms:
            days = (trade_ms - start_ms) / (24 * 60 * 60 * 1000)
            if side in ("BUY", "LONG"):
                buy_times.append(days)
                buy_prices.append(trade_price)
            elif side in ("SELL", "SHORT", "CLOSE", "FLAT"):
                sell_times.append(days)
                sell_prices.append(trade_price)
    
    if buy_times:
        ax.scatter(buy_times, buy_prices, color="green", marker="^", s=100, label="Buy", zorder=5, alpha=0.7)
    if sell_times:
        ax.scatter(sell_times, sell_prices, color="red", marker="v", s=100, label="Sell", zorder=5, alpha=0.7)
    
    ax.set_xlabel("Days Since Start")
    ax.set_ylabel("Price")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_price_with_trades(
    prices: List[float],
    open_time_ms: List[int],
    trades: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Price with Trade Markers",
    use_candlestick: bool = False,
    open_prices: Optional[List[float]] = None,
    high_prices: Optional[List[float]] = None,
    low_prices: Optional[List[float]] = None,
) -> None:
    """
    Plot price chart (candlestick or line) with buy/sell markers.
    
    Args:
        prices: Close prices (or single price series if not candlestick)
        open_time_ms: Timestamps in milliseconds
        trades: List of trade dicts with 'open_time_ms', 'side', 'price'
        output_path: Path to save the plot
        title: Plot title
        use_candlestick: If True, plot candlesticks (requires open/high/low)
        open_prices: Open prices for candlestick (required if use_candlestick=True)
        high_prices: High prices for candlestick (required if use_candlestick=True)
        low_prices: Low prices for candlestick (required if use_candlestick=True)
    """
    if not prices or len(prices) != len(open_time_ms):
        raise ValueError("prices and open_time_ms must have same length")
    
    if use_candlestick:
        if not (open_prices and high_prices and low_prices):
            use_candlestick = False  # Fallback to line plot
    
    # Convert timestamps to days since start
    start_ms = open_time_ms[0]
    days_since_start = [(ms - start_ms) / (24 * 60 * 60 * 1000) for ms in open_time_ms]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    if use_candlestick and open_prices and high_prices and low_prices:
        # Plot candlesticks
        for i, (days, open_p, high_p, low_p, close_p) in enumerate(
            zip(days_since_start, open_prices, high_prices, low_prices, prices)
        ):
            color = "green" if close_p >= open_p else "red"
            # Wick
            ax.plot([days, days], [low_p, high_p], color="black", linewidth=0.5, alpha=0.5)
            # Body
            body_height = abs(close_p - open_p)
            body_bottom = min(open_p, close_p)
            ax.bar(days, body_height, bottom=body_bottom, width=0.01, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
    else:
        # Plot line
        ax.plot(days_since_start, prices, color="black", linewidth=1.0, alpha=0.7, label="Price")
    
    # Plot trades
    buy_times = []
    buy_prices = []
    sell_times = []
    sell_prices = []
    
    for trade in trades:
        trade_ms = int(trade.get("open_time_ms", 0))
        trade_price = float(trade.get("price", 0.0))
        side = str(trade.get("side", "")).upper()
        
        if trade_ms >= start_ms:
            days = (trade_ms - start_ms) / (24 * 60 * 60 * 1000)
            if side in ("BUY", "LONG"):
                buy_times.append(days)
                buy_prices.append(trade_price)
            elif side in ("SELL", "SHORT", "CLOSE", "FLAT"):
                sell_times.append(days)
                sell_prices.append(trade_price)
    
    if buy_times:
        ax.scatter(buy_times, buy_prices, color="green", marker="^", s=150, label="Buy", zorder=5, alpha=0.8, edgecolors="darkgreen", linewidths=1)
    if sell_times:
        ax.scatter(sell_times, sell_prices, color="red", marker="v", s=150, label="Sell", zorder=5, alpha=0.8, edgecolors="darkred", linewidths=1)
    
    ax.set_xlabel("Days Since Start", fontsize=11)
    ax.set_ylabel("Price", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    
    fig.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

