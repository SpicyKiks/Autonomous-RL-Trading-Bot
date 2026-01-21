"""Canonical backtest command implementation."""

from __future__ import annotations

from typing import Any

from autonomous_rl_trading_bot.backtest.runner import run_backtest as _run_backtest_canonical


def run_backtest(
    mode: str,
    policy: str = "ppo",
    model_path: str | None = None,
    symbol: str | None = None,
    interval: str | None = None,
    run_id: str | None = None,
    train_split: float = 0.8,
    output_dir: str = "reports",
) -> dict[str, Any]:
    """
    Canonical backtest command.
    
    This wraps the existing backtest/runner.py implementation.
    """
    return _run_backtest_canonical(
        mode=mode,
        policy=policy,
        model_path=model_path,
        symbol=symbol,
        interval=interval,
        run_id=run_id,
        train_split=train_split,
        output_dir=output_dir,
    )
