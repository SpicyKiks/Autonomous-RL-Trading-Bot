"""Canonical baselines command implementation."""

from __future__ import annotations

# Import existing implementation
from autonomous_rl_trading_bot.evaluation.baselines import main as _baselines_main


def run_baselines(argv: list[str] | None = None) -> int:
    """
    Canonical baselines command.
    
    This wraps the existing evaluation/baselines.py implementation.
    """
    return _baselines_main(argv)
