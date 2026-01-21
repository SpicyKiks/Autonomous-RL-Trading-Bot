"""Canonical training command implementation."""

from __future__ import annotations

# Import existing implementation - we'll consolidate later
from autonomous_rl_trading_bot.run_train import main as _legacy_main


def run_train(argv: list[str] | None = None) -> int:
    """
    Canonical training command.
    
    This is a wrapper around the existing run_train.py implementation.
    Future consolidation will merge logic here.
    """
    return _legacy_main(argv)
