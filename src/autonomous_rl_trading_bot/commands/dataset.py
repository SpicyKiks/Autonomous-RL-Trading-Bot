"""Canonical dataset command implementations."""

from __future__ import annotations

from autonomous_rl_trading_bot.run_dataset import main as _build_main

# Import existing implementations
from autonomous_rl_trading_bot.run_fetch import main as _fetch_main


def run_dataset_fetch(argv: list[str] | None = None) -> int:
    """
    Canonical dataset fetch command.
    
    This wraps the existing run_fetch.py implementation.
    """
    return _fetch_main(argv)


def run_dataset_build(argv: list[str] | None = None) -> int:
    """
    Canonical dataset build command.
    
    This wraps the existing run_dataset.py implementation.
    """
    return _build_main(argv)
