"""Canonical command implementations for CLI operations."""

from __future__ import annotations

__all__ = [
    "run_train",
    "run_backtest",
    "run_live",
    "run_dataset_fetch",
    "run_dataset_build",
    "run_baselines",
]

# Direct imports - these modules don't import cli, so no circular dependency
from .backtest import run_backtest
from .baselines import run_baselines
from .dataset import run_dataset_build, run_dataset_fetch
from .live import run_live
from .train import run_train
