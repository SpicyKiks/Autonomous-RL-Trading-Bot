"""Pytest bootstrap.

This repo uses a src/ layout but doesn't require installing the package for local dev.
We prepend `<repo>/src` to sys.path so `import autonomous_rl_trading_bot` works.
"""

from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    repo = Path(__file__).resolve().parents[1]
    src = repo / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
