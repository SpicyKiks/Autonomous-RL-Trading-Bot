"""Canonical live trading command implementation."""

from __future__ import annotations

import sys
from pathlib import Path


def run_live(argv: list[str] | None = None) -> int:
    """
    Canonical live trading command.
    
    This is a wrapper around the existing run_live_demo.py implementation.
    """
    # Add project root to path to import run_live_demo
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import run_live_demo
    return int(run_live_demo.main(argv))
