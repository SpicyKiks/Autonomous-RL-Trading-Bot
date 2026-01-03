"""Dash dashboard for viewing artifacts (datasets, runs, reports).

This starts as a minimal, safe, offline UI. Later steps will add:
- equity curve / drawdown charts
- trade list / fills
- live mode monitoring (paper/live)
"""

from .app import create_dash_app

__all__ = ["create_dash_app"]

