from __future__ import annotations

from dash import Dash

from autonomous_rl_trading_bot.common.paths import ensure_artifact_tree
from autonomous_rl_trading_bot.version import __version__

from .callbacks import register_callbacks
from .data_api import load_artifact_index
from .layout import build_layout


def create_dash_app() -> Dash:
    """Create a minimal Dash app (offline).

    This is deliberately safe: it never calls exchanges or networks.
    It only reads local artifacts/ (datasets/runs).
    """
    ensure_artifact_tree()
    idx = load_artifact_index()

    app = Dash(
        __name__,
        title=f"Autonomous RL Trading Bot v{__version__}",
        suppress_callback_exceptions=True,
    )
    app.layout = build_layout(idx)
    register_callbacks(app)
    return app

