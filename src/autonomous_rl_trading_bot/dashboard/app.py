from __future__ import annotations

from pathlib import Path
from typing import Optional

from dash import Dash

from .callbacks import register_callbacks
from .data_api import DashboardDataAPI
from .layout import make_layout


def create_app(*, db_path: Path, title: str = "ARBT Dashboard") -> Dash:
    db_path = Path(db_path)

    app = Dash(__name__, title=title)
    api = DashboardDataAPI(db_path=db_path)

    app.layout = make_layout(str(db_path))
    register_callbacks(app, api)

    return app
