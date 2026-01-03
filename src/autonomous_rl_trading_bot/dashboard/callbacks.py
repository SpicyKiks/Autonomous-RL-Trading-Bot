from __future__ import annotations

from dash import Dash, Input, Output, html


def register_callbacks(app: Dash) -> None:
    @app.callback(
        Output("status", "children"),
        Input("dataset_id", "value"),
        Input("run_id", "value"),
    )
    def _status(dataset_id: str | None, run_id: str | None):
        parts = []
        parts.append(f"dataset={dataset_id}" if dataset_id else "dataset=<none>")
        parts.append(f"run={run_id}" if run_id else "run=<none>")
        return html.Code("  ".join(parts))

