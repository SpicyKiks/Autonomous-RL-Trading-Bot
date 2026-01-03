from __future__ import annotations

from dash import dcc, html

from .components import empty_state, header
from .data_api import ArtifactIndex


def build_layout(idx: ArtifactIndex) -> html.Div:
    runs = idx.runs or []
    datasets = idx.datasets or []

    return html.Div(
        [
            header(
                "Autonomous RL Trading Bot",
                "Dashboard (offline) — reads artifacts/ only",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Dataset"),
                            dcc.Dropdown(
                                id="dataset_id",
                                options=[{"label": d, "value": d} for d in datasets],
                                value=(datasets[0] if datasets else None),
                                placeholder="No datasets yet",
                                clearable=True,
                            ),
                        ],
                        style={"flex": 1, "minWidth": "280px"},
                    ),
                    html.Div(
                        [
                            html.Label("Run"),
                            dcc.Dropdown(
                                id="run_id",
                                options=[{"label": r, "value": r} for r in runs],
                                value=(runs[0] if runs else None),
                                placeholder="No runs yet",
                                clearable=True,
                            ),
                        ],
                        style={"flex": 1, "minWidth": "280px"},
                    ),
                    html.Div(
                        [
                            html.Label("Status"),
                            html.Div(id="status", style={"paddingTop": "8px"}),
                        ],
                        style={"flex": 1, "minWidth": "280px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "16px",
                    "padding": "16px",
                    "flexWrap": "wrap",
                },
            ),
            html.Div(
                [
                    empty_state(
                        "This is the step-0 dashboard shell. Later steps will add charts, trades, and live monitoring."
                    )
                ],
                style={"padding": "0 16px 16px 16px"},
            ),
        ],
        style={"fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"},
    )

