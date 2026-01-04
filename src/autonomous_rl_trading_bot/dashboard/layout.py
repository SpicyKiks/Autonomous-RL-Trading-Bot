from __future__ import annotations

from dash import dcc, html

from .components import card


def make_layout(db_path_str: str) -> html.Div:
    return html.Div(
        [
            html.H2("Autonomous RL Trading Bot — Dashboard"),
            html.Div(
                [
                    html.Div("DB:", style={"fontWeight": "bold", "marginRight": "8px"}),
                    html.Code(db_path_str),
                ],
                style={"marginBottom": "10px"},
            ),

            dcc.Interval(id="tick", interval=4000, n_intervals=0),

            dcc.Tabs(
                id="tabs",
                value="tab-backtests",
                children=[
                    dcc.Tab(
                        label="Backtests",
                        value="tab-backtests",
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Market type"),
                                            dcc.Dropdown(
                                                id="bt-market-filter",
                                                options=[
                                                    {"label": "All", "value": ""},
                                                    {"label": "Spot", "value": "spot"},
                                                    {"label": "Futures", "value": "futures"},
                                                ],
                                                value="",
                                                clearable=False,
                                            ),
                                        ],
                                        style={"width": "220px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Select backtest_id"),
                                            dcc.Dropdown(
                                                id="bt-id",
                                                options=[],
                                                value=None,
                                                placeholder="Pick a backtest…",
                                            ),
                                        ],
                                        style={"flex": 1, "marginLeft": "14px"},
                                    ),
                                ],
                                style={"display": "flex", "gap": "10px", "marginBottom": "12px"},
                            ),

                            html.Div(
                                id="bt-cards",
                                style={"display": "flex", "gap": "10px", "flexWrap": "wrap"},
                            ),

                            dcc.Graph(id="bt-equity-graph", style={"marginTop": "10px"}),

                            html.H4("Trades"),
                            html.Div(id="bt-trades-table-wrap"),
                        ],
                    ),

                    dcc.Tab(
                        label="Runs",
                        value="tab-runs",
                        children=[
                            html.H4("Runs"),
                            html.Div(id="runs-table-wrap"),
                        ],
                    ),

                    dcc.Tab(
                        label="Train Jobs",
                        value="tab-train",
                        children=[
                            html.H4("Train Jobs"),
                            html.Div(id="train-table-wrap"),
                        ],
                    ),
                ],
            ),
        ],
        style={"maxWidth": "1200px", "margin": "0 auto", "padding": "16px"},
    )
