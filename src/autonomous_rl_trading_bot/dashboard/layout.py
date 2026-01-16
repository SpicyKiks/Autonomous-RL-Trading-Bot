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

            dcc.Interval(id="tick", interval=4000, n_intervals=0),  # 4 seconds default refresh

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

                    dcc.Tab(
                        label="Live Trading",
                        value="tab-live",
                        children=[
                            html.H3("Live Trading Control"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Mode"),
                                            dcc.Dropdown(
                                                id="live-mode",
                                                options=[
                                                    {"label": "Spot", "value": "spot"},
                                                    {"label": "Futures", "value": "futures"},
                                                ],
                                                value="futures",
                                                clearable=False,
                                            ),
                                        ],
                                        style={"width": "150px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Symbol"),
                                            dcc.Input(
                                                id="live-symbol",
                                                type="text",
                                                value="BTCUSDT",
                                                placeholder="BTCUSDT",
                                            ),
                                        ],
                                        style={"width": "150px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Interval"),
                                            dcc.Dropdown(
                                                id="live-interval",
                                                options=[
                                                    {"label": "1m", "value": "1m"},
                                                    {"label": "5m", "value": "5m"},
                                                    {"label": "15m", "value": "15m"},
                                                    {"label": "1h", "value": "1h"},
                                                ],
                                                value="1m",
                                                clearable=False,
                                            ),
                                        ],
                                        style={"width": "100px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Demo Mode"),
                                            dcc.Checklist(
                                                id="live-demo",
                                                options=[{"label": "Enable", "value": "demo"}],
                                                value=[],
                                            ),
                                        ],
                                        style={"width": "120px"},
                                    ),
                                ],
                                style={"display": "flex", "gap": "10px", "marginBottom": "12px", "flexWrap": "wrap"},
                            ),

                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Policy"),
                                            dcc.Dropdown(
                                                id="live-policy",
                                                options=[
                                                    {"label": "Baseline", "value": "baseline"},
                                                    {"label": "SB3", "value": "sb3"},
                                                ],
                                                value="baseline",
                                                clearable=False,
                                            ),
                                        ],
                                        style={"width": "150px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Strategy (Baseline)"),
                                            dcc.Dropdown(
                                                id="live-strategy",
                                                options=[
                                                    {"label": "Buy & Hold", "value": "buy_and_hold"},
                                                    {"label": "SMA Crossover", "value": "sma_crossover"},
                                                    {"label": "EMA Crossover", "value": "ema_crossover"},
                                                    {"label": "RSI Reversion", "value": "rsi_reversion"},
                                                ],
                                                value="buy_and_hold",
                                                clearable=False,
                                            ),
                                        ],
                                        style={"width": "200px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label("SB3 Model Path"),
                                            dcc.Input(
                                                id="live-sb3-model-path",
                                                type="text",
                                                value="",
                                                placeholder="artifacts/runs/.../policy.zip",
                                            ),
                                        ],
                                        style={"flex": 1, "minWidth": "300px"},
                                    ),
                                ],
                                style={"display": "flex", "gap": "10px", "marginBottom": "12px", "flexWrap": "wrap"},
                            ),

                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Max Steps (optional)"),
                                            dcc.Input(id="live-max-steps", type="number", value=None, placeholder="None"),
                                        ],
                                        style={"width": "150px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Max Minutes (optional)"),
                                            dcc.Input(id="live-max-minutes", type="number", value=None, placeholder="None"),
                                        ],
                                        style={"width": "150px"},
                                    ),
                                ],
                                style={"display": "flex", "gap": "10px", "marginBottom": "12px"},
                            ),

                            html.Div(
                                [
                                    html.Button("Start Trading", id="live-start-btn", n_clicks=0, style={"marginRight": "10px"}),
                                    html.Button("Stop Trading", id="live-stop-btn", n_clicks=0, disabled=False),
                                    html.Div(
                                        id="live-status",
                                        children="Status: IDLE",
                                        style={"marginLeft": "20px", "fontWeight": "bold", "display": "inline-block"},
                                    ),
                                ],
                                style={"marginBottom": "12px"},
                            ),

                            html.Div(id="live-error", style={"color": "red", "marginBottom": "12px"}),

                            html.H4("Live Trading Data"),
                            html.Div(
                                id="live-cards",
                                style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "12px"},
                            ),

                            dcc.Graph(id="live-equity-graph", style={"marginTop": "10px"}),

                            html.H4("Trades"),
                            html.Div(id="live-trades-table-wrap"),

                            html.H4("Logs"),
                            html.Div(
                                id="live-logs",
                                style={
                                    "border": "1px solid #ccc",
                                    "borderRadius": "4px",
                                    "padding": "8px",
                                    "maxHeight": "300px",
                                    "overflowY": "auto",
                                    "fontFamily": "monospace",
                                    "fontSize": "11px",
                                    "backgroundColor": "#f5f5f5",
                                },
                            ),
                        ],
                    ),
                ],
            ),
        ],
        style={"maxWidth": "1200px", "margin": "0 auto", "padding": "16px"},
    )
