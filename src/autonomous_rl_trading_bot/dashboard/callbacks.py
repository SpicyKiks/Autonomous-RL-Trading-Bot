from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, html

from .components import card, make_table
from .data_api import DashboardDataAPI


def _safe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    return df if df is not None else pd.DataFrame()


def register_callbacks(app, api: DashboardDataAPI) -> None:
    # ─────────────────────────────────────────────────────────────
    # Tables: runs + train jobs (refresh periodically)
    # ─────────────────────────────────────────────────────────────
    @app.callback(
        Output("runs-table-wrap", "children"),
        Input("tick", "n_intervals"),
    )
    def _refresh_runs(_: int):
        df = api.runs(limit=250)
        return make_table(table_id="runs-table", df=df, page_size=15)

    @app.callback(
        Output("train-table-wrap", "children"),
        Input("tick", "n_intervals"),
    )
    def _refresh_train(_: int):
        df = api.train_jobs(limit=250)
        return make_table(table_id="train-table", df=df, page_size=15)

    # ─────────────────────────────────────────────────────────────
    # Backtest dropdown options
    # ─────────────────────────────────────────────────────────────
    @app.callback(
        Output("bt-id", "options"),
        Output("bt-id", "value"),
        Input("tick", "n_intervals"),
        Input("bt-market-filter", "value"),
        State("bt-id", "value"),
    )
    def _refresh_backtest_ids(_: int, market_filter: str, current_value: Optional[str]):
        mf = (market_filter or "").strip() or None
        df = api.backtests(limit=250, market_type=mf)
        options = [{"label": f'{r["backtest_id"]} ({r["market_type"]}, {r["symbol"]}, {r["status"]})', "value": r["backtest_id"]}
                   for r in df.to_dict("records")]
        # Keep current selection if still present, otherwise auto-pick newest if available.
        values = {o["value"] for o in options}
        if current_value in values:
            return options, current_value
        if options:
            return options, options[0]["value"]
        return [], None

    # ─────────────────────────────────────────────────────────────
    # Backtest details: cards + equity graph + trades table
    # ─────────────────────────────────────────────────────────────
    @app.callback(
        Output("bt-cards", "children"),
        Output("bt-equity-graph", "figure"),
        Output("bt-trades-table-wrap", "children"),
        Input("bt-id", "value"),
    )
    def _backtest_detail(backtest_id: Optional[str]):
        if not backtest_id:
            fig = go.Figure()
            fig.update_layout(title="No backtest selected")
            return [], fig, make_table(table_id="bt-trades-table", df=pd.DataFrame(), page_size=12)

        header = api.backtest_header(backtest_id)
        if not header:
            fig = go.Figure()
            fig.update_layout(title=f"Backtest not found: {backtest_id}")
            return [], fig, make_table(table_id="bt-trades-table", df=pd.DataFrame(), page_size=12)

        mt = str(header.get("market_type") or header.get("mode") or "spot")
        eq = api.backtest_equity(backtest_id, market_type=mt)
        tr = api.backtest_trades(backtest_id, market_type=mt)

        # Cards
        cards: List[Any] = [
            card("backtest_id", header.get("backtest_id")),
            card("market_type", header.get("market_type")),
            card("symbol", header.get("symbol")),
            card("interval", header.get("interval")),
            card("status", header.get("status")),
            card("total_return", header.get("total_return")),
            card("max_drawdown", header.get("max_drawdown")),
            card("trade_count", header.get("trade_count")),
            card("final_equity", header.get("final_equity")),
        ]

        # Equity graph (spot: equity/drawdown; futures: equity/drawdown + position)
        fig = go.Figure()
        if not eq.empty:
            x = eq["open_time_ms"]
            fig.add_trace(go.Scatter(x=x, y=eq["equity"], mode="lines", name="equity"))
            if "drawdown" in eq.columns:
                fig.add_trace(go.Scatter(x=x, y=eq["drawdown"], mode="lines", name="drawdown"))
            if mt.strip().lower() == "futures" and "position_qty" in eq.columns:
                fig.add_trace(go.Scatter(x=x, y=eq["position_qty"], mode="lines", name="position_qty"))

        fig.update_layout(
            title=f"Equity / Drawdown — {backtest_id}",
            xaxis_title="open_time_ms",
            yaxis_title="value",
            height=520,
        )

        trades_table = make_table(table_id="bt-trades-table", df=tr, page_size=12)
        return cards, fig, trades_table
