from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, html

from .components import card, make_table
from .data_api import DashboardDataAPI
from .live_session import get_session, start_trading, stop_trading, SessionStatus


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

    # ─────────────────────────────────────────────────────────────
    # Live trading: start/stop controls
    # ─────────────────────────────────────────────────────────────
    @app.callback(
        Output("live-status", "children"),
        Output("live-error", "children"),
        Output("live-error-container", "style"),
        Output("live-start-btn", "disabled"),
        Output("live-stop-btn", "disabled"),
        Input("live-start-btn", "n_clicks"),
        Input("live-stop-btn", "n_clicks"),
        Input("tick", "n_intervals"),
        State("live-mode", "value"),
        State("live-symbol", "value"),
        State("live-interval", "value"),
        State("live-demo", "value"),
        State("live-policy", "value"),
        State("live-strategy", "value"),
        State("live-sb3-model-path", "value"),
        State("live-max-steps", "value"),
        State("live-max-minutes", "value"),
    )
    def _live_trading_control(
        start_clicks: int,
        stop_clicks: int,
        tick: int,
        mode: str,
        symbol: str,
        interval: str,
        demo: list,
        policy: str,
        strategy: str,
        sb3_model_path: str,
        max_steps: Optional[int],
        max_minutes: Optional[float],
    ):
        session = get_session()
        error_msg = ""

        # Handle stop button
        if stop_clicks > 0:
            try:
                stop_trading()
            except Exception as e:
                error_msg = f"Error stopping: {e}"

        # Handle start button
        if start_clicks > 0:
            # Guard: only start if session is IDLE
            if session.status != SessionStatus.IDLE:
                error_msg = f"Session already {session.status.value}. Stop it first."
            else:
                try:
                    # Build args list for run_live_demo.main()
                    args_list = []
                    if mode:
                        args_list.extend(["--mode", mode])
                    if symbol:
                        args_list.extend(["--symbol", symbol])
                    if interval:
                        args_list.extend(["--interval", interval])
                    if "demo" in demo:
                        args_list.append("--demo")
                    if policy:
                        args_list.extend(["--policy", policy])
                    if policy == "baseline" and strategy:
                        args_list.extend(["--strategy", strategy])
                    if policy == "sb3" and sb3_model_path:
                        args_list.extend(["--sb3_model_path", sb3_model_path])
                    if max_steps:
                        args_list.extend(["--max_steps", str(int(max_steps))])
                    if max_minutes:
                        args_list.extend(["--max_minutes", str(float(max_minutes))])

                    # Generate unique run_id and run_dir (mimic run_live_demo logic)
                    from datetime import datetime, timezone
                    import traceback
                    import logging
                    from autonomous_rl_trading_bot.common.paths import artifacts_dir
                    from autonomous_rl_trading_bot.common.hashing import short_hash
                    from autonomous_rl_trading_bot.common.config import load_config

                    loaded = load_config(mode=mode)
                    cfg_hash = loaded.config_hash
                    symbol_upper = (symbol or "BTCUSDT").upper()
                    interval_str = (interval or "1m").strip()
                    tag = f"{symbol_upper}_{interval_str}"
                    # Include microseconds to ensure uniqueness even with rapid clicks
                    now = datetime.now(timezone.utc)
                    utc_ts = now.strftime("%Y%m%d_%H%M%S_%f")  # Add microseconds
                    run_id = f"{utc_ts}_{mode}_live_{tag}_{short_hash(cfg_hash, 10)}"
                    run_dir = artifacts_dir() / "runs" / run_id
                    
                    # Idempotent directory creation (exist_ok=True)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Log run start
                    logger = logging.getLogger("arbt")
                    logger.info(f"Starting live run_id={run_id} artifacts_dir={run_dir}")

                    start_trading(args_list, run_id, run_dir)
                except Exception as e:
                    import traceback
                    error_msg = f"Error starting: {e}\n\n{traceback.format_exc()}"
                    session.status = SessionStatus.ERROR
                    session.last_error = str(e)
                    session.last_error_traceback = traceback.format_exc()

        # Update status display
        status_text = f"Status: {session.status.value}"
        if session.run_id:
            status_text += f" | Run ID: {session.run_id}"

        # Button states
        start_disabled = session.status == SessionStatus.RUNNING
        stop_disabled = session.status != SessionStatus.RUNNING

        # Build comprehensive error message
        display_error = "No errors"
        error_style = {"marginBottom": "12px", "display": "none"}
        
        if error_msg:
            display_error = error_msg
            error_style = {"marginBottom": "12px"}
        elif session.last_error:
            display_error = session.last_error
            if getattr(session, 'last_error_traceback', None):
                display_error = f"{session.last_error}\n\nFull Traceback:\n{session.last_error_traceback}"
            error_style = {"marginBottom": "12px"}

        return status_text, display_error, error_style, start_disabled, stop_disabled

    # ─────────────────────────────────────────────────────────────
    # Live trading: display data (equity, trades, logs)
    # ─────────────────────────────────────────────────────────────
    @app.callback(
        Output("live-cards", "children"),
        Output("live-equity-graph", "figure"),
        Output("live-trades-table-wrap", "children"),
        Output("live-logs", "children"),
        Input("tick", "n_intervals"),
    )
    def _live_trading_data(_: int):
        session = get_session()

        if session.status == SessionStatus.IDLE or not session.run_dir:
            fig = go.Figure()
            fig.update_layout(title="No active trading session")
            return [], fig, make_table(table_id="live-trades-table", df=pd.DataFrame(), page_size=12), "No logs"

        # Load data
        eq = api.live_equity(session.run_dir)
        tr = api.live_trades(session.run_dir)
        metrics = api.live_metrics(session.run_dir)

        # Cards
        cards: List[Any] = [
            card("Run ID", session.run_id or "N/A"),
            card("Status", session.status.value),
        ]
        if metrics:
            cards.extend(
                [
                    card("Final Equity", metrics.get("final_equity", "N/A")),
                    card("Peak Equity", metrics.get("peak_equity", "N/A")),
                    card("Trade Count", metrics.get("trade_count", 0)),
                    card("Fee Total", metrics.get("fee_total", 0)),
                ]
            )

        # Equity graph
        fig = go.Figure()
        if not eq.empty:
            x = eq.get("open_time_ms", eq.index)
            fig.add_trace(go.Scatter(x=x, y=eq.get("equity", []), mode="lines", name="equity"))
            if "drawdown" in eq.columns:
                fig.add_trace(go.Scatter(x=x, y=eq["drawdown"], mode="lines", name="drawdown"))
            if "position_qty" in eq.columns:
                fig.add_trace(go.Scatter(x=x, y=eq["position_qty"], mode="lines", name="position_qty"))

        fig.update_layout(
            title=f"Live Equity / Drawdown — {session.run_id}",
            xaxis_title="open_time_ms",
            yaxis_title="value",
            height=520,
        )

        # Trades table
        trades_table = make_table(table_id="live-trades-table", df=tr, page_size=12)

        # Logs
        log_text = "\n".join(session.log_lines[-50:]) if session.log_lines else "No logs yet..."
        if session.last_error:
            log_text = f"ERROR: {session.last_error}\n\n{log_text}"

        return cards, fig, trades_table, log_text
