from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@dataclass(frozen=True)
class DashboardDataAPI:
    db_path: Path

    # ─────────────────────────────────────────────────────────────
    # High-level lists
    # ─────────────────────────────────────────────────────────────
    def runs(self, limit: int = 200) -> pd.DataFrame:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT
                  run_id, kind, mode, status, created_utc,
                  run_dir, run_json_path, run_log_path, global_log_path,
                  config_hash, seed
                FROM runs
                ORDER BY created_utc DESC
                LIMIT ?;
                """,
                (int(limit),),
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def backtests(self, limit: int = 200, market_type: Optional[str] = None) -> pd.DataFrame:
        if market_type:
            sql = """
                SELECT
                  backtest_id, run_id, mode, dataset_id, market_type, symbol, interval,
                  started_utc, finished_utc, status,
                  initial_cash, final_equity, total_return, max_drawdown, trade_count,
                  fee_total, slippage_total
                FROM backtests
                WHERE lower(market_type)=lower(?)
                ORDER BY started_utc DESC
                LIMIT ?;
            """
            params: Tuple[Any, ...] = (market_type, int(limit))
        else:
            sql = """
                SELECT
                  backtest_id, run_id, mode, dataset_id, market_type, symbol, interval,
                  started_utc, finished_utc, status,
                  initial_cash, final_equity, total_return, max_drawdown, trade_count,
                  fee_total, slippage_total
                FROM backtests
                ORDER BY started_utc DESC
                LIMIT ?;
            """
            params = (int(limit),)

        with _connect(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def train_jobs(self, limit: int = 200) -> pd.DataFrame:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT
                  train_id, run_id, mode, market_type, dataset_id,
                  algo, total_timesteps, seed,
                  started_utc, finished_utc, status,
                  model_path
                FROM train_jobs
                ORDER BY started_utc DESC
                LIMIT ?;
                """,
                (int(limit),),
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    # ─────────────────────────────────────────────────────────────
    # Backtest details
    # ─────────────────────────────────────────────────────────────
    def backtest_header(self, backtest_id: str) -> Dict[str, Any]:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM backtests WHERE backtest_id=?;",
                (backtest_id,),
            ).fetchone()
        return dict(row) if row else {}

    def backtest_equity(self, backtest_id: str, market_type: str) -> pd.DataFrame:
        mt = (market_type or "").strip().lower()
        if mt == "futures":
            sql = """
              SELECT *
              FROM backtest_futures_equity
              WHERE backtest_id=?
              ORDER BY open_time_ms ASC;
            """
        else:
            sql = """
              SELECT *
              FROM backtest_equity
              WHERE backtest_id=?
              ORDER BY open_time_ms ASC;
            """
        with _connect(self.db_path) as conn:
            rows = conn.execute(sql, (backtest_id,)).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def backtest_trades(self, backtest_id: str, market_type: str) -> pd.DataFrame:
        mt = (market_type or "").strip().lower()
        if mt == "futures":
            sql = """
              SELECT *
              FROM backtest_futures_trades
              WHERE backtest_id=?
              ORDER BY open_time_ms ASC;
            """
        else:
            sql = """
              SELECT *
              FROM backtest_trades
              WHERE backtest_id=?
              ORDER BY open_time_ms ASC;
            """
        with _connect(self.db_path) as conn:
            rows = conn.execute(sql, (backtest_id,)).fetchall()
        return pd.DataFrame([dict(r) for r in rows])
