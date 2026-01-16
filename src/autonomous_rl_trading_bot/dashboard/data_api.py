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

    # ─────────────────────────────────────────────────────────────
    # Live trading data
    # ─────────────────────────────────────────────────────────────
    def live_equity(self, run_dir: Optional[Path], run_id: Optional[str] = None) -> pd.DataFrame:
        """Load live equity from CSV (if exists) or database (during run)."""
        if not run_dir:
            return pd.DataFrame()
        
        # Try CSV first (written at end of run)
        equity_path = run_dir / "equity.csv"
        if equity_path.exists():
            try:
                return pd.read_csv(equity_path)
            except Exception:
                pass
        
        # If CSV doesn't exist, read from database (during active run)
        if run_id:
            try:
                with _connect(self.db_path) as conn:
                    rows = conn.execute(
                        """
                        SELECT open_time_ms, equity, drawdown, exposure
                        FROM live_equity
                        WHERE live_id = ?
                        ORDER BY open_time_ms ASC
                        """,
                        (run_id,),
                    ).fetchall()
                if rows:
                    return pd.DataFrame([dict(r) for r in rows])
            except Exception:
                pass
        
        return pd.DataFrame()

    def live_trades(self, run_dir: Optional[Path], run_id: Optional[str] = None) -> pd.DataFrame:
        """Load live trades from CSV (if exists) or database (during run)."""
        if not run_dir:
            return pd.DataFrame()
        
        # Try CSV first (written at end of run)
        trades_path = run_dir / "trades.csv"
        if trades_path.exists():
            try:
                return pd.read_csv(trades_path)
            except Exception:
                pass
        
        # If CSV doesn't exist, read from database (during active run)
        if run_id:
            try:
                with _connect(self.db_path) as conn:
                    rows = conn.execute(
                        """
                        SELECT 
                            open_time_ms,
                            side,
                            qty as qty_base,
                            fill_price as price,
                            notional,
                            fee,
                            slippage_cost,
                            realized_pnl,
                            reason
                        FROM live_trades
                        WHERE live_id = ?
                        ORDER BY open_time_ms ASC
                        """,
                        (run_id,),
                    ).fetchall()
                if rows:
                    df = pd.DataFrame([dict(r) for r in rows])
                    # Add trade_id column for compatibility
                    df.insert(0, "trade_id", range(1, len(df) + 1))
                    return df
            except Exception:
                pass
        
        return pd.DataFrame()

    def live_metrics(self, run_dir: Optional[Path]) -> Dict[str, Any]:
        """Load live metrics JSON."""
        if not run_dir:
            return {}
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            return {}
        try:
            import json

            return json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
