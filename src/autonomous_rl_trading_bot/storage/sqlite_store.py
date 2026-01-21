from __future__ import annotations

import sqlite3
from pathlib import Path

from autonomous_rl_trading_bot.storage.models import BacktestModel, RunModel, TrainJobModel


class SQLiteStore:
    """Lightweight SQLite store for runs/backtests/train_jobs."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def list_runs(self, *, limit: int = 100, kind: str | None = None) -> list[RunModel]:
        """List runs, optionally filtered by kind."""
        with self._connect() as conn:
            if kind:
                rows = conn.execute(
                    "SELECT * FROM runs WHERE kind=? ORDER BY created_utc DESC LIMIT ?",
                    (kind, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM runs ORDER BY created_utc DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [RunModel(**dict(row)) for row in rows]

    def get_run(self, run_id: str) -> RunModel | None:
        """Get a single run by ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            return RunModel(**dict(row)) if row else None

    def list_backtests(self, *, limit: int = 100, market_type: str | None = None) -> list[BacktestModel]:
        """List backtests, optionally filtered by market_type."""
        with self._connect() as conn:
            if market_type:
                rows = conn.execute(
                    "SELECT * FROM backtests WHERE market_type=? ORDER BY started_utc DESC LIMIT ?",
                    (market_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM backtests ORDER BY started_utc DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [BacktestModel(**dict(row)) for row in rows]

    def get_backtest(self, backtest_id: str) -> BacktestModel | None:
        """Get a single backtest by ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM backtests WHERE backtest_id=?", (backtest_id,)).fetchone()
            return BacktestModel(**dict(row)) if row else None

    def list_train_jobs(self, *, limit: int = 100, algo: str | None = None) -> list[TrainJobModel]:
        """List train jobs, optionally filtered by algo."""
        with self._connect() as conn:
            if algo:
                rows = conn.execute(
                    "SELECT * FROM train_jobs WHERE algo=? ORDER BY started_utc DESC LIMIT ?",
                    (algo, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM train_jobs ORDER BY started_utc DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [TrainJobModel(**dict(row)) for row in rows]

    def get_train_job(self, train_id: str) -> TrainJobModel | None:
        """Get a single train job by ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM train_jobs WHERE train_id=?", (train_id,)).fetchone()
            return TrainJobModel(**dict(row)) if row else None

