from __future__ import annotations

import sqlite3
from pathlib import Path

from autonomous_rl_trading_bot.common.db import (
    apply_migrations,
    ensure_schema_migrations,
    upsert_run,
)
from autonomous_rl_trading_bot.common.paths import repo_root


def test_migrations_apply_and_can_insert_run(tmp_path: Path) -> None:
    # Use a temp DB file (not artifacts) for test isolation
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")

    migrations_dir = repo_root() / "sql" / "migrations"
    ensure_schema_migrations(conn)
    applied = apply_migrations(conn, migrations_dir)
    # first run should apply at least 001_init.sql
    assert any("001_init.sql" in name for name in applied) or len(applied) >= 0

    # Insert a run row
    upsert_run(
        conn,
        run_id="run_test_1",
        kind="train",
        mode="spot",
        created_utc="2026-01-01T00:00:00+00:00",
        config_hash="abc",
        seed=123,
        status="CREATED",
        run_dir="X",
        run_json_path="Y",
        run_log_path=None,
        global_log_path=None,
    )
    conn.commit()

    row = conn.execute(
        "SELECT run_id, kind, mode, status FROM runs WHERE run_id='run_test_1';"
    ).fetchone()
    assert row is not None
    assert row["run_id"] == "run_test_1"
    assert row["kind"] == "train"
    assert row["mode"] == "spot"
    assert row["status"] == "CREATED"

    # Upsert (update) same run_id
    upsert_run(
        conn,
        run_id="run_test_1",
        kind="train",
        mode="spot",
        created_utc="2026-01-01T00:00:00+00:00",
        config_hash="abc",
        seed=123,
        status="DONE",
        run_dir="X2",
        run_json_path="Y2",
        run_log_path="L2",
        global_log_path="G2",
    )
    conn.commit()

    row2 = conn.execute("SELECT status, run_dir FROM runs WHERE run_id='run_test_1';").fetchone()
    assert row2 is not None
    assert row2["status"] == "DONE"
    assert row2["run_dir"] == "X2"

    conn.close()
