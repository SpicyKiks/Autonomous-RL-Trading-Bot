from __future__ import annotations

import sqlite3
from pathlib import Path

from autonomous_rl_trading_bot.common.db import apply_migrations, ensure_schema_migrations
from autonomous_rl_trading_bot.common.paths import repo_root
from autonomous_rl_trading_bot.data.dataset_builder import build_dataset_from_db, validate_candles


def _insert_candle(conn: sqlite3.Connection, t: int, close: float) -> None:
    interval_ms = 60_000
    conn.execute(
        """
        INSERT INTO candles(
          exchange, market_type, symbol, interval, open_time_ms,
          open, high, low, close, volume, close_time_ms,
          quote_asset_volume, number_of_trades,
          taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "binance",
            "spot",
            "BTCUSDT",
            "1m",
            t,
            close,
            close,
            close,
            close,
            1.0,
            t + interval_ms - 1,
            None,
            None,
            None,
            None,
            None,
        ),
    )


def test_validate_detects_gap() -> None:
    interval_ms = 60_000
    times = [0, 60_000, 120_000, 240_000]  # gap between 120_000 and 240_000
    rep = validate_candles(times, interval_ms)
    assert rep.gaps == 1


def test_build_dataset_non_strict_can_extract_segment(tmp_path: Path) -> None:
    db_path = tmp_path / "t.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")

    migrations_dir = repo_root() / "sql" / "migrations"
    ensure_schema_migrations(conn)
    apply_migrations(conn, migrations_dir)

    # Insert 10 minutes, but with a gap in the middle so contiguous tail is shorter
    # times: 0..9 min, remove minute 5
    for i in range(10):
        if i == 5:
            continue
        _insert_candle(conn, i * 60_000, close=100.0 + i)
    conn.commit()

    out_base = tmp_path / "datasets"
    # window_minutes=3 => need 3 points (at 1m interval)
    res = build_dataset_from_db(
        conn,
        market_type="spot",
        symbol="BTCUSDT",
        interval="1m",
        window_minutes=3,
        strict_gaps=False,
        features=["return", "log_return"],
        out_base_dir=out_base,
        dataset_id="ds1",
        end_ms=None,
    )

    assert res.out_dir.exists()
    assert res.npz_path.exists()
    assert res.csv_path.exists()
    assert res.meta_path.exists()
    conn.close()


def test_build_dataset_strict_raises_on_gap(tmp_path: Path) -> None:
    db_path = tmp_path / "t2.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")

    migrations_dir = repo_root() / "sql" / "migrations"
    ensure_schema_migrations(conn)
    apply_migrations(conn, migrations_dir)

    for i in range(6):
        if i == 3:
            continue
        _insert_candle(conn, i * 60_000, close=200.0 + i)
    conn.commit()

    out_base = tmp_path / "datasets"
    raised = False
    try:
        build_dataset_from_db(
            conn,
            market_type="spot",
            symbol="BTCUSDT",
            interval="1m",
            window_minutes=3,
            strict_gaps=True,
            features=["return", "log_return"],
            out_base_dir=out_base,
            dataset_id="ds2",
            end_ms=None,
        )
    except ValueError:
        raised = True

    assert raised
    conn.close()
