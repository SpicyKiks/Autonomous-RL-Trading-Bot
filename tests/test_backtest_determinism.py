from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from autonomous_rl_trading_bot.common.db import apply_migrations, ensure_schema_migrations
from autonomous_rl_trading_bot.common.paths import repo_root
from autonomous_rl_trading_bot.common.reproducibility import set_global_seed
from autonomous_rl_trading_bot.evaluation.baselines import make_strategy
from autonomous_rl_trading_bot.evaluation.backtester import (
    BacktestConfig,
    load_dataset,
    run_futures_backtest,
    run_spot_backtest,
)
from autonomous_rl_trading_bot.evaluation.backtest_runner import run_backtest


def _write_synth_dataset(
    *,
    out_dir: Path,
    mode: str,
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    n: int = 200,
    start_ms: int = 0,
    step_ms: int = 60_000,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    times = np.arange(start_ms, start_ms + n * step_ms, step_ms, dtype=np.int64)
    # deterministic, slightly trending prices
    close = (100.0 + 0.05 * np.arange(n)).astype(np.float64)

    meta = {
        "dataset_id": "ds_test_1",
        "exchange": "binance",
        "market_type": mode,
        "symbol": symbol,
        "interval": interval,
        "window_minutes": int((n * step_ms) / 60_000),
        "required_points": int(n),
        "features": ["open_time_ms", "close"],
    }

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "dataset.npz", open_time_ms=times, close=close)
    return out_dir


def test_spot_backtest_is_deterministic(tmp_path: Path) -> None:
    # Set seed for determinism (bootstrap CI uses random)
    set_global_seed(1337)
    
    ds_dir = _write_synth_dataset(out_dir=tmp_path / "ds", mode="spot")
    meta, arrays = load_dataset(ds_dir)

    cfg = BacktestConfig(
        initial_cash=1000.0,
        order_size_quote=0.0,
        taker_fee_rate=0.001,
        slippage_bps=0.0,
    )
    strat = make_strategy("buy_and_hold")

    e1, t1, m1, _ = run_spot_backtest(dataset_meta=meta, arrays=arrays, strategy=strat, cfg=cfg)
    # re-instantiate strategy to avoid internal state bleed (should still be deterministic)
    set_global_seed(1337)  # Reset seed for second run
    strat2 = make_strategy("buy_and_hold")
    e2, t2, m2, _ = run_spot_backtest(dataset_meta=meta, arrays=arrays, strategy=strat2, cfg=cfg)

    assert m1.to_dict() == m2.to_dict()
    assert e1 == e2
    assert t1 == t2


def test_futures_backtest_is_deterministic(tmp_path: Path) -> None:
    # Set seed for determinism (bootstrap CI uses random)
    set_global_seed(1337)
    
    ds_dir = _write_synth_dataset(out_dir=tmp_path / "ds", mode="futures")
    meta, arrays = load_dataset(ds_dir)

    cfg = BacktestConfig(
        initial_cash=1000.0,
        order_size_quote=0.0,
        taker_fee_rate=0.0004,
        slippage_bps=0.0,
        leverage=3.0,
        maintenance_margin_rate=0.005,
        allow_short=True,
        stop_on_liquidation=True,
    )
    # Use a stateful strategy; determinism means same outputs with same fresh instance.
    strat = make_strategy("sma_crossover", {"fast": 5, "slow": 15})
    e1, t1, m1, _ = run_futures_backtest(dataset_meta=meta, arrays=arrays, strategy=strat, cfg=cfg)

    set_global_seed(1337)  # Reset seed for second run
    strat2 = make_strategy("sma_crossover", {"fast": 5, "slow": 15})
    e2, t2, m2, _ = run_futures_backtest(dataset_meta=meta, arrays=arrays, strategy=strat2, cfg=cfg)

    assert m1.to_dict() == m2.to_dict()
    assert e1 == e2
    assert t1 == t2


def test_backtest_runner_writes_artifacts_and_db(tmp_path: Path) -> None:
    # Prepare synthetic dataset
    ds_dir = _write_synth_dataset(out_dir=tmp_path / "artifacts" / "datasets" / "ds_test_1", mode="spot")

    # Minimal config dict (no YAML needed)
    cfg = {
        "mode": {"id": "spot"},
        "run": {"seed": 1337},
        "logging": {"level": "INFO", "console": False},
        "db": {"path": str(tmp_path / "artifacts" / "db" / "bot.db")},
        "data": {"dataset": {"symbol": "BTCUSDT", "interval": "1m"}},
        "evaluation": {
            "backtest": {
                "dataset_id": "ds_test_1",
                "strategy": "buy_and_hold",
                "strategies": {},
                "initial_cash": 1000.0,
                "order_size_quote": 0.0,
                "taker_fee_rate": 0.001,
                "slippage_bps": 0.0,
            }
        },
    }

    run_payload = run_backtest(
        cfg=cfg,
        artifacts_base_dir=tmp_path / "artifacts",
        dataset_dir=ds_dir,
        run_id="run_backtest_unit_1",
    )

    out_dir = Path(run_payload["result"]["out_dir"])
    assert out_dir.exists()
    assert (out_dir / "equity.csv").exists()
    assert (out_dir / "trades.csv").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "params.json").exists()
    assert (out_dir / "dataset_meta.json").exists()

    # DB rows exist
    db_path = Path(cfg["db"]["path"])
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    ensure_schema_migrations(conn)
    apply_migrations(conn, repo_root() / "sql" / "migrations")

    row = conn.execute(
        "SELECT backtest_id, run_id, status FROM backtests WHERE run_id=?;",
        ("run_backtest_unit_1",),
    ).fetchone()
    assert row is not None
    assert row["status"] == "DONE"
    conn.close()

