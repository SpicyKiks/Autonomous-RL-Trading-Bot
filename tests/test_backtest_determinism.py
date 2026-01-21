from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from autonomous_rl_trading_bot.common.reproducibility import set_global_seed
from autonomous_rl_trading_bot.evaluation.backtest_runner import run_backtest
from autonomous_rl_trading_bot.evaluation.backtester import (
    BacktestConfig,
    load_dataset,
    run_futures_backtest,
    run_spot_backtest,
)
from autonomous_rl_trading_bot.evaluation.baselines import make_strategy


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
    strat = make_strategy("sma_crossover", fast=5, slow=15)
    e1, t1, m1, _ = run_futures_backtest(dataset_meta=meta, arrays=arrays, strategy=strat, cfg=cfg)

    set_global_seed(1337)  # Reset seed for second run
    strat2 = make_strategy("sma_crossover", fast=5, slow=15)
    e2, t2, m2, _ = run_futures_backtest(dataset_meta=meta, arrays=arrays, strategy=strat2, cfg=cfg)

    assert m1.to_dict() == m2.to_dict()
    assert e1 == e2
    assert t1 == t2


def test_backtest_runner_writes_artifacts_and_db(tmp_path: Path) -> None:
    # Prepare synthetic dataset
    artifacts_base = tmp_path / "artifacts"
    ds_dir = _write_synth_dataset(out_dir=artifacts_base / "datasets" / "ds_test_1", mode="spot")

    # Mock artifacts_dir() to return tmp_path / "artifacts"
    with patch("autonomous_rl_trading_bot.evaluation.backtest_runner.artifacts_dir", return_value=artifacts_base):
        # Minimal config dict (no YAML needed)
        cfg = {
            "initial_cash": 1000.0,
            "order_size_quote": 0.0,
            "taker_fee_rate": 0.001,
            "slippage_bps": 0.0,
        }

        run_payload = run_backtest(
            mode="spot",
            dataset_id="ds_test_1",
            run_id="run_backtest_unit_1",
            cfg=cfg,
        )

        out_dir = Path(run_payload["artifacts_dir"])
        assert out_dir.exists()
        assert (out_dir / "equity.csv").exists()
        assert (out_dir / "trades.csv").exists()
        assert (out_dir / "metrics.json").exists()
        assert (out_dir / "run.json").exists()
        assert (out_dir / "run_input.json").exists()

