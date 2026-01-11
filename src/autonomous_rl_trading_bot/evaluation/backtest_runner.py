from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from autonomous_rl_trading_bot.common.db import apply_migrations, ensure_schema_migrations
from autonomous_rl_trading_bot.common.paths import ensure_dir, repo_root
from autonomous_rl_trading_bot.common.reproducibility import set_global_seed
from autonomous_rl_trading_bot.evaluation.baselines import make_strategy
from autonomous_rl_trading_bot.evaluation.backtester import (
    BacktestConfig,
    load_dataset,
    persist_backtest_to_db,
    run_futures_backtest,
    run_spot_backtest,
)
from autonomous_rl_trading_bot.evaluation.reporting import build_repro_payload, generate_run_report, write_json


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    pref = [
        "t",
        "step",
        "open_time_ms",
        "equity",
        "price",
        "cash",
        "qty_base",
        "drawdown",
        "exposure",
        "side",
        "notional",
        "fee",
        "slippage_cost",
        "pnl",
        "realized_pnl",
    ]
    keys: list[str] = []
    for k in pref:
        if k in rows[0] and k not in keys:
            keys.append(k)
    for k in rows[0].keys():
        if k not in keys:
            keys.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _ensure_run_row(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    kind: str,
    mode: str,
    created_utc: str,
    seed: int,
    status: str,
    run_dir: str,
    run_json_path: str,
    run_log_path: str | None = None,
    global_log_path: str | None = None,
    config_hash: str,
) -> None:
    """
    Inserts a runs row satisfying NOT NULL constraints.
    Uses INSERT OR REPLACE to be idempotent for unit tests.
    """
    conn.execute(
        """
        INSERT OR REPLACE INTO runs(
          run_id, kind, mode, created_utc, config_hash, seed, status,
          run_dir, run_json_path, run_log_path, global_log_path
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            run_id,
            kind,
            mode,
            created_utc,
            config_hash,
            int(seed),
            status,
            run_dir,
            run_json_path,
            run_log_path,
            global_log_path,
        ),
    )


def run_backtest(
    *,
    cfg: Dict[str, Any],
    artifacts_base_dir: Path,
    dataset_dir: Path,
    run_id: str,
) -> Dict[str, Any]:
    """
    Backwards-compatible backtest entrypoint required by existing unit tests.

    Writes required artifacts under:
      <artifacts_base_dir>/backtests/<run_id>/

    Also writes Step 11/12 reporting artifacts in the SAME directory:
      metrics.md, metrics.json, repro.json, report.html, report.pdf, plots, etc.

    Persists into SQLite and guarantees FK integrity by inserting a valid runs row
    (config_hash is NOT NULL in schema).
    """
    artifacts_base_dir = Path(artifacts_base_dir)
    dataset_dir = Path(dataset_dir)

    mode = str(cfg["mode"]["id"]).strip().lower()
    seed = int(cfg["run"]["seed"])
    set_global_seed(seed)

    dataset_meta, arrays = load_dataset(dataset_dir)

    bt_cfg_raw = cfg["evaluation"]["backtest"]
    strategy_name = str(bt_cfg_raw.get("strategy", "buy_and_hold"))
    strategies_cfg = bt_cfg_raw.get("strategies") or {}

    bt_cfg = BacktestConfig(
        initial_cash=float(bt_cfg_raw.get("initial_cash", 1000.0)),
        order_size_quote=float(bt_cfg_raw.get("order_size_quote", 0.0)),
        taker_fee_rate=float(bt_cfg_raw.get("taker_fee_rate", 0.001)),
        slippage_bps=float(bt_cfg_raw.get("slippage_bps", 0.0)),
    )

    started_utc = _iso_utc_now()

    # ✅ backtester expects a Strategy object, not a string
    strategy = make_strategy(strategy_name, strategies_cfg)

    if mode == "spot":
        equity_rows, trade_rows, metrics, params = run_spot_backtest(
            dataset_meta=dataset_meta,
            arrays=arrays,
            strategy=strategy,
            cfg=bt_cfg,
        )
    elif mode == "futures":
        equity_rows, trade_rows, metrics, params = run_futures_backtest(
            dataset_meta=dataset_meta,
            arrays=arrays,
            strategy=strategy,
            cfg=bt_cfg,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    finished_utc = _iso_utc_now()

    out_dir = ensure_dir(artifacts_base_dir / "backtests" / run_id)

    # Core outputs expected by determinism/unit tests
    _write_csv(out_dir / "equity.csv", equity_rows)
    _write_csv(out_dir / "trades.csv", trade_rows)
    write_json(out_dir / "metrics.json", metrics.to_dict())
    write_json(out_dir / "params.json", params)
    write_json(out_dir / "dataset_meta.json", dict(dataset_meta))

    # Minimal run metadata (useful + aligns with schema expectations)
    run_json = {
        "run_id": run_id,
        "kind": "backtest",
        "mode": mode,
        "created_utc": started_utc,
        "finished_utc": finished_utc,
        "seed": seed,
        "dataset_id": str(bt_cfg_raw.get("dataset_id", dataset_meta.get("dataset_id", ""))),
        "strategy": strategy_name,
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "run.json", run_json)

    # Step 11/12 report artifacts
    repro = build_repro_payload(
        seed=seed,
        dataset_id=str(bt_cfg_raw.get("dataset_id", dataset_meta.get("dataset_id", ""))),
        dataset_hash=str(dataset_meta.get("dataset_hash")) if dataset_meta.get("dataset_hash") is not None else None,
        kind="backtest",
        run_id=run_id,
        mode=mode,
        config_hash=None,
        extra={
            "strategy": strategy_name,
            "taker_fee_rate": bt_cfg.taker_fee_rate,
            "slippage_bps": bt_cfg.slippage_bps,
        },
    )

    generate_run_report(
        out_dir,
        title=f"Backtest - {mode.upper()} - {dataset_meta.get('symbol','')} - {strategy_name}",
        equity_rows=equity_rows,
        trades_rows=trade_rows,
        metrics=metrics.to_dict(),
        repro=repro,
    )

    # Persist to DB (tests verify FK + DONE status)
    db_path = Path(cfg["db"]["path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    ensure_schema_migrations(conn)
    apply_migrations(conn, repo_root() / "sql" / "migrations")

    # runs.config_hash is NOT NULL. Provide a deterministic non-null value.
    config_hash = f"cfg::{mode}::{run_id}"

    _ensure_run_row(
        conn,
        run_id=run_id,
        kind="backtest",
        mode=mode,
        created_utc=started_utc,
        seed=seed,
        status="DONE",
        run_dir=str(out_dir),
        run_json_path=str(out_dir / "run.json"),
        run_log_path=None,
        global_log_path=None,
        config_hash=config_hash,
    )
    conn.commit()

    persist_backtest_to_db(
        conn=conn,
        backtest_id=run_id,
        run_id=run_id,
        mode=mode,
        dataset_meta=dataset_meta,
        cfg=bt_cfg,
        params_json=json.dumps(params, ensure_ascii=False),
        metrics=metrics,
        equity_rows=equity_rows,
        trade_rows=trade_rows,
        started_utc=started_utc,
        finished_utc=finished_utc,
        status="DONE",
    )
    conn.commit()
    conn.close()

    return {
        "result": {"out_dir": str(out_dir)},
        "metrics": metrics.to_dict(),
        "params": params,
    }
