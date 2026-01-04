from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from autonomous_rl_trading_bot.common.config import LoadedConfig, load_config
from autonomous_rl_trading_bot.common.db import apply_migrations, ensure_schema_migrations, upsert_run
from autonomous_rl_trading_bot.common.hashing import sha256_of_obj
from autonomous_rl_trading_bot.common.logging import configure_logging
from autonomous_rl_trading_bot.common.paths import ensure_artifact_tree, repo_root
from autonomous_rl_trading_bot.common.reproducibility import set_global_seed
from autonomous_rl_trading_bot.evaluation.baselines import make_strategy
from autonomous_rl_trading_bot.evaluation.backtester import (
    BacktestConfig,
    load_dataset,
    persist_backtest_to_db,
    persist_futures_backtest_to_db,
    run_futures_backtest,
    run_spot_backtest,
)
from autonomous_rl_trading_bot.evaluation.reporting import (
    write_equity_csv,
    write_json,
    write_trades_csv,
)
from autonomous_rl_trading_bot.storage.artifact_store import ArtifactStore


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _expand_env_vars(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    if isinstance(obj, str):

        def repl(m: re.Match) -> str:
            key = m.group(1)
            return os.getenv(key, "")

        return _ENV_PATTERN.sub(repl, obj)
    return obj


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config YAML must be a mapping (dict): {path}")
    return data


def _resolve_config(*, config_arg: Optional[str], mode_arg: Optional[str]) -> Dict[str, Any]:
    """Load config from either:

    - default configs/base.yaml (+ configs/modes/<mode>.yaml)
    - a file path
    - a base config name inside configs/ (e.g. "base.yaml")
    """

    if not config_arg:
        loaded: LoadedConfig = load_config(mode=mode_arg)
        return loaded.config

    p = Path(config_arg)
    if p.exists() and p.is_file():
        # Custom YAML path: merge with mode override if present.
        base_cfg = _expand_env_vars(_load_yaml_file(p))
        resolved_mode = mode_arg or (base_cfg.get("mode", {}) or {}).get("id")
        if resolved_mode:
            mode_path = repo_root() / "configs" / "modes" / f"{resolved_mode}.yaml"
            if mode_path.exists():
                mode_cfg = _expand_env_vars(_load_yaml_file(mode_path))
                base_cfg = _deep_merge(base_cfg, mode_cfg)
        return base_cfg

    # Otherwise interpret as base_name under configs/
    loaded = load_config(mode=mode_arg, base_name=config_arg)
    return loaded.config


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def run_backtest(
    *,
    cfg: Dict[str, Any],
    artifacts_base_dir: Optional[Path] = None,
    dataset_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a deterministic offline backtest and persist outputs to artifacts + SQLite.

    Notes:
      - We use run_id == backtest_id (see sql/migrations/003_backtests.sql).
      - Equity/trade determinism is independent from run_id randomness.
      - Provide `run_id` for repeatable paths in tests.
    """

    # If artifacts_base_dir is not provided we default to repo artifacts/ and ensure the standard tree exists.
    if artifacts_base_dir is None:
        ensure_artifact_tree()
        store = ArtifactStore.default()
    else:
        store = ArtifactStore(base_dir=artifacts_base_dir)

    mode = str(_get(cfg, "mode", "id", default="spot")).lower().strip()
    if mode not in ("spot", "futures"):
        raise ValueError(f"Unsupported mode.id={mode!r}; expected 'spot' or 'futures'.")

    seed = int(_get(cfg, "run", "seed", default=1337))
    seed_report = set_global_seed(seed)

    # Logging (console + run log file)
    if run_id is None:
        run_id = store.new_run_id(kind="backtest", mode=mode)
    backtest_id = run_id

    run_dir = store.run_dir(run_id)
    run_log = run_dir / "run.log"
    logger = configure_logging(
        level=str(_get(cfg, "logging", "level", default="INFO")),
        console=bool(_get(cfg, "logging", "console", default=True)),
        file_paths=[run_log],
        run_id=run_id,
    )

    logger.info("Starting backtest")

    # Resolve dataset
    if dataset_dir is None:
        ds_id = _get(cfg, "evaluation", "backtest", "dataset_id", default=None)
        if ds_id:
            dataset_dir = store.dataset_dir(str(ds_id))
        else:
            dataset_dir = store.latest_dataset_dir(
                market_type=mode,
                symbol=_get(cfg, "data", "dataset", "symbol", default=None),
                interval=_get(cfg, "data", "dataset", "interval", default=None),
            )

    if dataset_dir is None or not dataset_dir.exists():
        raise FileNotFoundError(
            "No dataset_dir provided and no dataset found under artifacts/datasets. "
            "Build a dataset in Step 4 first."
        )

    dataset_meta, arrays = load_dataset(Path(dataset_dir))
    dataset_id = str(dataset_meta.get("dataset_id") or "")

    # Strategy selection
    strat_name = str(_get(cfg, "evaluation", "backtest", "strategy", default="buy_and_hold"))
    strat_params_all = _get(cfg, "evaluation", "backtest", "strategies", default={})
    if not isinstance(strat_params_all, dict):
        strat_params_all = {}
    strat_params = strat_params_all.get(strat_name, strat_params_all.get(strat_name.lower(), {}))
    if not isinstance(strat_params, dict):
        strat_params = {}

    strategy = make_strategy(strat_name, params=strat_params)

    # Engine config
    bt_cfg = BacktestConfig(
        initial_cash=float(_get(cfg, "evaluation", "backtest", "initial_cash", default=1000.0)),
        order_size_quote=float(_get(cfg, "evaluation", "backtest", "order_size_quote", default=0.0)),
        taker_fee_rate=float(_get(cfg, "evaluation", "backtest", "taker_fee_rate", default=0.001)),
        slippage_bps=float(_get(cfg, "evaluation", "backtest", "slippage_bps", default=0.0)),
        leverage=float(_get(cfg, "evaluation", "backtest", "leverage", default=1.0)),
        maintenance_margin_rate=float(
            _get(cfg, "evaluation", "backtest", "maintenance_margin_rate", default=0.005)
        ),
        allow_short=bool(_get(cfg, "evaluation", "backtest", "allow_short", default=True)),
        stop_on_liquidation=bool(
            _get(cfg, "evaluation", "backtest", "stop_on_liquidation", default=True)
        ),
    )

    started_utc = _utc_iso()
    status = "DONE"
    err: Optional[str] = None
    result: Dict[str, Any] = {}

    # DB (migrations + runs/backtests rows)
    db_cfg = _get(cfg, "db", default={})
    db_path: Optional[Path] = None
    if isinstance(db_cfg, dict) and db_cfg.get("path"):
        db_path = Path(str(db_cfg["path"]))
    if db_path is None:
        db_path = store.db_dir() / str(_get(cfg, "db", "filename", default="bot.db"))
    db_path.parent.mkdir(parents=True, exist_ok=True)

    migrations_dir = repo_root() / "sql" / "migrations"

    # Prepare paths for run record
    run_json_path = run_dir / "run.json"

    # Backtest output dir
    out_dir = store.backtest_dir(backtest_id)

    # Parameter payload for persistence
    params_payload = {
        "mode": mode,
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "dataset_id": dataset_id,
        "strategy": {
            "name": strat_name,
            "params": strat_params,
        },
        "backtest_config": asdict(bt_cfg),
        "seed": seed,
    }
    params_json = json.dumps(params_payload, ensure_ascii=False)

    # Insert/Upsert run row early as CREATED
    config_hash = sha256_of_obj(cfg)
    created_utc = started_utc

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        ensure_schema_migrations(conn)
        apply_migrations(conn, migrations_dir)

        upsert_run(
            conn,
            run_id=run_id,
            kind="backtest",
            mode=mode,
            created_utc=created_utc,
            config_hash=config_hash,
            seed=seed,
            status="CREATED",
            run_dir=str(run_dir),
            run_json_path=str(run_json_path),
            run_log_path=str(run_log),
            global_log_path=None,
        )
        conn.commit()

        # Execute engine
        if mode == "spot":
            equity_rows, trade_rows, metrics, extra = run_spot_backtest(
                dataset_meta=dataset_meta,
                arrays=arrays,
                strategy=strategy,
                cfg=bt_cfg,
            )

            # Artifacts
            write_equity_csv(out_dir / "equity.csv", equity_rows)
            write_trades_csv(out_dir / "trades.csv", trade_rows)
            write_json(out_dir / "metrics.json", metrics.to_dict())
            write_json(out_dir / "params.json", params_payload)
            write_json(out_dir / "dataset_meta.json", dict(dataset_meta))

            finished_utc = _utc_iso()
            persist_backtest_to_db(
                conn=conn,
                backtest_id=backtest_id,
                run_id=run_id,
                mode=mode,
                dataset_meta=dataset_meta,
                cfg=bt_cfg,
                params_json=params_json,
                metrics=metrics,
                equity_rows=equity_rows,
                trade_rows=trade_rows,
                started_utc=started_utc,
                finished_utc=finished_utc,
                status=status,
            )
            conn.commit()

            result = {
                "mode": mode,
                "run_id": run_id,
                "backtest_id": backtest_id,
                "dataset_id": dataset_id,
                "out_dir": str(out_dir),
                "metrics": metrics.to_dict(),
                "extra": extra,
            }

        else:
            equity_rows, trade_rows, metrics, extra = run_futures_backtest(
                dataset_meta=dataset_meta,
                arrays=arrays,
                strategy=strategy,
                cfg=bt_cfg,
            )

            write_equity_csv(out_dir / "equity.csv", equity_rows)
            write_trades_csv(out_dir / "trades.csv", trade_rows)
            write_json(out_dir / "metrics.json", metrics.to_dict())
            write_json(out_dir / "params.json", params_payload)
            write_json(out_dir / "dataset_meta.json", dict(dataset_meta))

            finished_utc = _utc_iso()
            persist_futures_backtest_to_db(
                conn=conn,
                backtest_id=backtest_id,
                run_id=run_id,
                mode=mode,
                dataset_meta=dataset_meta,
                params_json=params_json,
                metrics=metrics,
                equity_rows=equity_rows,
                trade_rows=trade_rows,
                started_utc=started_utc,
                finished_utc=finished_utc,
                status=status,
            )
            conn.commit()

            result = {
                "mode": mode,
                "run_id": run_id,
                "backtest_id": backtest_id,
                "dataset_id": dataset_id,
                "out_dir": str(out_dir),
                "metrics": metrics.to_dict(),
                "extra": extra,
            }

        # Mark run DONE
        upsert_run(
            conn,
            run_id=run_id,
            kind="backtest",
            mode=mode,
            created_utc=created_utc,
            config_hash=config_hash,
            seed=seed,
            status="DONE",
            run_dir=str(run_dir),
            run_json_path=str(run_json_path),
            run_log_path=str(run_log),
            global_log_path=None,
        )
        conn.commit()

    except Exception as e:
        conn.rollback()
        status = "FAILED"
        err = f"{type(e).__name__}: {e}"
        logger.exception("Backtest failed")
        try:
            upsert_run(
                conn,
                run_id=run_id,
                kind="backtest",
                mode=mode,
                created_utc=created_utc,
                config_hash=config_hash,
                seed=seed,
                status="FAILED",
                run_dir=str(run_dir),
                run_json_path=str(run_json_path),
                run_log_path=str(run_log),
                global_log_path=None,
            )
            conn.commit()
        except Exception:
            conn.rollback()
        raise
    finally:
        conn.close()

    # Always write run.json at the end
    run_payload = {
        "run_id": run_id,
        "kind": "backtest",
        "mode": mode,
        "created_utc": created_utc,
        "started_utc": started_utc,
        "status": status,
        "error": err,
        "config_hash": config_hash,
        "seed_report": seed_report,
        "result": result,
    }
    run_json_path.write_text(json.dumps(run_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Backtest finished: status=%s, out_dir=%s", status, out_dir)
    return run_payload


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run deterministic offline backtests (spot/futures).")
    p.add_argument(
        "--config",
        default=None,
        help=(
            "Config selector. Either an explicit YAML path, or a base config name under configs/ "
            "(e.g. base.yaml). If omitted: configs/base.yaml merged with configs/modes/<mode>.yaml."
        ),
    )
    p.add_argument("--mode", choices=["spot", "futures"], default=None)
    p.add_argument("--dataset-dir", default=None, help="Explicit dataset directory (contains meta.json + dataset.npz).")
    p.add_argument("--run-id", default=None, help="Override run_id/backtest_id (useful for tests).")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    cfg = _resolve_config(config_arg=args.config, mode_arg=args.mode)

    ds_dir = Path(args.dataset_dir) if args.dataset_dir else None

    try:
        run_backtest(cfg=cfg, dataset_dir=ds_dir, run_id=args.run_id)
        return 0
    except Exception as e:
        logging.getLogger("autonomous_rl_trading_bot").error("Backtest error: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

