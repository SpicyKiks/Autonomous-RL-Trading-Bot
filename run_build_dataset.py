from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from autonomous_rl_trading_bot.common.config import load_config
from autonomous_rl_trading_bot.common.db import connect, migrate, upsert_run
from autonomous_rl_trading_bot.common.hashing import short_hash
from autonomous_rl_trading_bot.common.logging import configure_logging
from autonomous_rl_trading_bot.common.paths import artifacts_dir, ensure_artifact_tree
from autonomous_rl_trading_bot.common.reproducibility import set_global_seed
from autonomous_rl_trading_bot.data.dataset_builder import build_dataset_from_db


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _market_type_from_mode(mode: str) -> str:
    m = (mode or "").strip().lower()
    if m in ("spot", "futures"):
        return m
    return "spot"


def _make_dataset_id(mode: str, symbol: str, interval: str, cfg_hash: str) -> str:
    return f"{_utc_ts()}_{mode}_dataset_{symbol}_{interval}_{short_hash(cfg_hash, 10)}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a clean contiguous dataset from DB candles (Step 4)."
    )
    parser.add_argument(
        "--mode", default=None, help="spot|futures (via configs/modes/* overrides)."
    )
    parser.add_argument("--symbol", default=None, help="e.g. BTCUSDT")
    parser.add_argument("--interval", default=None, help="e.g. 1m, 1h")
    parser.add_argument("--minutes", type=int, default=None, help="window size in minutes")
    parser.add_argument("--strict", action="store_true", help="fail if gaps/duplicates detected")
    parser.add_argument("--no-strict", action="store_true", help="override strict to false")
    parser.add_argument("--train-frac", type=float, default=0.75, help="Training split fraction (default: 0.75)")
    parser.add_argument("--val-frac", type=float, default=0.10, help="Validation split fraction (default: 0.10)")
    parser.add_argument("--test-frac", type=float, default=0.15, help="Test split fraction (default: 0.15)")
    parser.add_argument("--scaler", default="robust", help="Scaler type: robust (default)")
    args = parser.parse_args()

    ensure_artifact_tree()

    loaded = load_config(mode=args.mode)
    cfg = loaded.config
    cfg_hash = loaded.config_hash
    mode = cfg["mode"]["id"]
    market_type = _market_type_from_mode(mode)

    ds_cfg = (cfg.get("data", {}) or {}).get("dataset", {}) or {}
    symbol = (args.symbol or ds_cfg.get("symbol") or "BTCUSDT").upper()
    interval = (args.interval or ds_cfg.get("interval") or "1m").strip()
    window_minutes = int(args.minutes or ds_cfg.get("window_minutes") or 120)

    strict_default = bool(ds_cfg.get("strict_gaps", True))
    strict = strict_default
    if args.strict:
        strict = True
    if args.no_strict:
        strict = False

    features = list(ds_cfg.get("features") or ["return", "log_return"])

    train_frac = float(args.train_frac)
    val_frac = float(args.val_frac)
    test_frac = float(args.test_frac)
    scaler_type = str(args.scaler).strip().lower()

    # Validate splits sum to 1.0
    total_frac = train_frac + val_frac + test_frac
    if abs(total_frac - 1.0) > 1e-6:
        raise SystemExit(f"ERROR: Split fractions must sum to 1.0, got {total_frac}")

    db_path = migrate(cfg)

    dataset_id = _make_dataset_id(mode, symbol, interval, cfg_hash)
    run_id = dataset_id  # keep run_id==dataset_id for traceability

    run_dir = artifacts_dir() / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    log_level = (cfg.get("logging", {}) or {}).get("level", "INFO")
    log_console = bool((cfg.get("logging", {}) or {}).get("console", True))
    log_file = bool((cfg.get("logging", {}) or {}).get("file", True))

    per_run_log: Optional[str] = None
    global_log: Optional[str] = None
    log_paths: List[str] = []

    if log_file:
        per_run_log_p = run_dir / "run.log"
        global_log_p = artifacts_dir() / "logs" / f"{run_id}.log"
        per_run_log = str(per_run_log_p)
        global_log = str(global_log_p)
        log_paths = [per_run_log, global_log]
        logger = configure_logging(
            level=log_level,
            console=log_console,
            file_paths=[per_run_log_p, global_log_p],
            run_id=run_id,
        )
    else:
        logger = configure_logging(
            level=log_level, console=log_console, file_paths=None, run_id=run_id
        )

    seed = int(cfg["run"]["seed"])
    seed_report = set_global_seed(seed)

    created_utc = datetime.now(timezone.utc).isoformat()
    run_json_path = str(run_dir / "run.json")

    with connect(db_path) as conn:
        upsert_run(
            conn,
            run_id=run_id,
            kind="dataset",
            mode=mode,
            created_utc=created_utc,
            config_hash=cfg_hash,
            seed=seed,
            status="CREATED",
            run_dir=str(run_dir),
            run_json_path=run_json_path,
            run_log_path=per_run_log,
            global_log_path=global_log,
        )
        conn.commit()

    out_base = artifacts_dir() / "datasets"
    end_ms = _now_ms()

    logger.info("Step 4 dataset build starting.")
    logger.info(
        "market_type=%s symbol=%s interval=%s window_minutes=%s strict=%s",
        market_type,
        symbol,
        interval,
        window_minutes,
        strict,
    )
    logger.info("db_path=%s", str(db_path))

    with connect(db_path) as conn:
        result = build_dataset_from_db(
            conn,
            market_type=market_type,
            symbol=symbol,
            interval=interval,
            window_minutes=window_minutes,
            strict_gaps=strict,
            features=features,
            out_base_dir=out_base,
            dataset_id=dataset_id,
            end_ms=end_ms,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            scaler_type=scaler_type,
        )

    run_meta: Dict[str, Any] = {
        "run_id": run_id,
        "created_utc": created_utc,
        "kind": "dataset",
        "mode": mode,
        "market_type": market_type,
        "symbol": symbol,
        "interval": interval,
        "window_minutes": window_minutes,
        "strict_gaps": strict,
        "features": features,
        "quality_report": result.report.to_dict(),
        "candles_used": result.candles_used,
        "window_points": result.window_points,
        "dataset_dir": str(result.out_dir),
        "npz_path": str(result.npz_path),
        "csv_path": str(result.csv_path),
        "meta_path": str(result.meta_path),
        "config_hash": cfg_hash,
        "db_path": str(db_path),
        "seed_report": seed_report,
        "log_paths": log_paths,
        "status": "DONE",
    }
    Path(run_json_path).write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with connect(db_path) as conn:
        upsert_run(
            conn,
            run_id=run_id,
            kind="dataset",
            mode=mode,
            created_utc=created_utc,
            config_hash=cfg_hash,
            seed=seed,
            status="DONE",
            run_dir=str(run_dir),
            run_json_path=run_json_path,
            run_log_path=per_run_log,
            global_log_path=global_log,
        )
        conn.commit()

    logger.info("DONE: dataset_id=%s out_dir=%s", dataset_id, str(result.out_dir))
    print(f"OK: dataset built at {result.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
