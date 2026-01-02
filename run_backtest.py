from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from autonomous_rl_trading_bot.common.config import load_config
from autonomous_rl_trading_bot.common.db import connect, migrate, upsert_run
from autonomous_rl_trading_bot.common.hashing import short_hash
from autonomous_rl_trading_bot.common.logging import configure_logging
from autonomous_rl_trading_bot.common.paths import artifacts_dir, ensure_artifact_tree
from autonomous_rl_trading_bot.common.reproducibility import set_global_seed


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _make_run_id(mode: str, cfg_hash: str) -> str:
    return f"{_utc_ts()}_{mode}_backtest_{short_hash(cfg_hash, 10)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest runner (Step 2 DB-integrated stub).")
    parser.add_argument("--mode", default=None, help="Mode id (spot/futures). Overrides base.yaml.")
    args = parser.parse_args()

    ensure_artifact_tree()

    loaded = load_config(mode=args.mode)
    cfg = loaded.config
    cfg_hash = loaded.config_hash
    mode = cfg["mode"]["id"]

    # Ensure DB migrated before run creation
    db_path = migrate(cfg)

    run_id = _make_run_id(mode, cfg_hash)

    run_dir = artifacts_dir() / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    log_level = (cfg.get("logging", {}) or {}).get("level", "INFO")
    log_console = bool((cfg.get("logging", {}) or {}).get("console", True))
    log_file = bool((cfg.get("logging", {}) or {}).get("file", True))

    per_run_log: Optional[str] = None
    global_log: Optional[str] = None
    log_paths = []

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

    run_meta: Dict[str, Any] = {
        "run_id": run_id,
        "created_utc": created_utc,
        "kind": "backtest",
        "mode": mode,
        "config_hash": cfg_hash,
        "config": cfg,
        "seed_report": seed_report,
        "log_paths": log_paths,
        "db_path": str(db_path),
        "status": "CREATED",
    }

    (run_dir / "run.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Record in DB
    with connect(db_path) as conn:
        upsert_run(
            conn,
            run_id=run_id,
            kind="backtest",
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

    logger.info("Step 2 backtest run created + recorded in DB.")
    logger.info("run_id=%s", run_id)
    logger.info("run_dir=%s", str(run_dir))
    logger.info("db_path=%s", str(db_path))
    logger.info("config_hash=%s", cfg_hash)

    print(f"OK: created backtest run {run_id} at {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
