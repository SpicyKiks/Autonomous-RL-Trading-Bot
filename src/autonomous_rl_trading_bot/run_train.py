from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from autonomous_rl_trading_bot.common.config import load_config
from autonomous_rl_trading_bot.common.db import connect, migrate, upsert_run
from autonomous_rl_trading_bot.common.hashing import short_hash
from autonomous_rl_trading_bot.common.logging import configure_logging
from autonomous_rl_trading_bot.common.paths import artifacts_dir, ensure_artifact_tree
from autonomous_rl_trading_bot.common.reproducibility import set_global_seed
from autonomous_rl_trading_bot.rl.dataset import load_dataset_npz, select_latest_dataset
from autonomous_rl_trading_bot.training.trainer import TrainConfig, train_and_evaluate


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_run_id(mode: str, dataset_id: str, algo: str, cfg_hash: str) -> str:
    # Example: 20260109T195326Z_spot_train_ds123_ppo_ab12cd34
    return f"{_utc_ts()}_{mode}_train_{dataset_id}_{algo}_{short_hash(cfg_hash, 8)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 7: Offline RL training runner (SB3).")
    parser.add_argument("--mode", default=None, help="spot|futures. Overrides base.yaml.")
    parser.add_argument("--dataset-id", default=None, help="Dataset id under artifacts/datasets/ (default: latest for mode).")
    parser.add_argument("--algo", default="ppo", help="ppo|dqn")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total training timesteps.")
    parser.add_argument("--lookback", type=int, default=30, help="Lookback window for observations.")
    parser.add_argument(
        "--train-split",
        default=None,
        help="Dataset split for training: train|val|test (default: train from meta)",
    )
    parser.add_argument(
        "--eval-split",
        default=None,
        help="Dataset split for evaluation: train|val|test (default: val from meta)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override seed (default from config).")
    parser.add_argument("--fee-bps", type=float, default=10.0, help="Fee bps per trade.")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="Slippage bps per trade.")
    parser.add_argument("--position-fraction", type=float, default=1.0, help="Fraction of equity to size position.")
    parser.add_argument("--futures-leverage", type=float, default=3.0, help="Leverage used for futures sizing.")
    parser.add_argument("--reward", default="log_equity", help="log_equity|delta_equity")
    args = parser.parse_args(argv)

    ensure_artifact_tree()

    loaded = load_config(mode=args.mode)
    cfg = loaded.config
    cfg_hash = loaded.config_hash
    mode = cfg["mode"]["id"]
    if mode not in ("spot", "futures"):
        raise SystemExit(f"ERROR: mode must be spot|futures, got {mode}")

    # Ensure DB migrated before run creation
    db_path = migrate(cfg)

    # Dataset selection
    datasets_dir = artifacts_dir() / "datasets"
    if args.dataset_id:
        ds_dir = datasets_dir / args.dataset_id
        dataset = load_dataset_npz(ds_dir)
    else:
        dataset = select_latest_dataset(datasets_dir, market_type=mode)

    market_type = str(dataset.meta.get("market_type") or mode)
    dataset_id = dataset.dataset_id

    algo = str(args.algo).strip().lower()
    if algo not in ("ppo", "dqn"):
        raise SystemExit(f"ERROR: --algo must be ppo|dqn, got {algo}")

    run_id = _make_run_id(mode, dataset_id, algo, cfg_hash)
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
        logger = configure_logging(level=log_level, console=log_console, file_paths=None, run_id=run_id)

    seed = int(args.seed if args.seed is not None else int(cfg["run"]["seed"]))
    seed_report = set_global_seed(seed)

    created_utc = _iso_utc_now()
    run_json_path = str(run_dir / "run.json")

    train_split = str(args.train_split).strip().lower() if args.train_split else None
    eval_split = str(args.eval_split).strip().lower() if args.eval_split else None

    tcfg = TrainConfig(
        dataset_id=dataset_id,
        mode=mode,
        algo=algo,  # type: ignore[arg-type]
        timesteps=int(args.timesteps),
        seed=seed,
        train_split=train_split,
        eval_split=eval_split,
        lookback=int(args.lookback),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        initial_equity=1000.0,
        position_fraction=float(args.position_fraction),
        futures_leverage=float(args.futures_leverage),
        reward_kind=str(args.reward),
        run_dir=artifacts_dir() / "runs",
        models_dir=artifacts_dir() / "models",
    )

    run_meta: Dict[str, Any] = {
        "run_id": run_id,
        "created_utc": created_utc,
        "kind": "train",
        "mode": mode,
        "market_type": market_type,
        "dataset_id": dataset_id,
        "config_hash": cfg_hash,
        "config": cfg,
        "seed": seed,
        "seed_report": seed_report,
        "log_paths": log_paths,
        "db_path": str(db_path),
        "status": "CREATED",
        "train_params": {
            "algo": algo,
            "timesteps": int(args.timesteps),
            "lookback": int(args.lookback),
            "train_split": train_split,
            "eval_split": eval_split,
            "fee_bps": float(args.fee_bps),
            "slippage_bps": float(args.slippage_bps),
            "position_fraction": float(args.position_fraction),
            "futures_leverage": float(args.futures_leverage),
            "reward_kind": str(args.reward),
            "dataset_dir": str(dataset.dataset_dir),
            "npz_path": str(dataset.npz_path),
        },
    }
    (run_dir / "run.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # Record in DB: runs(CREATED)
    with connect(db_path) as conn:
        upsert_run(
            conn,
            run_id=run_id,
            kind="train",
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

    logger.info("Training starting")
    logger.info("run_id=%s market_type=%s dataset_id=%s algo=%s timesteps=%s", run_id, market_type, dataset_id, algo, tcfg.timesteps)

    status = "DONE"
    finished_utc: Optional[str] = None
    try:
        with connect(db_path) as conn:
            upsert_run(
                conn,
                run_id=run_id,
                kind="train",
                mode=mode,
                created_utc=created_utc,
                config_hash=cfg_hash,
                seed=seed,
                status="RUNNING",
                run_dir=str(run_dir),
                run_json_path=run_json_path,
                run_log_path=per_run_log,
                global_log_path=global_log,
            )
            conn.commit()

        out = train_and_evaluate(dataset=dataset, cfg=tcfg, run_id=run_id, out_dir=run_dir)
        finished_utc = _iso_utc_now()

        run_meta["status"] = "DONE"
        run_meta["finished_utc"] = finished_utc
        run_meta["train_output"] = out
        (run_dir / "run.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info("Training DONE")
        logger.info("Model: %s", out.get("model_path"))
        logger.info("Metrics: %s", out.get("metrics"))

    except Exception:
        status = "FAILED"
        finished_utc = _iso_utc_now()
        tb = traceback.format_exc()
        (run_dir / "error.txt").write_text(tb, encoding="utf-8")
        run_meta["status"] = "FAILED"
        run_meta["finished_utc"] = finished_utc
        (run_dir / "run.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.error("Training FAILED. See error.txt")
        logger.error(tb)

    with connect(db_path) as conn:
        upsert_run(
            conn,
            run_id=run_id,
            kind="train",
            mode=mode,
            created_utc=created_utc,
            config_hash=cfg_hash,
            seed=seed,
            status=status,
            run_dir=str(run_dir),
            run_json_path=run_json_path,
            run_log_path=per_run_log,
            global_log_path=global_log,
        )
        conn.commit()

    if status == "DONE":
        print(f"OK: training {run_id} DONE (mode={mode}, dataset_id={dataset_id}, run_dir={run_dir})")
        return 0
    print(f"ERROR: training {run_id} FAILED (see {run_dir}/error.txt)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
