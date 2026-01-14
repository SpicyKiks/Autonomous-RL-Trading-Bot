from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from autonomous_rl_trading_bot.common.logging import configure_logging
from autonomous_rl_trading_bot.common.time import utc_now_compact
from autonomous_rl_trading_bot.common.utils import safe_json_dump


def _repo_root() -> Path:
    # src/autonomous_rl_trading_bot/run_backtest.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _artifacts_dir() -> Path:
    return _repo_root() / "artifacts"


def _find_latest_dataset_id(*, mode: str) -> str:
    """
    Choose the most recent dataset directory under artifacts/datasets that matches mode hint.
    This matches how your CLI behaved previously (auto-select dataset when not provided).
    """
    base = _artifacts_dir() / "datasets"
    if not base.exists():
        raise FileNotFoundError(f"No datasets directory found at: {base}")

    mode = mode.lower().strip()
    candidates = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        name = p.name.lower()
        if mode in name:
            candidates.append(p)

    if not candidates:
        # fallback: pick any dataset dir if none match
        candidates = [p for p in base.iterdir() if p.is_dir()]

    if not candidates:
        raise FileNotFoundError(f"No dataset directories found under: {base}")

    # Pick newest by modified time
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0].name


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="arbt backtest", description="Run backtest evaluation (baseline + reporting artifacts).")
    parser.add_argument("--mode", choices=["spot", "futures"], required=True)
    parser.add_argument("--policy", default=None, help="Path to SB3 policy.zip file, or baseline strategy name (default: baseline). If path ends with .zip, loads SB3 model.")
    parser.add_argument("--dataset-id", default=None, help="Dataset id under artifacts/datasets/. If omitted, auto-select latest.")
    parser.add_argument("--run-id", default=None, help="Optional explicit run id. If omitted, generated.")
    args = parser.parse_args(argv)

    artifacts = _artifacts_dir()
    artifacts.mkdir(parents=True, exist_ok=True)

    dataset_id = args.dataset_id or _find_latest_dataset_id(mode=args.mode)
    dataset_dir = artifacts / "datasets" / dataset_id
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    run_id = args.run_id or f"run_backtest_{args.mode}_{utc_now_compact()}_{dataset_id[:8]}"
    run_dir = artifacts / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(run_id=run_id, run_dir=run_dir, console=True, log_level="INFO")
    logger.info("Backtest starting")
    logger.info(f"run_id={run_id} market_type={args.mode} dataset_id={dataset_id} policy={args.policy}")

    # Determine policy argument: if --policy is a .zip file path, use it; otherwise treat as strategy name
    policy_arg = args.policy
    if policy_arg is None:
        policy_arg = "baseline"
    
    # Check if policy is a file path (ends with .zip)
    policy_path = None
    if policy_arg and policy_arg.endswith(".zip"):
        # Normalize path separators (handle Windows backslashes)
        policy_arg_normalized = policy_arg.replace("\\", "/")
        policy_path = Path(policy_arg_normalized)
        
        # If not absolute, try multiple locations
        if not policy_path.is_absolute():
            # Try relative to repo root first
            policy_path = _repo_root() / policy_arg_normalized
            if not policy_path.exists():
                # Try relative to current working directory
                policy_path = Path(policy_arg_normalized).resolve()
            if not policy_path.exists():
                # Try with original path (in case it's already relative to cwd)
                policy_path = Path(policy_arg).resolve()
        else:
            # Absolute path - just resolve it
            policy_path = policy_path.resolve()
            
        # If policy.zip doesn't exist, check for model.zip or best_model.zip in the same directory
        if not policy_path.exists():
            policy_dir = policy_path.parent
            policy_name = policy_path.name
            # Try model.zip or best_model.zip as fallback
            for alt_name in ["model.zip", "best_model.zip"]:
                alt_path = policy_dir / alt_name
                if alt_path.exists():
                    logger.info(f"Policy file {policy_path.name} not found, using {alt_name} instead")
                    policy_path = alt_path
                    break
            
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_arg} (resolved to: {policy_path})")
        policy_arg = str(policy_path)

    try:
        from autonomous_rl_trading_bot.evaluation.backtest_runner import run_backtest

        payload = run_backtest(
            mode=args.mode,
            dataset_id=dataset_id,
            run_id=run_id,
            policy=policy_arg if policy_arg != "baseline" else None,
            cfg=None,  # Let backtest_runner load cfg from dataset meta
        )

        # Mirror generated artifacts into run_dir for easy marking evidence
        from autonomous_rl_trading_bot.common.fs import copytree_merge

        produced_dir = artifacts / "backtests" / run_id
        if produced_dir.exists():
            copytree_merge(produced_dir, run_dir)

        safe_json_dump(run_dir / "run_output.json", payload)

        logger.info("Backtest DONE")
        logger.info(f"Artifacts: {run_dir}")
        return 0
    except Exception as e:
        # Ensure error is captured for marking evidence
        err_path = run_dir / "error.txt"
        err_path.write_text(f"{type(e).__name__}: {e}\n", encoding="utf-8")
        logger.exception("Backtest FAILED. See error.txt")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
