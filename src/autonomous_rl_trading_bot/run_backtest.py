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
    parser.add_argument("--policy", default="baseline", help="Policy/strategy name (default: baseline)")
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

    # Build cfg in the exact structure expected by evaluation.backtest_runner (unit-tested).
    cfg: Dict[str, Any] = {
        "mode": {"id": args.mode},
        "run": {"seed": 1337},
        "logging": {"level": "INFO", "console": True},
        "db": {"path": str(artifacts / "db" / "bot.db")},
        "data": {"dataset": {"symbol": "BTCUSDT", "interval": "1m"}},
        "evaluation": {
            "backtest": {
                "dataset_id": dataset_id,
                "strategy": "buy_and_hold" if args.policy == "baseline" else str(args.policy),
                "strategies": {},
                "initial_cash": 1000.0,
                "order_size_quote": 0.0,
                "taker_fee_rate": 0.001,
                "slippage_bps": 0.0,
            }
        },
    }

    # Persist the effective run input (Step 12 reproducibility / V&V evidence)
    safe_json_dump(run_dir / "run_input.json", cfg)

    try:
        from autonomous_rl_trading_bot.evaluation.backtest_runner import run_backtest

        payload = run_backtest(
            cfg=cfg,
            artifacts_base_dir=artifacts,
            dataset_dir=dataset_dir,
            run_id=run_id,
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
