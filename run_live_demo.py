from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from autonomous_rl_trading_bot.common.config import load_config
from autonomous_rl_trading_bot.common.db import connect, migrate, upsert_run
from autonomous_rl_trading_bot.common.hashing import short_hash
from autonomous_rl_trading_bot.common.logging import configure_logging
from autonomous_rl_trading_bot.common.paths import artifacts_dir, ensure_artifact_tree
from autonomous_rl_trading_bot.common.reproducibility import set_global_seed
from autonomous_rl_trading_bot.live.live_runner import LiveRunner, LiveRunnerConfig


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _make_run_id(mode: str, cfg_hash: str, symbol: str, interval: str) -> str:
    tag = f"{symbol}_{interval}"
    return f"{_utc_ts()}_{mode}_live_{tag}_{short_hash(cfg_hash, 10)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run live (paper) trading on CLOSED candles (Step 7)."
    )
    parser.add_argument(
        "--mode", default=None, help="spot|futures (uses configs/modes/* overrides)."
    )
    parser.add_argument("--symbol", default=None, help="e.g. BTCUSDT")
    parser.add_argument("--interval", default=None, help="e.g. 1m, 5m, 1h")
    parser.add_argument(
        "--policy",
        default=None,
        help="baseline|sb3 (default from config/live).",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="baseline strategy: buy_and_hold|sma_crossover|ema_crossover|rsi_reversion",
    )
    parser.add_argument("--sb3_algo", default=None, help="ppo|dqn")
    parser.add_argument("--sb3_model_path", default=None, help="path to SB3 model zip")
    parser.add_argument("--max_steps", type=int, default=None, help="stop after N candles")
    parser.add_argument(
        "--max_minutes",
        type=float,
        default=None,
        help="stop after N minutes wall-clock",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Enable Binance Demo trading (requires BINANCE_DEMO_API_KEY and BINANCE_DEMO_API_SECRET)",
    )
    args = parser.parse_args(argv)

    ensure_artifact_tree()

    loaded = load_config(mode=args.mode)
    cfg = loaded.config
    cfg_hash = loaded.config_hash
    mode = cfg["mode"]["id"]

    exch_defaults = (cfg.get("exchange", {}) or {}).get("defaults", {}) or {}
    symbol = (args.symbol or exch_defaults.get("symbol") or "BTCUSDT").upper()
    interval = (args.interval or exch_defaults.get("interval") or "1m").strip()

    live_cfg = cfg.get("live", {}) or {}
    eval_cfg = (cfg.get("evaluation", {}) or {}).get("backtest", {}) or {}

    policy = (args.policy or live_cfg.get("policy") or "baseline").strip()

    strategy = (
        args.strategy
        or live_cfg.get("strategy")
        or eval_cfg.get("strategy")
        or "buy_and_hold"
    )

    strat_params: Dict[str, Any] = {}
    strat_block = (eval_cfg.get("strategies", {}) or {}).get(strategy, {})
    if isinstance(strat_block, dict):
        strat_params.update(strat_block)
    user_params = live_cfg.get("strategy_params")
    if isinstance(user_params, dict):
        strat_params.update(user_params)

    db_path = migrate(cfg)

    run_id = _make_run_id(mode, cfg_hash, symbol, interval)
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

    with connect(db_path) as conn:
        upsert_run(
            conn,
            run_id=run_id,
            kind="live",
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

    fee_bps = float(
        live_cfg.get("fee_bps", float(eval_cfg.get("taker_fee_rate", 0.001)) * 10000.0)
    )
    slippage_bps = float(live_cfg.get("slippage_bps", float(eval_cfg.get("slippage_bps", 5))))
    initial_equity = float(live_cfg.get("initial_equity", float(eval_cfg.get("initial_cash", 1000.0))))
    position_fraction = float(live_cfg.get("position_fraction", 1.0))

    futures_leverage = float(live_cfg.get("futures_leverage", float(eval_cfg.get("leverage", 3.0))))
    maintenance_margin_rate = float(
        live_cfg.get("maintenance_margin_rate", float(eval_cfg.get("maintenance_margin_rate", 0.005)))
    )
    stop_on_liquidation = bool(
        live_cfg.get("stop_on_liquidation", bool(eval_cfg.get("stop_on_liquidation", True)))
    )

    kill_switch_path = str(
        live_cfg.get("kill_switch_path") or (artifacts_dir() / "KILL_SWITCH").as_posix()
    )

    runner_cfg = LiveRunnerConfig(
        market_type=mode,
        symbol=symbol,
        interval=interval,
        policy=policy,
        strategy=strategy,
        strategy_params=strat_params,
        sb3_algo=str(args.sb3_algo or live_cfg.get("sb3_algo") or "ppo"),
        sb3_model_path=str(args.sb3_model_path or live_cfg.get("sb3_model_path") or ""),
        initial_equity=initial_equity,
        position_fraction=position_fraction,
        futures_leverage=futures_leverage,
        maintenance_margin_rate=maintenance_margin_rate,
        stop_on_liquidation=stop_on_liquidation,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        lookback=int(live_cfg.get("lookback", 30)),
        poll_seconds=float(live_cfg.get("poll_seconds", 2.0)),
        require_settled_ms=int(live_cfg.get("require_settled_ms", 250)),
        kill_switch_path=kill_switch_path,
        max_drawdown=float(live_cfg.get("max_drawdown", 0.25)),
        min_seconds_between_trades=float(live_cfg.get("min_seconds_between_trades", 15.0)),
        max_trades_per_hour=int(live_cfg.get("max_trades_per_hour", 30)),
        db_path=str(db_path),
        run_id=run_id,
        run_dir=str(run_dir),
        demo=bool(args.demo),
    )

    logger.info("Step 7 live runner starting")
    logger.info("mode=%s symbol=%s interval=%s policy=%s", mode, symbol, interval, policy)
    logger.info("kill_switch_path=%s", kill_switch_path)
    logger.info("db_path=%s", str(db_path))

    runner = LiveRunner(cfg, runner_cfg)
    out = runner.run(max_steps=args.max_steps, max_minutes=args.max_minutes)

    meta: Dict[str, Any] = {
        "run_id": run_id,
        "created_utc": created_utc,
        "kind": "live",
        "mode": mode,
        "symbol": symbol,
        "interval": interval,
        "policy": policy,
        "strategy": strategy,
        "strategy_params": strat_params,
        "config_hash": cfg_hash,
        "config": cfg,
        "seed_report": seed_report,
        "log_paths": log_paths,
        "db_path": str(db_path),
        "kill_switch_path": kill_switch_path,
        "result": out,
    }
    Path(run_json_path).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    with connect(db_path) as conn:
        upsert_run(
            conn,
            run_id=run_id,
            kind="live",
            mode=mode,
            created_utc=created_utc,
            config_hash=cfg_hash,
            seed=seed,
            status=str(out.get("status") or "DONE"),
            run_dir=str(run_dir),
            run_json_path=run_json_path,
            run_log_path=per_run_log,
            global_log_path=global_log,
        )
        conn.commit()

    print(
        f"OK: live run finished. run_id={run_id} status={out.get('status')} trades={out.get('trade_count')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
