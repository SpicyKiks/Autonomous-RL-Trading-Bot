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
from autonomous_rl_trading_bot.common.timeframes import interval_to_ms
from autonomous_rl_trading_bot.data.candles_store import insert_candles, write_candles_csv
from autonomous_rl_trading_bot.exchange.binance_public import Candle, fetch_klines


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _make_run_id(mode: str, cfg_hash: str, symbol: str, interval: str) -> str:
    tag = f"{symbol}_{interval}"
    return f"{_utc_ts()}_{mode}_data_{tag}_{short_hash(cfg_hash, 10)}"


def _market_type_from_mode(mode: str) -> str:
    m = (mode or "").strip().lower()
    if m in ("spot", "futures"):
        return m
    # default fallback: treat unknown as spot
    return "spot"


def _download(
    cfg: Dict[str, Any],
    *,
    market_type: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int,
) -> List[Candle]:
    """
    Loop batches until end_ms reached (best-effort).
    """
    out: List[Candle] = []
    interval_ms = interval_to_ms(interval)
    cursor = int(start_ms)

    while cursor < end_ms:
        batch = fetch_klines(
            cfg,
            market_type=market_type,
            symbol=symbol,
            interval=interval,
            start_time_ms=cursor,
            end_time_ms=end_ms,
            limit=limit,
        )
        if not batch:
            break

        out.extend(batch)

        last_open = batch[-1].open_time_ms
        next_cursor = last_open + interval_ms
        if next_cursor <= cursor:
            # safety against infinite loop
            break

        cursor = next_cursor

        # If we got fewer than limit, likely no more data in that window
        if len(batch) < limit:
            break

        # Gentle pacing
        time.sleep(0.25)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Binance klines and store into SQLite (Step 3)."
    )
    parser.add_argument(
        "--mode", default=None, help="spot|futures (uses configs/modes/* overrides)."
    )
    parser.add_argument("--symbol", default=None, help="e.g. BTCUSDT")
    parser.add_argument("--interval", default=None, help="e.g. 1m, 1h")
    parser.add_argument("--limit", type=int, default=None, help="batch limit (<=1000)")
    parser.add_argument(
        "--minutes", type=int, default=None, help="lookback minutes (default from config)"
    )
    args = parser.parse_args()

    ensure_artifact_tree()

    loaded = load_config(mode=args.mode)
    cfg = loaded.config
    cfg_hash = loaded.config_hash
    mode = cfg["mode"]["id"]
    market_type = _market_type_from_mode(mode)

    exch_defaults = (cfg.get("exchange", {}) or {}).get("defaults", {}) or {}
    symbol = (args.symbol or exch_defaults.get("symbol") or "BTCUSDT").upper()
    interval = (args.interval or exch_defaults.get("interval") or "1m").strip()
    limit = int(args.limit or exch_defaults.get("limit") or 1000)
    limit = max(1, min(1000, limit))

    lookback = int(
        args.minutes
        or (((cfg.get("data", {}) or {}).get("candles", {}) or {}).get("lookback_minutes", 180))
    )

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
            kind="data",
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

    end_ms = _now_ms()
    start_ms = end_ms - (lookback * 60_000)

    logger.info("Step 3 data fetch starting.")
    logger.info(
        "market_type=%s symbol=%s interval=%s lookback_minutes=%s limit=%s",
        market_type,
        symbol,
        interval,
        lookback,
        limit,
    )
    logger.info("db_path=%s", str(db_path))

    candles = _download(
        cfg,
        market_type=market_type,
        symbol=symbol,
        interval=interval,
        start_ms=start_ms,
        end_ms=end_ms,
        limit=limit,
    )

    inserted_attempted = 0
    with connect(db_path) as conn:
        inserted_attempted = insert_candles(conn, candles)
        conn.commit()

    csv_path = run_dir / f"candles_{market_type}_{symbol}_{interval}.csv"
    write_candles_csv(csv_path, candles)

    run_meta: Dict[str, Any] = {
        "run_id": run_id,
        "created_utc": created_utc,
        "kind": "data",
        "mode": mode,
        "market_type": market_type,
        "symbol": symbol,
        "interval": interval,
        "lookback_minutes": lookback,
        "limit": limit,
        "candles_fetched": len(candles),
        "candles_insert_attempted": inserted_attempted,
        "config_hash": cfg_hash,
        "config": cfg,
        "seed_report": seed_report,
        "log_paths": log_paths,
        "db_path": str(db_path),
        "csv_path": str(csv_path),
        "status": "DONE",
    }
    Path(run_json_path).write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with connect(db_path) as conn:
        upsert_run(
            conn,
            run_id=run_id,
            kind="data",
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

    logger.info(
        "DONE: fetched=%s inserted_attempted=%s csv=%s",
        len(candles),
        inserted_attempted,
        str(csv_path),
    )
    print(f"OK: fetched {len(candles)} candles and stored to DB. CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
