from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from autonomous_rl_trading_bot.common.timeframes import interval_to_ms
from autonomous_rl_trading_bot.evaluation.baselines import Strategy
from autonomous_rl_trading_bot.evaluation.metrics import BacktestMetrics, compute_metrics


@dataclass(frozen=True)
class BacktestConfig:
    # shared
    initial_cash: float
    order_size_quote: float  # 0 => all-in / close-all
    taker_fee_rate: float
    slippage_bps: float

    # futures-only (ignored for spot)
    leverage: float = 1.0
    maintenance_margin_rate: float = 0.005
    allow_short: bool = True
    stop_on_liquidation: bool = True


@dataclass
class SpotState:
    cash: float
    qty_base: float
    avg_entry_price: float
    peak_equity: float


@dataclass
class FuturesState:
    collateral: float
    position_qty: float  # +long, -short
    entry_price: float  # avg entry
    peak_equity: float
    liquidated: bool


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_dataset(dataset_dir: Path, split: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Load meta.json + dataset.npz from a dataset directory.
    
    Args:
        dataset_dir: Directory containing meta.json and dataset.npz
        split: Optional split name ('train', 'val', 'test') to filter data
    
    Returns:
        Tuple of (meta, arrays) with potentially filtered arrays based on split
    """
    meta_path = dataset_dir / "meta.json"
    npz_path = dataset_dir / "dataset.npz"

    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in dataset dir: {dataset_dir}")
    if not npz_path.exists():
        raise FileNotFoundError(f"dataset.npz not found in dataset dir: {dataset_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    arrays = dict(np.load(npz_path, allow_pickle=False))

    required = ["open_time_ms", "close"]
    for k in required:
        if k not in arrays:
            raise ValueError(f"dataset.npz missing required key: {k}")
    
    # Filter by split if requested
    if split is not None:
        splits = meta.get("splits")
        if not splits:
            raise ValueError(f"Dataset does not have splits metadata, cannot filter by split={split}")
        if split not in splits:
            raise ValueError(f"Unknown split: {split}. Available: {list(splits.keys())}")
        
        split_info = splits[split]
        start_idx = split_info["start_idx"]
        end_idx = split_info["end_idx"]
        
        # Filter all arrays
        arrays = {k: v[start_idx:end_idx] for k, v in arrays.items()}

    return meta, arrays


def _calc_drawdown(equity: float, peak: float) -> float:
    if peak <= 0.0:
        return 0.0
    return max(0.0, 1.0 - (equity / peak))


def _slippage_frac(slippage_bps: float) -> float:
    return max(0.0, float(slippage_bps) / 10_000.0)


# ─────────────────────────────────────────────────────────────
# SPOT engine (unchanged behavior)
# ─────────────────────────────────────────────────────────────
def _execute_buy_spot(
    *,
    state: SpotState,
    price: float,
    open_time_ms: int,
    cfg: BacktestConfig,
) -> Optional[Dict[str, Any]]:
    if state.cash <= 0.0:
        return None

    slip = _slippage_frac(cfg.slippage_bps)
    fill_price = price * (1.0 + slip)

    desired_notional = (
        state.cash if cfg.order_size_quote <= 0.0 else min(state.cash, cfg.order_size_quote)
    )

    fee_rate = max(0.0, float(cfg.taker_fee_rate))
    notional = min(desired_notional, state.cash / (1.0 + fee_rate))
    if notional <= 0.0:
        return None

    fee = notional * fee_rate
    qty_add = notional / fill_price
    slippage_cost = max(0.0, (fill_price - price) * qty_add)

    prev_qty = state.qty_base
    prev_avg = state.avg_entry_price
    new_qty = prev_qty + qty_add

    if new_qty > 0.0:
        if prev_qty <= 0.0:
            new_avg = fill_price
        else:
            new_avg = (prev_qty * prev_avg + qty_add * fill_price) / new_qty
    else:
        new_avg = 0.0

    state.cash -= notional + fee
    state.qty_base = new_qty
    state.avg_entry_price = new_avg

    equity_after = state.cash + state.qty_base * price

    return {
        "open_time_ms": int(open_time_ms),
        "side": "BUY",
        "qty_base": float(qty_add),
        "price": float(fill_price),
        "notional": float(notional),
        "fee": float(fee),
        "slippage_cost": float(slippage_cost),
        "reason": "strategy",
        "cash_after": float(state.cash),
        "qty_after": float(state.qty_base),
        "equity_after": float(equity_after),
    }


def _execute_sell_spot(
    *,
    state: SpotState,
    price: float,
    open_time_ms: int,
    cfg: BacktestConfig,
    close_all: bool,
) -> Optional[Dict[str, Any]]:
    if state.qty_base <= 0.0:
        return None

    slip = _slippage_frac(cfg.slippage_bps)
    fill_price = price * (1.0 - slip)

    fee_rate = max(0.0, float(cfg.taker_fee_rate))

    if close_all or cfg.order_size_quote <= 0.0:
        qty_to_sell = state.qty_base
    else:
        desired_notional = float(cfg.order_size_quote)
        qty_to_sell = min(state.qty_base, desired_notional / fill_price)

    if qty_to_sell <= 0.0:
        return None

    notional = qty_to_sell * fill_price
    fee = notional * fee_rate
    slippage_cost = max(0.0, (price - fill_price) * qty_to_sell)

    state.cash += notional - fee
    state.qty_base -= qty_to_sell
    if state.qty_base <= 1e-12:
        state.qty_base = 0.0
        state.avg_entry_price = 0.0

    equity_after = state.cash + state.qty_base * price

    return {
        "open_time_ms": int(open_time_ms),
        "side": "SELL",
        "qty_base": float(qty_to_sell),
        "price": float(fill_price),
        "notional": float(notional),
        "fee": float(fee),
        "slippage_cost": float(slippage_cost),
        "reason": "strategy" if not close_all else "close_end",
        "cash_after": float(state.cash),
        "qty_after": float(state.qty_base),
        "equity_after": float(equity_after),
    }


def run_spot_backtest(
    *,
    dataset_meta: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    strategy: Strategy,
    cfg: BacktestConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], BacktestMetrics, Dict[str, Any]]:
    times = np.asarray(arrays["open_time_ms"], dtype=np.int64)
    close = np.asarray(arrays["close"], dtype=np.float64)

    if times.ndim != 1 or close.ndim != 1 or len(times) != len(close):
        raise ValueError("dataset arrays must be 1D and aligned")
    if len(times) < 2:
        raise ValueError("dataset must have at least 2 rows")

    interval = str(dataset_meta.get("interval") or "")
    interval_ms = interval_to_ms(interval) if interval else int(times[1] - times[0])

    state = SpotState(
        cash=float(cfg.initial_cash),
        qty_base=0.0,
        avg_entry_price=0.0,
        peak_equity=float(cfg.initial_cash),
    )

    equity_rows: List[Dict[str, Any]] = []
    trade_rows: List[Dict[str, Any]] = []

    fee_total = 0.0
    slippage_total = 0.0

    for t in range(len(times)):
        ot = int(times[t])
        price = float(close[t])

        action = int(strategy.act(t, price))
        trade: Optional[Dict[str, Any]] = None

        if action == 1:
            trade = _execute_buy_spot(state=state, price=price, open_time_ms=ot, cfg=cfg)
        elif action == 2:
            trade = _execute_sell_spot(
                state=state, price=price, open_time_ms=ot, cfg=cfg, close_all=False
            )
        elif action == 3:
            trade = _execute_sell_spot(
                state=state, price=price, open_time_ms=ot, cfg=cfg, close_all=True
            )

        equity = float(state.cash + state.qty_base * price)
        state.peak_equity = max(state.peak_equity, equity)
        dd = _calc_drawdown(equity, state.peak_equity)
        exposure = float((state.qty_base * price) / equity) if equity > 0.0 else 0.0

        equity_rows.append(
            {
                "open_time_ms": ot,
                "price": price,
                "cash": float(state.cash),
                "qty_base": float(state.qty_base),
                "equity": equity,
                "drawdown": float(dd),
                "exposure": exposure,
            }
        )

        if trade is not None:
            trade_id = len(trade_rows) + 1
            trade_rows.append({"trade_id": trade_id, **trade})
            fee_total += float(trade["fee"])
            slippage_total += float(trade["slippage_cost"])

    # Force-close at end
    if state.qty_base > 0.0:
        ot = int(times[-1])
        price = float(close[-1])
        trade = _execute_sell_spot(
            state=state, price=price, open_time_ms=ot, cfg=cfg, close_all=True
        )
        if trade is not None:
            trade_id = len(trade_rows) + 1
            trade_rows.append({"trade_id": trade_id, **trade})
            fee_total += float(trade["fee"])
            slippage_total += float(trade["slippage_cost"])

            equity = float(state.cash + state.qty_base * price)
            peak = max(r["equity"] for r in equity_rows) if equity_rows else equity
            dd = _calc_drawdown(equity, float(peak))
            equity_rows[-1] = {
                "open_time_ms": ot,
                "price": price,
                "cash": float(state.cash),
                "qty_base": float(state.qty_base),
                "equity": equity,
                "drawdown": float(dd),
                "exposure": 0.0,
            }

    metrics = compute_metrics(
        open_time_ms=[int(r["open_time_ms"]) for r in equity_rows],
        equity=[float(r["equity"]) for r in equity_rows],
        drawdown=[float(r["drawdown"]) for r in equity_rows],
        trade_count=len(trade_rows),
        fee_total=fee_total,
        slippage_total=slippage_total,
        interval_ms=int(interval_ms),
        trades=trade_rows,
        exposure=[float(r.get("exposure", 0.0)) for r in equity_rows],
        seed=None,  # Will use global seed if set
    )

    extra = {
        "interval_ms": int(interval_ms),
        "fee_total": float(fee_total),
        "slippage_total": float(slippage_total),
    }
    return equity_rows, trade_rows, metrics, extra


def persist_backtest_to_db(
    *,
    conn: sqlite3.Connection,
    backtest_id: str,
    run_id: str,
    mode: str,
    dataset_meta: Mapping[str, Any],
    cfg: BacktestConfig,
    params_json: str,
    metrics: BacktestMetrics,
    equity_rows: List[Dict[str, Any]],
    trade_rows: List[Dict[str, Any]],
    started_utc: str,
    finished_utc: str,
    status: str,
) -> None:
    dataset_id = str(dataset_meta.get("dataset_id") or "")
    market_type = str(dataset_meta.get("market_type") or mode)
    symbol = str(dataset_meta.get("symbol") or "")
    interval = str(dataset_meta.get("interval") or "")

    conn.execute(
        """
        INSERT INTO backtests(
          backtest_id, run_id, mode, dataset_id, market_type, symbol, interval,
          started_utc, finished_utc, status,
          initial_cash, final_equity, total_return, max_drawdown, trade_count,
          fee_total, slippage_total, params_json, metrics_json
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            backtest_id,
            run_id,
            mode,
            dataset_id,
            market_type,
            symbol,
            interval,
            started_utc,
            finished_utc,
            status,
            float(metrics.initial_cash),
            float(metrics.final_equity),
            float(metrics.total_return),
            float(metrics.max_drawdown),
            int(metrics.trade_count),
            float(metrics.fee_total),
            float(metrics.slippage_total),
            params_json,
            json.dumps(metrics.to_dict(), ensure_ascii=False),
        ),
    )

    conn.executemany(
        """
        INSERT INTO backtest_equity(
          backtest_id, open_time_ms, price, cash, qty_base, equity, drawdown, exposure
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?);
        """,
        [
            (
                backtest_id,
                int(r["open_time_ms"]),
                float(r["price"]),
                float(r["cash"]),
                float(r["qty_base"]),
                float(r["equity"]),
                float(r["drawdown"]),
                float(r["exposure"]),
            )
            for r in equity_rows
        ],
    )

    if trade_rows:
        conn.executemany(
            """
            INSERT INTO backtest_trades(
              backtest_id, open_time_ms, side, qty_base, price, notional, fee,
              slippage_cost, reason, cash_after, qty_after, equity_after
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                (
                    backtest_id,
                    int(r["open_time_ms"]),
                    str(r["side"]),
                    float(r["qty_base"]),
                    float(r["price"]),
                    float(r["notional"]),
                    float(r["fee"]),
                    float(r["slippage_cost"]),
                    str(r.get("reason") or ""),
                    float(r["cash_after"]),
                    float(r["qty_after"]),
                    float(r["equity_after"]),
                )
                for r in trade_rows
            ],
        )


# ─────────────────────────────────────────────────────────────
# FUTURES engine (new)
# ─────────────────────────────────────────────────────────────
def _unrealized_pnl(price: float, entry: float, qty: float) -> float:
    return (price - entry) * qty


def _cap_delta_to_leverage(
    *,
    equity: float,
    price: float,
    current_qty: float,
    desired_delta_qty: float,
    leverage: float,
) -> float:
    """Cap delta so that resulting notional <= equity*leverage."""
    lev = max(1e-9, float(leverage))
    max_notional = max(0.0, equity) * lev
    target_qty = current_qty + desired_delta_qty
    target_notional = abs(target_qty) * price

    if target_notional <= max_notional + 1e-12:
        return desired_delta_qty

    max_qty = (max_notional / price) if price > 0.0 else 0.0
    if target_qty > 0:
        capped_target = max_qty
    elif target_qty < 0:
        capped_target = -max_qty
    else:
        capped_target = 0.0

    return capped_target - current_qty


def _apply_futures_delta(
    *,
    state: FuturesState,
    mid_price: float,
    fill_price: float,
    delta_qty: float,
    fee_rate: float,
    open_time_ms: int,
    slippage_cost: float,
    reason: str,
) -> Optional[Dict[str, Any]]:
    if abs(delta_qty) <= 0.0:
        return None

    prev_qty = state.position_qty
    prev_entry = state.entry_price

    # Notional traded (abs)
    notional = abs(delta_qty) * fill_price
    fee = notional * fee_rate

    realized = 0.0
    new_qty = prev_qty + delta_qty
    new_entry = prev_entry

    # If we are reducing/closing (delta opposes prev)
    if prev_qty != 0.0 and (math.copysign(1.0, prev_qty) != math.copysign(1.0, delta_qty)):
        close_qty = min(abs(delta_qty), abs(prev_qty))
        close_signed = close_qty * math.copysign(1.0, prev_qty)
        # Realized PnL on closed portion:
        realized = (fill_price - prev_entry) * close_signed
        remaining_qty = prev_qty + delta_qty

        if abs(remaining_qty) <= 1e-12:
            remaining_qty = 0.0
            new_entry = 0.0
        else:
            # If we crossed through zero, the remaining part is a new position opened at fill_price
            if math.copysign(1.0, remaining_qty) != math.copysign(1.0, prev_qty):
                new_entry = fill_price
            else:
                new_entry = prev_entry

        new_qty = remaining_qty

    # If we are adding in same direction (or from flat)
    elif prev_qty == 0.0 or (math.copysign(1.0, prev_qty) == math.copysign(1.0, delta_qty)):
        if prev_qty == 0.0:
            new_entry = fill_price
        else:
            # Weighted avg entry
            new_entry = (abs(prev_qty) * prev_entry + abs(delta_qty) * fill_price) / (
                abs(prev_qty) + abs(delta_qty)
            )

    # Update collateral: realized pnl - fee
    state.collateral = float(state.collateral + realized - fee)
    state.position_qty = float(new_qty)
    state.entry_price = float(new_entry)

    equity_after = float(
        state.collateral
        + (
            _unrealized_pnl(mid_price, state.entry_price, state.position_qty)
            if state.position_qty != 0
            else 0.0
        )
    )

    side = "BUY" if delta_qty > 0 else "SELL"
    return {
        "open_time_ms": int(open_time_ms),
        "side": side,
        "qty": float(delta_qty),
        "fill_price": float(fill_price),
        "notional": float(notional),
        "fee": float(fee),
        "slippage_cost": float(slippage_cost),
        "realized_pnl": float(realized),
        "reason": reason,
        "collateral_after": float(state.collateral),
        "position_qty_after": float(state.position_qty),
        "entry_price_after": float(state.entry_price),
        "equity_after": float(equity_after),
    }


def run_futures_backtest(
    *,
    dataset_meta: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    strategy: Strategy,
    cfg: BacktestConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], BacktestMetrics, Dict[str, Any]]:
    times = np.asarray(arrays["open_time_ms"], dtype=np.int64)
    close = np.asarray(arrays["close"], dtype=np.float64)

    if times.ndim != 1 or close.ndim != 1 or len(times) != len(close):
        raise ValueError("dataset arrays must be 1D and aligned")
    if len(times) < 2:
        raise ValueError("dataset must have at least 2 rows")

    interval = str(dataset_meta.get("interval") or "")
    interval_ms = interval_to_ms(interval) if interval else int(times[1] - times[0])

    fee_rate = max(0.0, float(cfg.taker_fee_rate))
    slip = _slippage_frac(cfg.slippage_bps)

    state = FuturesState(
        collateral=float(cfg.initial_cash),
        position_qty=0.0,
        entry_price=0.0,
        peak_equity=float(cfg.initial_cash),
        liquidated=False,
    )

    equity_rows: List[Dict[str, Any]] = []
    trade_rows: List[Dict[str, Any]] = []

    fee_total = 0.0
    slippage_total = 0.0
    liquidated = False

    allow_short = bool(cfg.allow_short)
    mmr = max(0.0, float(cfg.maintenance_margin_rate))
    lev = max(1e-9, float(cfg.leverage))

    for t in range(len(times)):
        ot = int(times[t])
        price = float(close[t])  # use close as mark/mid

        # Compute equity pre-trade for leverage cap
        unpnl = (
            _unrealized_pnl(price, state.entry_price, state.position_qty)
            if state.position_qty != 0
            else 0.0
        )
        equity = float(state.collateral + unpnl)

        action = int(strategy.act(t, price))
        trade: Optional[Dict[str, Any]] = None

        # Determine desired delta qty
        delta_qty = 0.0
        if action in (1, 2, 3):
            # close action
            if action == 3:
                delta_qty = -state.position_qty
            else:
                # order_size_quote=0 => "all-in" toward max notional allowed
                fill_price = price * (1.0 + slip) if action == 1 else price * (1.0 - slip)

                if cfg.order_size_quote <= 0.0:
                    # target max notional by leverage; if already positioned, push toward cap in action direction
                    max_notional = max(0.0, equity) * lev
                    max_qty = (max_notional / fill_price) if fill_price > 0.0 else 0.0
                    desired_qty = max_qty if action == 1 else (-max_qty if allow_short else 0.0)
                    delta_qty = desired_qty - state.position_qty
                else:
                    notional = float(cfg.order_size_quote)
                    qty = (notional / fill_price) if fill_price > 0.0 else 0.0
                    delta_qty = qty if action == 1 else (-qty if allow_short else 0.0)

        if abs(delta_qty) > 0.0:
            # Cap by leverage using current equity and mid price (conservative)
            capped_delta = _cap_delta_to_leverage(
                equity=equity,
                price=price,
                current_qty=state.position_qty,
                desired_delta_qty=delta_qty,
                leverage=lev,
            )

            # If shorts disabled, prevent going negative
            if not allow_short and (state.position_qty + capped_delta) < 0.0:
                capped_delta = -state.position_qty  # flatten only

            if abs(capped_delta) > 0.0:
                # Fill price & slippage cost
                if capped_delta > 0:
                    fill_price = price * (1.0 + slip)
                    slippage_cost = max(0.0, (fill_price - price) * abs(capped_delta))
                    trade = _apply_futures_delta(
                        state=state,
                        mid_price=price,
                        fill_price=fill_price,
                        delta_qty=capped_delta,
                        fee_rate=fee_rate,
                        open_time_ms=ot,
                        slippage_cost=slippage_cost,
                        reason="strategy" if action != 3 else "close",
                    )
                else:
                    fill_price = price * (1.0 - slip)
                    slippage_cost = max(0.0, (price - fill_price) * abs(capped_delta))
                    trade = _apply_futures_delta(
                        state=state,
                        mid_price=price,
                        fill_price=fill_price,
                        delta_qty=capped_delta,
                        fee_rate=fee_rate,
                        open_time_ms=ot,
                        slippage_cost=slippage_cost,
                        reason="strategy" if action != 3 else "close",
                    )

        # Mark-to-market after trade
        unpnl = (
            _unrealized_pnl(price, state.entry_price, state.position_qty)
            if state.position_qty != 0
            else 0.0
        )
        equity = float(state.collateral + unpnl)

        # Liquidation check
        notional_now = abs(state.position_qty) * price
        maint_margin = mmr * notional_now
        if abs(state.position_qty) > 0.0 and equity <= maint_margin:
            liquidated = True
            state.liquidated = True
            # stop immediately; record final snapshot then break
        state.peak_equity = max(state.peak_equity, equity)
        dd = _calc_drawdown(equity, state.peak_equity)

        margin_used = (notional_now / lev) if lev > 0 else 0.0
        leverage_used = (notional_now / equity) if equity > 0.0 else float("inf")
        exposure = (notional_now / equity) if equity > 0.0 else 0.0

        equity_rows.append(
            {
                "open_time_ms": ot,
                "price": price,
                "collateral": float(state.collateral),
                "position_qty": float(state.position_qty),
                "entry_price": float(state.entry_price),
                "unrealized_pnl": float(unpnl),
                "equity": float(equity),
                "drawdown": float(dd),
                "notional": float(notional_now),
                "margin_used": float(margin_used),
                "leverage_used": float(leverage_used),
                "exposure": float(exposure),
            }
        )

        if trade is not None:
            trade_id = len(trade_rows) + 1
            trade_rows.append({"trade_id": trade_id, **trade})
            fee_total += float(trade["fee"])
            slippage_total += float(trade["slippage_cost"])

        if liquidated and cfg.stop_on_liquidation:
            break

    # Force close at end if not liquidated
    if (not state.liquidated) and state.position_qty != 0.0:
        ot = int(times[min(len(times) - 1, len(equity_rows) - 1)])
        price = float(close[min(len(times) - 1, len(equity_rows) - 1)])
        delta_qty = -state.position_qty
        if abs(delta_qty) > 0.0:
            fill_price = price * (1.0 - slip) if delta_qty < 0 else price * (1.0 + slip)
            slippage_cost = max(0.0, abs(fill_price - price) * abs(delta_qty))
            trade = _apply_futures_delta(
                state=state,
                mid_price=price,
                fill_price=fill_price,
                delta_qty=delta_qty,
                fee_rate=fee_rate,
                open_time_ms=ot,
                slippage_cost=slippage_cost,
                reason="close_end",
            )
            if trade is not None:
                trade_id = len(trade_rows) + 1
                trade_rows.append({"trade_id": trade_id, **trade})
                fee_total += float(trade["fee"])
                slippage_total += float(trade["slippage_cost"])

            # update last equity row to reflect flat
            unpnl = 0.0
            equity = float(state.collateral)
            peak = max(r["equity"] for r in equity_rows) if equity_rows else equity
            dd = _calc_drawdown(equity, float(peak))
            equity_rows[-1] = {
                "open_time_ms": ot,
                "price": price,
                "collateral": float(state.collateral),
                "position_qty": float(state.position_qty),
                "entry_price": float(state.entry_price),
                "unrealized_pnl": float(unpnl),
                "equity": float(equity),
                "drawdown": float(dd),
                "notional": 0.0,
                "margin_used": 0.0,
                "leverage_used": 0.0,
                "exposure": 0.0,
            }

    metrics = compute_metrics(
        open_time_ms=[int(r["open_time_ms"]) for r in equity_rows],
        equity=[float(r["equity"]) for r in equity_rows],
        drawdown=[float(r["drawdown"]) for r in equity_rows],
        trade_count=len(trade_rows),
        fee_total=fee_total,
        slippage_total=slippage_total,
        interval_ms=int(interval_ms),
        trades=trade_rows,
        exposure=[float(r.get("exposure", 0.0)) for r in equity_rows],
        seed=None,  # Will use global seed if set
    )

    extra = {
        "interval_ms": int(interval_ms),
        "fee_total": float(fee_total),
        "slippage_total": float(slippage_total),
        "leverage": float(lev),
        "maintenance_margin_rate": float(mmr),
        "allow_short": bool(allow_short),
        "liquidated": bool(state.liquidated),
    }
    return equity_rows, trade_rows, metrics, extra


def persist_futures_backtest_to_db(
    *,
    conn: sqlite3.Connection,
    backtest_id: str,
    run_id: str,
    mode: str,
    dataset_meta: Mapping[str, Any],
    params_json: str,
    metrics: BacktestMetrics,
    equity_rows: List[Dict[str, Any]],
    trade_rows: List[Dict[str, Any]],
    started_utc: str,
    finished_utc: str,
    status: str,
) -> None:
    dataset_id = str(dataset_meta.get("dataset_id") or "")
    market_type = str(dataset_meta.get("market_type") or mode)
    symbol = str(dataset_meta.get("symbol") or "")
    interval = str(dataset_meta.get("interval") or "")

    conn.execute(
        """
        INSERT INTO backtests(
          backtest_id, run_id, mode, dataset_id, market_type, symbol, interval,
          started_utc, finished_utc, status,
          initial_cash, final_equity, total_return, max_drawdown, trade_count,
          fee_total, slippage_total, params_json, metrics_json
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            backtest_id,
            run_id,
            mode,
            dataset_id,
            market_type,
            symbol,
            interval,
            started_utc,
            finished_utc,
            status,
            float(metrics.initial_cash),
            float(metrics.final_equity),
            float(metrics.total_return),
            float(metrics.max_drawdown),
            int(metrics.trade_count),
            float(metrics.fee_total),
            float(metrics.slippage_total),
            params_json,
            json.dumps(metrics.to_dict(), ensure_ascii=False),
        ),
    )

    conn.executemany(
        """
        INSERT INTO backtest_futures_equity(
          backtest_id, open_time_ms, price,
          collateral, position_qty, entry_price,
          unrealized_pnl, equity, drawdown,
          notional, margin_used, leverage_used, exposure
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        [
            (
                backtest_id,
                int(r["open_time_ms"]),
                float(r["price"]),
                float(r["collateral"]),
                float(r["position_qty"]),
                float(r["entry_price"]),
                float(r["unrealized_pnl"]),
                float(r["equity"]),
                float(r["drawdown"]),
                float(r["notional"]),
                float(r["margin_used"]),
                float(r["leverage_used"]) if math.isfinite(float(r["leverage_used"])) else 1e308,
                float(r["exposure"]),
            )
            for r in equity_rows
        ],
    )

    if trade_rows:
        conn.executemany(
            """
            INSERT INTO backtest_futures_trades(
              backtest_id, open_time_ms,
              side, qty, fill_price, notional,
              fee, slippage_cost, realized_pnl, reason,
              collateral_after, position_qty_after, entry_price_after, equity_after
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                (
                    backtest_id,
                    int(r["open_time_ms"]),
                    str(r["side"]),
                    float(r["qty"]),
                    float(r["fill_price"]),
                    float(r["notional"]),
                    float(r["fee"]),
                    float(r["slippage_cost"]),
                    float(r["realized_pnl"]),
                    str(r.get("reason") or ""),
                    float(r["collateral_after"]),
                    float(r["position_qty_after"]),
                    float(r["entry_price_after"]),
                    float(r["equity_after"]),
                )
                for r in trade_rows
            ],
        )
