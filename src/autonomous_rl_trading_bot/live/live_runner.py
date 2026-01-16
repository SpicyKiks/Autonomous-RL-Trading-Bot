from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from autonomous_rl_trading_bot.broker.base_broker import BrokerAdapter
from autonomous_rl_trading_bot.broker.execution import order_for_target_fraction
from autonomous_rl_trading_bot.common.db import connect
from autonomous_rl_trading_bot.common.timeframes import interval_to_ms
from autonomous_rl_trading_bot.common.types import OrderQtyUnit
from autonomous_rl_trading_bot.data.candles_store import insert_candles
from autonomous_rl_trading_bot.evaluation.baselines import Strategy, make_strategy
from autonomous_rl_trading_bot.exchange.binance_public import Candle
from autonomous_rl_trading_bot.live.candle_sync import CandleSync, CandleSyncConfig
from autonomous_rl_trading_bot.live.position_sync import LivePosition, PositionSync
from autonomous_rl_trading_bot.live.safeguards import Safeguards, SafeguardsConfig


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _slip_frac(slippage_bps: float) -> float:
    return max(0.0, float(slippage_bps)) / 10000.0


def _fee_rate(fee_bps: float) -> float:
    return max(0.0, float(fee_bps)) / 10000.0


@dataclass
class LiveRunnerConfig:
    """Configuration for live (paper) trading."""

    market_type: str  # spot|futures
    symbol: str
    interval: str

    # Policy selection
    policy: str = "baseline"  # baseline|sb3
    strategy: str = "buy_and_hold"  # baseline only
    strategy_params: Optional[Dict[str, float | int]] = None  # baseline only
    sb3_algo: str = "ppo"  # ppo|dqn (sb3 only)
    sb3_model_path: str = ""  # sb3 only

    # Execution + portfolio
    initial_equity: float = 1000.0
    position_fraction: float = 1.0
    futures_leverage: float = 3.0
    maintenance_margin_rate: float = 0.005
    stop_on_liquidation: bool = True

    fee_bps: float = 10.0
    slippage_bps: float = 5.0

    # Observation
    lookback: int = 30

    # Candle polling
    poll_seconds: float = 2.0
    require_settled_ms: int = 250

    # Safeguards
    kill_switch_path: str = ""
    max_drawdown: float = 0.25
    min_seconds_between_trades: float = 15.0
    max_trades_per_hour: int = 30

    # Persistence
    db_path: str = ""
    run_id: str = ""
    run_dir: str = ""

    # Demo mode
    demo: bool = False  # If True, use real Binance Demo broker instead of paper trading


class _Policy:
    def act_target(self, *, t: int, obs: np.ndarray, price: float) -> float:
        raise NotImplementedError


class _BaselinePolicy(_Policy):
    def __init__(self, *, market_type: str, strategy: Strategy):
        self.market_type = market_type
        self.strategy = strategy
        self._price_history: List[float] = []
        self._time_history: List[int] = []

    def act_target(self, *, t: int, obs: np.ndarray, price: float) -> float:
        # Handle strategies with act() method (step-wise)
        if hasattr(self.strategy, "act"):
            try:
                a = int(self.strategy.act(t, float(price)))
            except (TypeError, AttributeError):
                # Fallback to generate_positions if act() fails
                a = self._act_from_positions(t, price)
        else:
            # Handle strategies with generate_positions() method (vectorized)
            a = self._act_from_positions(t, price)

        mt = self.market_type

        # evaluation.baselines actions:
        #   spot:   0=Hold, 1=Buy, 2=Sell, 3=Close
        #   futures:0=Hold, 1=Buy/Long, 2=Sell/Short, 3=Close
        if mt == "spot":
            if a == 1:
                return 1.0
            if a in (2, 3):
                return 0.0
            return 0.0

        if a == 1:
            return 1.0
        if a == 2:
            return -1.0
        if a == 3:
            return 0.0
        return 0.0

    def _act_from_positions(self, t: int, price: float) -> int:
        """Convert generate_positions() output to action for live trading."""
        import pandas as pd

        # Maintain price history for strategies that need DataFrame
        self._price_history.append(float(price))
        self._time_history.append(int(time.time() * 1000))  # Current timestamp in ms

        # Keep reasonable history (last 1000 prices)
        if len(self._price_history) > 1000:
            self._price_history = self._price_history[-1000:]
            self._time_history = self._time_history[-1000:]

        # Build DataFrame for strategy
        df = pd.DataFrame(
            {
                "open_time_ms": self._time_history,
                "close": self._price_history,
            }
        )

        # Call generate_positions
        positions = self.strategy.generate_positions(df)

        # Get position at current time step (last position)
        if isinstance(positions, pd.Series):
            pos_value = float(positions.iloc[-1]) if len(positions) > 0 else 0.0
        else:
            pos_value = float(positions[-1]) if len(positions) > 0 else 0.0

        # Convert position to action
        # position: +1=long, 0=flat, -1=short
        # action: 0=Hold, 1=Buy/Long, 2=Sell/Short, 3=Close
        if self.market_type == "spot":
            return 1 if pos_value > 0 else 3  # Buy or close-all
        else:  # futures
            if pos_value > 0:
                return 1  # Long
            if pos_value < 0:
                return 2  # Short
            return 3  # Close


class _SB3Policy(_Policy):
    def __init__(self, *, market_type: str, algo: str, model_path: str):
        self.market_type = market_type
        self.algo = (algo or "").strip().lower()
        self.model_path = model_path

        if not self.model_path:
            raise ValueError("sb3_model_path is required when policy=sb3")

        try:
            if self.algo == "ppo":
                from stable_baselines3 import PPO  # type: ignore

                self.model = PPO.load(self.model_path)
            elif self.algo == "dqn":
                from stable_baselines3 import DQN  # type: ignore

                self.model = DQN.load(self.model_path)
            else:
                raise ValueError("sb3_algo must be ppo|dqn")
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "stable-baselines3 is not installed. Install project deps (pip install -e .)"
            ) from e

    def act_target(self, *, t: int, obs: np.ndarray, price: float) -> float:
        action, _ = self.model.predict(obs.astype(np.float32, copy=False), deterministic=True)
        a = int(action)

        # env_trading.TradingEnv actions:
        #   spot:   0=HOLD, 1=LONG, 2=FLAT
        #   futures:0=HOLD, 1=LONG, 2=FLAT, 3=SHORT
        if self.market_type == "spot":
            if a == 1:
                return 1.0
            if a == 2:
                return 0.0
            return 0.0

        if a == 1:
            return 1.0
        if a == 2:
            return 0.0
        if a == 3:
            return -1.0
        return 0.0


@dataclass
class _PortfolioState:
    cash: float
    qty: float  # spot: base qty (>=0), futures: signed contracts
    entry_price: float
    peak_equity: float
    fee_total: float = 0.0
    slippage_total: float = 0.0


class LiveRunner:
    """Candle-driven live runner (paper trading)."""

    def __init__(self, cfg: Dict[str, Any], rcfg: LiveRunnerConfig):
        self.cfg = cfg
        self.rcfg = rcfg

        mt = (rcfg.market_type or "").strip().lower()
        if mt not in ("spot", "futures"):
            raise ValueError(f"market_type must be spot|futures, got {mt!r}")
        self.market_type = mt

        if rcfg.lookback < 5:
            raise ValueError("lookback should be >= 5 for stable features")

        if not rcfg.db_path:
            raise ValueError("db_path is required")
        if not rcfg.run_id:
            raise ValueError("run_id is required")
        if not rcfg.run_dir:
            raise ValueError("run_dir is required")

        self.run_dir = Path(rcfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.candle_sync = CandleSync(
            cfg,
            CandleSyncConfig(
                market_type=self.market_type,
                symbol=rcfg.symbol,
                interval=rcfg.interval,
                poll_seconds=float(rcfg.poll_seconds),
                require_settled_ms=int(rcfg.require_settled_ms),
            ),
        )

        # Broker setup: paper trading (default) or Binance Demo (if demo=True)
        self.broker = None
        if rcfg.demo:
            import os
            from autonomous_rl_trading_bot.broker.futures_broker import FuturesBroker
            from autonomous_rl_trading_bot.broker.spot_broker import SpotBroker

            api_key = os.getenv("BINANCE_DEMO_API_KEY")
            api_secret = os.getenv("BINANCE_DEMO_API_SECRET")

            if not api_key or not api_secret:
                raise RuntimeError(
                    "Demo mode requires BINANCE_DEMO_API_KEY and BINANCE_DEMO_API_SECRET environment variables. "
                    "Set them before running with --demo flag."
                )

            # Get demo base URL from config
            exchange_cfg = cfg.get("exchange", {}) or {}
            if self.market_type == "spot":
                spot_cfg = exchange_cfg.get("spot", {}) or {}
                base_url_demo = spot_cfg.get("base_url_demo", "https://testnet.binance.vision")
                self.broker = SpotBroker(
                    api_key=api_key,
                    api_secret=api_secret,
                    demo=True,
                    base_url_demo=base_url_demo,
                )
            else:  # futures
                futures_cfg = exchange_cfg.get("futures", {}) or {}
                base_url_demo = futures_cfg.get("base_url_demo", "https://demo-fapi.binance.com")
                self.broker = FuturesBroker(
                    api_key=api_key,
                    api_secret=api_secret,
                    demo=True,
                    base_url_demo=base_url_demo,
                    leverage=rcfg.futures_leverage,
                )

        self.pos_sync = PositionSync(
            broker=self.broker,
            symbol=rcfg.symbol,
            market_type=self.market_type,
        )

        self.safeguards = Safeguards(
            SafeguardsConfig(
                kill_switch_path=rcfg.kill_switch_path,
                max_drawdown=float(rcfg.max_drawdown),
                min_seconds_between_trades=float(rcfg.min_seconds_between_trades),
                max_trades_per_hour=int(rcfg.max_trades_per_hour),
            )
        )

        self.policy = self._make_policy()

        self._interval_ms = interval_to_ms(rcfg.interval)

        self._closes: List[float] = []
        self._vols: List[float] = []
        self._open_times: List[int] = []
        self._first_close: Optional[float] = None

        self.state = _PortfolioState(
            cash=float(rcfg.initial_equity),
            qty=0.0,
            entry_price=0.0,
            peak_equity=float(rcfg.initial_equity),
        )

        self._t = 0
        self._init_db_rows()

        # Log mode
        import logging

        logger = logging.getLogger("arbt")
        logger.info("=" * 60)
        if rcfg.demo:
            logger.info("LIVE MODE: BINANCE DEMO")
        else:
            logger.info("LIVE MODE: PAPER")
        logger.info(f"Market Type: {self.market_type.upper()}")
        logger.info(f"Symbol: {rcfg.symbol}")
        logger.info("=" * 60)

    def _make_policy(self) -> _Policy:
        p = (self.rcfg.policy or "baseline").strip().lower()
        if p == "baseline":
            params = self.rcfg.strategy_params or {}
            strat = make_strategy(self.rcfg.strategy, params=params)
            return _BaselinePolicy(market_type=self.market_type, strategy=strat)
        if p == "sb3":
            return _SB3Policy(
                market_type=self.market_type,
                algo=self.rcfg.sb3_algo,
                model_path=self.rcfg.sb3_model_path,
            )
        raise ValueError("policy must be baseline|sb3")

    # -----------------
    # Persistence layer
    # -----------------

    def _init_db_rows(self) -> None:
        params = {
            "market_type": self.market_type,
            "symbol": self.rcfg.symbol,
            "interval": self.rcfg.interval,
            "policy": self.rcfg.policy,
            "strategy": self.rcfg.strategy,
            "strategy_params": self.rcfg.strategy_params or {},
            "sb3_algo": self.rcfg.sb3_algo,
            "sb3_model_path": self.rcfg.sb3_model_path,
            "initial_equity": self.rcfg.initial_equity,
            "position_fraction": self.rcfg.position_fraction,
            "futures_leverage": self.rcfg.futures_leverage,
            "maintenance_margin_rate": self.rcfg.maintenance_margin_rate,
            "fee_bps": self.rcfg.fee_bps,
            "slippage_bps": self.rcfg.slippage_bps,
            "lookback": self.rcfg.lookback,
            "safeguards": self.safeguards.summary(),
        }
        params_json = json.dumps(params, ensure_ascii=False)

        with connect(Path(self.rcfg.db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO live_sessions
                (live_id, run_id, mode, market_type, symbol, interval,
                 started_utc, finished_utc, status,
                 initial_equity, final_equity,
                 fee_total, slippage_total,
                 params_json, metrics_json)
                VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.rcfg.run_id,
                    self.rcfg.run_id,
                    self.market_type,
                    self.market_type,
                    self.rcfg.symbol,
                    self.rcfg.interval,
                    _iso_utc_now(),
                    None,
                    "RUNNING",
                    float(self.rcfg.initial_equity),
                    None,
                    0.0,
                    0.0,
                    params_json,
                    None,
                ),
            )
            conn.commit()

    def _persist_equity_row(self, *, candle: Candle, equity: float, drawdown: float, exposure: float) -> None:
        with connect(Path(self.rcfg.db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO live_equity
                (live_id, open_time_ms, price, cash, position_qty, entry_price,
                 unrealized_pnl, equity, drawdown,
                 notional, margin_used, leverage_used, exposure)
                VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.rcfg.run_id,
                    int(candle.open_time_ms),
                    float(candle.close),
                    float(self.state.cash),
                    float(self.state.qty),
                    float(self.state.entry_price),
                    float(self._unrealized_pnl(float(candle.close))),
                    float(equity),
                    float(drawdown),
                    float(self._notional(float(candle.close))),
                    float(self._margin_used(float(candle.close))),
                    float(self._leverage_used(float(candle.close), equity)),
                    float(exposure),
                ),
            )
            conn.commit()

    def _persist_trade_row(self, trade: Dict[str, Any]) -> None:
        with connect(Path(self.rcfg.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO live_trades
                (live_id, open_time_ms, side, qty, fill_price, notional,
                 fee, slippage_cost, realized_pnl, reason,
                 cash_after, position_qty_after, entry_price_after, equity_after)
                VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.rcfg.run_id,
                    int(trade["open_time_ms"]),
                    str(trade["side"]),
                    float(trade["qty"]),
                    float(trade["fill_price"]),
                    float(trade["notional"]),
                    float(trade["fee"]),
                    float(trade["slippage_cost"]),
                    float(trade.get("realized_pnl", 0.0)),
                    str(trade.get("reason") or "strategy"),
                    float(trade["cash_after"]),
                    float(trade["position_qty_after"]),
                    float(trade["entry_price_after"]),
                    float(trade["equity_after"]),
                ),
            )
            conn.commit()

    def _finalize_session(self, *, status: str, final_equity: float, metrics: Dict[str, Any]) -> None:
        with connect(Path(self.rcfg.db_path)) as conn:
            conn.execute(
                """
                UPDATE live_sessions
                SET finished_utc=?, status=?, final_equity=?, fee_total=?, slippage_total=?, metrics_json=?
                WHERE live_id=?
                """,
                (
                    _iso_utc_now(),
                    str(status),
                    float(final_equity),
                    float(self.state.fee_total),
                    float(self.state.slippage_total),
                    json.dumps(metrics, ensure_ascii=False),
                    self.rcfg.run_id,
                ),
            )
            conn.commit()

    # -----------------
    # Accounting helpers
    # -----------------

    def _equity(self, price: float) -> float:
        if self.market_type == "spot":
            return float(self.state.cash + self.state.qty * price)
        if abs(self.state.qty) < 1e-18:
            return float(self.state.cash)
        return float(self.state.cash + self.state.qty * (price - self.state.entry_price))

    def _unrealized_pnl(self, price: float) -> float:
        if self.market_type != "futures":
            return 0.0
        if abs(self.state.qty) < 1e-18:
            return 0.0
        return float(self.state.qty * (price - self.state.entry_price))

    def _notional(self, price: float) -> float:
        return float(abs(self.state.qty) * price)

    def _margin_used(self, price: float) -> float:
        if self.market_type != "futures":
            return 0.0
        lev = max(1e-6, float(self.rcfg.futures_leverage))
        return float(self._notional(price) / lev)

    def _leverage_used(self, price: float, equity: float) -> float:
        if equity <= 0.0:
            return 0.0
        return float(self._notional(price) / equity)

    # -----------------
    # Feature window
    # -----------------

    def _update_history(self, c: Candle) -> None:
        price = float(c.close)
        vol = float(c.volume)

        self._open_times.append(int(c.open_time_ms))
        self._closes.append(price)
        self._vols.append(vol)

        lb = int(self.rcfg.lookback)
        if len(self._closes) > lb + 1:
            self._open_times = self._open_times[-(lb + 1) :]
            self._closes = self._closes[-(lb + 1) :]
            self._vols = self._vols[-(lb + 1) :]

        if self._first_close is None:
            self._first_close = price

    def _obs(self) -> np.ndarray:
        lb = int(self.rcfg.lookback)
        if len(self._closes) < lb + 1:
            n_features = 4
            return np.zeros((lb * n_features,), dtype=np.float32)

        closes = np.asarray(self._closes[-(lb + 1) :], dtype=np.float64)
        vols = np.asarray(self._vols[-(lb + 1) :], dtype=np.float64)

        prev = closes[:-1]
        cur = closes[1:]

        ret = np.where(prev > 0, (cur / prev) - 1.0, 0.0)
        log_ret = np.where(prev > 0, np.log(np.maximum(cur, 1e-12) / prev), 0.0)

        first = float(self._first_close or cur[0])
        close_norm = cur / max(first, 1e-12)
        vol_norm = np.log1p(np.maximum(vols[1:], 0.0))

        feats = np.stack([log_ret, ret, close_norm, vol_norm], axis=1).astype(np.float32)
        return feats.reshape(-1)

    # -----------------
    # Execution
    # -----------------

    def _execute_to_target(self, *, target: float, price: float, open_time_ms: int) -> Optional[Dict[str, Any]]:
        target = float(max(-1.0, min(1.0, target)))
        if self.market_type == "spot":
            target = max(0.0, target)

        equity_now = self._equity(price)
        if equity_now <= 0.0:
            return None

        # If broker exists (demo mode), submit real orders
        if self.broker is not None:
            return self._execute_to_target_demo(target=target, price=price, open_time_ms=open_time_ms, equity_now=equity_now)

        # Paper trading mode: simulate execution
        if self.market_type == "spot":
            target_notional = equity_now * float(self.rcfg.position_fraction) * target
            target_qty = target_notional / price
        else:
            target_notional = (
                equity_now
                * float(self.rcfg.position_fraction)
                * float(self.rcfg.futures_leverage)
                * abs(target)
            )
            target_qty = (target_notional / price) * (1.0 if target >= 0 else -1.0)

        delta = target_qty - float(self.state.qty)
        if abs(delta) <= 1e-12:
            return None

        slip = _slip_frac(self.rcfg.slippage_bps)
        fee_r = _fee_rate(self.rcfg.fee_bps)

        side = "BUY" if delta > 0 else "SELL"
        fill_price = price * (1.0 + slip) if delta > 0 else price * (1.0 - slip)
        qty = float(delta)
        notional = abs(qty) * float(fill_price)
        fee = notional * fee_r
        slippage_cost = abs(fill_price - price) * abs(qty)

        realized_pnl = 0.0

        if self.market_type == "spot":
            if qty > 0:
                max_affordable = max(self.state.cash - fee, 0.0)
                max_qty = max_affordable / float(fill_price)
                if max_qty <= 1e-12:
                    return None
                if qty > max_qty:
                    qty = max_qty
                    notional = abs(qty) * float(fill_price)
                    fee = notional * fee_r
                    slippage_cost = abs(fill_price - price) * abs(qty)

            if qty > 0:
                self.state.cash -= notional + fee
            else:
                self.state.cash += abs(qty) * float(fill_price) - fee
            self.state.qty += qty
            if self.state.qty < 0:
                self.state.qty = 0.0

        else:
            if abs(self.state.qty) > 1e-12:
                if target == 0.0 or math.copysign(1.0, target) != math.copysign(1.0, self.state.qty):
                    realized_pnl = float(self.state.qty * (price - self.state.entry_price))
                    self.state.cash += realized_pnl
                    self.state.qty = 0.0
                    self.state.entry_price = 0.0

            self.state.cash -= fee

            if abs(target_qty) > 1e-12:
                if abs(self.state.qty) <= 1e-12:
                    self.state.qty = target_qty
                    self.state.entry_price = price
                else:
                    prev_qty = float(self.state.qty)
                    prev_entry = float(self.state.entry_price)
                    add_qty = float(delta)
                    new_qty = target_qty
                    denom = abs(prev_qty) + abs(add_qty)
                    if denom > 1e-12:
                        self.state.entry_price = (prev_entry * abs(prev_qty) + float(fill_price) * abs(add_qty)) / denom
                    self.state.qty = new_qty

            if self.rcfg.stop_on_liquidation and abs(self.state.qty) > 1e-12:
                notional_now = abs(self.state.qty) * price
                maint = notional_now * float(self.rcfg.maintenance_margin_rate)
                equity_now2 = self._equity(price)
                if equity_now2 <= maint:
                    realized_pnl2 = float(self.state.qty * (price - self.state.entry_price))
                    self.state.cash += realized_pnl2
                    realized_pnl += realized_pnl2
                    self.state.qty = 0.0
                    self.state.entry_price = 0.0
                    side = "CLOSE"

        self.state.fee_total += float(fee)
        self.state.slippage_total += float(slippage_cost)

        equity_after = self._equity(price)
        self.state.peak_equity = max(self.state.peak_equity, float(equity_after))

        self.pos_sync.set_local(LivePosition(qty=float(self.state.qty), entry_price=float(self.state.entry_price)))

        return {
            "open_time_ms": int(open_time_ms),
            "side": side,
            "qty": float(qty),
            "fill_price": float(fill_price),
            "notional": float(notional),
            "fee": float(fee),
            "slippage_cost": float(slippage_cost),
            "realized_pnl": float(realized_pnl),
            "reason": "strategy",
            "cash_after": float(self.state.cash),
            "position_qty_after": float(self.state.qty),
            "entry_price_after": float(self.state.entry_price),
            "equity_after": float(equity_after),
        }

    def _execute_to_target_demo(
        self, *, target: float, price: float, open_time_ms: int, equity_now: float
    ) -> Optional[Dict[str, Any]]:
        """Execute order via real broker (demo mode)."""
        import logging

        logger = logging.getLogger("arbt")

        # Get current position from broker
        broker_pos = self.pos_sync.sync_from_broker()
        current_qty = broker_pos.qty if broker_pos else 0.0

        # Determine qty_unit based on market type
        qty_unit: OrderQtyUnit = "base" if self.market_type == "spot" else "contracts"

        # Create order request
        order_req = order_for_target_fraction(
            symbol=self.rcfg.symbol,
            current_qty=current_qty,
            equity=equity_now,
            price=price,
            target_fraction=target,
            market_type=self.market_type,
            qty_unit=qty_unit,
            leverage=self.rcfg.futures_leverage if self.market_type == "futures" else 1.0,
        )

        if order_req is None:
            return None  # No order needed

        # Submit order via broker
        try:
            ack = self.broker.submit_order(order_req)
            if ack.status == "rejected":
                logger.warning(f"Order rejected: {ack.reason}")
                return None

            # Wait for fills (poll briefly)
            import time

            time.sleep(0.5)  # Brief wait for order to fill
            fills = list(self.broker.iter_fills())

            # Process fills and update state
            total_qty = 0.0
            total_fee = 0.0
            weighted_price = 0.0
            realized_pnl = 0.0

            for fill in fills:
                if fill.order_id == ack.order_id:
                    total_qty += fill.qty
                    total_fee += fill.fee_paid
                    weighted_price += fill.price * abs(fill.qty)

            if abs(total_qty) > 1e-12:
                avg_fill_price = weighted_price / abs(total_qty) if weighted_price > 0 else price

                # Update state based on fills
                if self.market_type == "spot":
                    if total_qty > 0:
                        self.state.cash -= abs(total_qty) * avg_fill_price + total_fee
                    else:
                        self.state.cash += abs(total_qty) * avg_fill_price - total_fee
                    self.state.qty = max(0.0, self.state.qty + total_qty)
                else:  # futures
                    # Close existing position if reversing
                    if abs(self.state.qty) > 1e-12:
                        if (total_qty > 0 and self.state.qty < 0) or (total_qty < 0 and self.state.qty > 0):
                            realized_pnl = float(self.state.qty * (avg_fill_price - self.state.entry_price))
                            self.state.cash += realized_pnl
                            self.state.qty = 0.0
                            self.state.entry_price = 0.0

                    # Update position
                    if abs(total_qty) > 1e-12:
                        if abs(self.state.qty) <= 1e-12:
                            self.state.qty = total_qty
                            self.state.entry_price = avg_fill_price
                        else:
                            # Weighted average entry price
                            prev_qty = self.state.qty
                            prev_entry = self.state.entry_price
                            new_qty = self.state.qty + total_qty
                            denom = abs(prev_qty) + abs(total_qty)
                            if denom > 1e-12:
                                self.state.entry_price = (prev_entry * abs(prev_qty) + avg_fill_price * abs(total_qty)) / denom
                            self.state.qty = new_qty

                    self.state.cash -= total_fee

                self.state.fee_total += total_fee
                self.pos_sync.set_local(LivePosition(qty=float(self.state.qty), entry_price=float(self.state.entry_price)))

                equity_after = self._equity(price)
                self.state.peak_equity = max(self.state.peak_equity, float(equity_after))

                side_str = "BUY" if total_qty > 0 else "SELL"
                if self.market_type == "futures" and abs(total_qty) > 0 and abs(self.state.qty) <= 1e-12:
                    side_str = "CLOSE"

                return {
                    "open_time_ms": int(open_time_ms),
                    "side": side_str,
                    "qty": float(total_qty),
                    "fill_price": float(avg_fill_price),
                    "notional": abs(total_qty) * float(avg_fill_price),
                    "fee": float(total_fee),
                    "slippage_cost": abs(avg_fill_price - price) * abs(total_qty),
                    "realized_pnl": float(realized_pnl),
                    "reason": "strategy",
                    "cash_after": float(self.state.cash),
                    "position_qty_after": float(self.state.qty),
                    "entry_price_after": float(self.state.entry_price),
                    "equity_after": float(equity_after),
                    "order_id": ack.order_id,
                }

        except Exception as e:
            logger.error(f"Failed to execute order via broker: {e}", exc_info=True)
            return None

        return None

    # -----------------
    # Main loop
    # -----------------

    def run(self, *, max_steps: Optional[int] = None, max_minutes: Optional[float] = None) -> Dict[str, Any]:
        start_ms = _now_ms()
        trades_out: List[Dict[str, Any]] = []
        status = "DONE"
        stop_reason: Optional[str] = None

        try:
            while True:
                if max_steps is not None and self._t >= int(max_steps):
                    stop_reason = "max_steps"
                    break
                if max_minutes is not None:
                    if (_now_ms() - start_ms) >= int(float(max_minutes) * 60_000):
                        stop_reason = "max_minutes"
                        break
                if self.safeguards.kill_switch_triggered():
                    stop_reason = "kill_switch"
                    break

                candle = self.candle_sync.wait_next()

                with connect(Path(self.rcfg.db_path)) as conn:
                    insert_candles(conn, [candle])
                    conn.commit()

                self._update_history(candle)
                obs = self._obs()
                price = float(candle.close)

                equity_now = self._equity(price)
                self.state.peak_equity = max(self.state.peak_equity, equity_now)
                drawdown = max(0.0, 1.0 - (equity_now / max(self.state.peak_equity, 1e-12)))
                notional = self._notional(price)
                exposure = float(notional / equity_now) if equity_now > 1e-12 else 0.0

                self._persist_equity_row(candle=candle, equity=equity_now, drawdown=drawdown, exposure=exposure)

                if self.safeguards.drawdown_breached(equity=equity_now, peak_equity=self.state.peak_equity):
                    stop_reason = "max_drawdown"
                    break

                target = float(self.policy.act_target(t=self._t, obs=obs, price=price))

                if self.market_type == "spot":
                    current_target = 1.0 if self.state.qty > 1e-12 else 0.0
                else:
                    current_target = 0.0 if abs(self.state.qty) <= 1e-12 else math.copysign(1.0, self.state.qty)

                want_change = abs(target - current_target) > 1e-12

                if want_change and self.safeguards.allow_trade():
                    trade = self._execute_to_target(target=target, price=price, open_time_ms=int(candle.open_time_ms))
                    if trade is not None:
                        self.safeguards.record_trade()
                        trades_out.append(trade)
                        self._persist_trade_row(trade)

                self._t += 1

        except KeyboardInterrupt:
            status = "STOPPED"
            stop_reason = stop_reason or "keyboard_interrupt"
        except Exception:
            status = "FAILED"
            stop_reason = stop_reason or "exception"
            raise
        finally:
            # Force flat on exit (always attempt; do not rate-limit shutdown)
            try:
                last_price = float(self._closes[-1]) if self._closes else 0.0
                if last_price > 0:
                    trade = self._execute_to_target(
                        target=0.0,
                        price=last_price,
                        open_time_ms=int(self._open_times[-1] if self._open_times else _now_ms()),
                    )
                    if trade is not None:
                        trade["reason"] = "shutdown_close"
                        trades_out.append(trade)
                        self._persist_trade_row(trade)
            except Exception:
                pass

            # Close broker connection if demo mode
            if self.broker is not None:
                try:
                    self.broker.close()
                except Exception:
                    pass

            final_equity = self._equity(float(self._closes[-1]) if self._closes else 0.0)
            metrics = {
                "final_equity": float(final_equity),
                "peak_equity": float(self.state.peak_equity),
                "trade_count": int(len(trades_out)),
                "fee_total": float(self.state.fee_total),
                "slippage_total": float(self.state.slippage_total),
                "stop_reason": stop_reason,
            }
            self._finalize_session(status=status, final_equity=final_equity, metrics=metrics)

            try:
                (self.run_dir / "live_trades.json").write_text(
                    json.dumps(trades_out, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except OSError as e:
                logger.error(f"Failed to write live_trades.json: {e}")
            try:
                (self.run_dir / "live_metrics.json").write_text(
                    json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except OSError as e:
                logger.error(f"Failed to write live_metrics.json: {e}")

        return {
            "status": status,
            "stop_reason": stop_reason,
            "trade_count": len(trades_out),
            "final_equity": float(self._equity(float(self._closes[-1]) if self._closes else 0.0)),
            "run_id": self.rcfg.run_id,
        }

