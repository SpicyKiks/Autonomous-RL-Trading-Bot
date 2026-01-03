from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class TradingEnvConfig:
    market_type: str  # spot|futures
    lookback: int = 30
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    initial_equity: float = 1000.0
    position_fraction: float = 1.0
    futures_leverage: float = 3.0  # used only for futures
    reward_kind: str = "log_equity"  # log_equity|delta_equity
    # Split boundaries (inclusive start, exclusive end)
    start_index: int = 0
    end_index: int = 0


class TradingEnv(gym.Env):
    """
    Deterministic offline trading env over a fixed price series.
    Observation: flattened [lookback x n_features] window ending at t (inclusive).
    Actions:
      spot:   0=HOLD, 1=LONG(+1), 2=FLAT(0)
      futures:0=HOLD, 1=LONG(+1), 2=FLAT(0), 3=SHORT(-1)
    """
    metadata = {"render_modes": []}

    def __init__(self, data: Dict[str, np.ndarray], cfg: TradingEnvConfig, seed: int = 0, feature_list: Optional[List[str]] = None):
        super().__init__()
        self.data = data
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)

        mt = cfg.market_type
        if mt not in ("spot", "futures"):
            raise ValueError(f"market_type must be spot|futures, got {mt}")
        if cfg.lookback < 1:
            raise ValueError("lookback must be >= 1")

        # required columns
        for k in ("close", "volume"):
            if k not in data:
                raise KeyError(f"dataset missing key: {k}")

        self.close = np.asarray(data["close"], dtype=np.float64)
        self.volume = np.asarray(data["volume"], dtype=np.float64)
        self.open_time_ms = np.asarray(data.get("open_time_ms", np.arange(self.close.size)), dtype=np.int64)

        if cfg.end_index <= 0 or cfg.end_index > self.close.size:
            raise ValueError("end_index must be set to a valid upper bound")
        if not (0 <= cfg.start_index < cfg.end_index):
            raise ValueError("invalid start/end indices")

        # Use feature_list from meta if provided, otherwise fallback to default
        if feature_list is None:
            feature_list = ["log_return", "return", "close_norm", "vol_norm"]
        
        self.feature_list = feature_list
        
        # Build feature matrix using scaled features if available, otherwise compute on-the-fly
        feature_arrays = []
        for feat_name in feature_list:
            scaled_key = f"{feat_name}_scaled"
            if scaled_key in data:
                # Use pre-scaled feature
                feat = np.asarray(data[scaled_key], dtype=np.float32)
            elif feat_name == "close_norm":
                first_close = float(self.close[cfg.start_index])
                feat = (self.close / max(first_close, 1e-12)).astype(np.float32)
            elif feat_name == "vol_norm":
                feat = np.log1p(np.maximum(self.volume, 0.0)).astype(np.float32)
            elif feat_name in data:
                feat = np.asarray(data[feat_name], dtype=np.float32)
            else:
                raise KeyError(f"Feature {feat_name} not found in dataset and cannot be computed")
            feature_arrays.append(feat)
        
        self.features = np.stack(feature_arrays, axis=1).astype(np.float32)  # (T, feature_dim)
        self.n_features = int(self.features.shape[1])
        
        # Account vector dimension: [cash_ratio, position_ratio, equity_ratio]
        self.account_dim = 3
        
        # Observation: flattened lookback window + account vector
        obs_shape = (cfg.lookback * self.n_features + self.account_dim,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        if mt == "spot":
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Discrete(4)

        self._reset_state()

    def _reset_state(self) -> None:
        self.t = max(self.cfg.start_index, self.cfg.lookback)
        self.done = False

        # Portfolio state
        self.cash = float(self.cfg.initial_equity)
        self.qty = 0.0  # base qty for spot, contracts qty for futures
        self.entry_price = 0.0  # futures avg entry
        self.equity = float(self.cfg.initial_equity)

        self.fee_total = 0.0
        self.slip_total = 0.0

        # For logging trades
        self.trades: List[Dict[str, Any]] = []

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is None:
            seed = 0
        self._rng = np.random.default_rng(int(seed))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)
        self._reset_state()
        obs = self._get_obs()
        info = {"equity": self.equity, "t": self.t}
        return obs, info

    def _get_obs(self) -> np.ndarray:
        lb = self.cfg.lookback
        start = self.t - lb
        end = self.t
        window = self.features[start:end, :]  # shape (lb, n_features)
        window_flat = window.reshape(-1).astype(np.float32, copy=False)
        
        # Account vector: [cash/initial_equity, position_value/equity, equity/initial_equity]
        cash_ratio = self.cash / max(self.cfg.initial_equity, 1e-12)
        
        price = self._price()
        if self.cfg.market_type == "spot":
            position_value = self.qty * price
        else:
            position_value = abs(self.qty) * price if abs(self.qty) > 1e-12 else 0.0
        
        position_ratio = position_value / max(self.equity, 1e-12) if self.equity > 0 else 0.0
        equity_ratio = self.equity / max(self.cfg.initial_equity, 1e-12)
        
        account_vec = np.array([cash_ratio, position_ratio, equity_ratio], dtype=np.float32)
        
        # Concatenate window + account vector
        obs = np.concatenate([window_flat, account_vec], axis=0).astype(np.float32)
        return obs

    def _price(self) -> float:
        return float(self.close[self.t])

    def _apply_costs(self, notional: float) -> Tuple[float, float]:
        fee = notional * (self.cfg.fee_bps / 10000.0)
        slip = notional * (self.cfg.slippage_bps / 10000.0)
        self.fee_total += fee
        self.slip_total += slip
        return fee, slip

    def _mark_to_market(self, price: float) -> float:
        if self.cfg.market_type == "spot":
            return self.cash + self.qty * price
        # futures: equity = cash + unrealized pnl
        if abs(self.qty) < 1e-18:
            return self.cash
        return self.cash + self.qty * (price - self.entry_price)

    def _execute_spot_to_target(self, target: float, price: float) -> None:
        # target in {-1,0,+1} but spot will clamp negative to 0
        target = max(0.0, float(target))
        # target_qty uses cash-based sizing (no leverage)
        target_notional = self._mark_to_market(price) * self.cfg.position_fraction
        target_qty = (target_notional / price) * target

        delta = target_qty - self.qty
        if abs(delta) < 1e-12:
            return

        notional = abs(delta) * price
        fee, slip = self._apply_costs(notional)

        # For buys, ensure we have enough cash; scale down if needed
        if delta > 0:
            max_affordable = max(self.cash - fee - slip, 0.0)
            max_qty = max_affordable / price
            buy_qty = min(delta, max_qty)
            if buy_qty <= 1e-12:
                return
            spend = buy_qty * price
            self.cash -= (spend + fee + slip)
            self.qty += buy_qty
            side = "BUY"
            fill_qty = buy_qty
        else:
            sell_qty = min(-delta, self.qty)
            if sell_qty <= 1e-12:
                return
            receive = sell_qty * price
            self.cash += (receive - fee - slip)
            self.qty -= sell_qty
            side = "SELL"
            fill_qty = -sell_qty

        self.equity = self._mark_to_market(price)
        i = min(self.t, len(self.open_time_ms) - 1)
        open_time_ms = int(self.open_time_ms[i])
        self.trades.append(
            dict(
                t=int(self.t),
                open_time_ms=open_time_ms,
                side=side,
                qty=float(fill_qty),
                price=float(price),
                notional=float(abs(fill_qty) * price),
                fee=float(fee),
                slippage_cost=float(slip),
                cash_after=float(self.cash),
                qty_after=float(self.qty),
                equity_after=float(self.equity),
            )
        )

    def _execute_futures_to_target(self, target: float, price: float) -> None:
        # target in {-1,0,+1}
        target = float(target)

        equity_now = self._mark_to_market(price)
        notional = equity_now * self.cfg.futures_leverage * self.cfg.position_fraction
        target_qty = (notional / price) * target  # signed

        delta = target_qty - self.qty
        if abs(delta) < 1e-12:
            self.equity = equity_now
            return

        trade_notional = abs(delta) * price
        fee, slip = self._apply_costs(trade_notional)

        # Realize pnl if closing or flipping
        realized_pnl = 0.0
        if abs(self.qty) > 1e-12 and (target == 0.0 or np.sign(target) != np.sign(self.qty)):
            realized_pnl = self.qty * (price - self.entry_price)
            self.cash += realized_pnl
            self.qty = 0.0
            self.entry_price = 0.0

        # Apply fees/slippage to collateral
        self.cash -= (fee + slip)

        # Open/resize position if target non-zero
        if abs(target) > 1e-12:
            # If currently flat, set entry
            if abs(self.qty) < 1e-12:
                self.qty = target_qty
                self.entry_price = price
            else:
                # same direction resize: update avg entry
                new_qty = target_qty
                if abs(new_qty) > 1e-12:
                    # weighted average entry price approximation
                    self.entry_price = (self.entry_price * abs(self.qty) + price * abs(delta)) / (
                        abs(self.qty) + abs(delta)
                    )
                    self.qty = new_qty

        self.equity = self._mark_to_market(price)
        side = "BUY" if delta > 0 else "SELL"
        i = min(self.t, len(self.open_time_ms) - 1)
        open_time_ms = int(self.open_time_ms[i])
        self.trades.append(
            dict(
                t=int(self.t),
                open_time_ms=open_time_ms,
                side=side,
                qty=float(delta),
                price=float(price),
                notional=float(trade_notional),
                fee=float(fee),
                slippage_cost=float(slip),
                realized_pnl=float(realized_pnl),
                cash_after=float(self.cash),
                position_qty_after=float(self.qty),
                entry_price_after=float(self.entry_price),
                equity_after=float(self.equity),
            )
        )

    def step(self, action: int):
        if self.done:
            raise RuntimeError("step() called after done=True. Call reset().")

        price = self._price()
        prev_equity = self._mark_to_market(price)

        target: Optional[float] = None
        mt = self.cfg.market_type

        if mt == "spot":
            # 0 hold, 1 long, 2 flat
            if action == 1:
                target = 1.0
            elif action == 2:
                target = 0.0
        else:
            # 0 hold, 1 long, 2 flat, 3 short
            if action == 1:
                target = 1.0
            elif action == 2:
                target = 0.0
            elif action == 3:
                target = -1.0

        if target is not None:
            if mt == "spot":
                self._execute_spot_to_target(target, price)
            else:
                self._execute_futures_to_target(target, price)
        else:
            self.equity = prev_equity

        # Advance time
        self.t += 1
        terminal = self.t >= self.cfg.end_index

        # If terminal, force close and recompute equity
        if terminal:
            final_price = float(self.close[self.cfg.end_index - 1])
            if mt == "spot":
                if self.qty > 1e-12:
                    self._execute_spot_to_target(0.0, final_price)
            else:
                if abs(self.qty) > 1e-12:
                    self._execute_futures_to_target(0.0, final_price)
            self.done = True

        curr_price = float(self.close[min(self.t, self.cfg.end_index - 1)])
        curr_equity = self._mark_to_market(curr_price)

        # Reward
        if self.cfg.reward_kind == "delta_equity":
            reward = (curr_equity - prev_equity) / max(self.cfg.initial_equity, 1e-12)
        else:
            # log equity ratio
            reward = float(np.log(max(curr_equity, 1e-12) / max(prev_equity, 1e-12)))

        obs = self._get_obs() if not self.done else self._get_obs()
        info = {"equity": float(curr_equity), "t": int(self.t), "fee_total": self.fee_total, "slippage_total": self.slip_total}
        return obs, float(reward), self.done, False, info

