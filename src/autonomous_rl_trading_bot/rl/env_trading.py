from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class TradingEnvConfig:
    """Configuration for :class:`TradingEnv`.

    Notes:
    - trainer.py passes `mode`, `seed`, `start_index`, and `end_index`
    - `market_type` is kept for backwards compatibility (spot|futures)
    """

    # Backwards-compatible: keep `market_type`, but allow trainer to pass `mode`.
    market_type: str = "spot"  # spot|futures
    mode: str = "spot"  # alias for market_type (trainer passes this)
    seed: int = 0

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

    def __post_init__(self) -> None:
        # Normalise: prefer explicit mode if provided.
        if self.mode:
            self.market_type = str(self.mode).strip().lower()
        else:
            self.market_type = str(self.market_type).strip().lower()

        if self.market_type not in {"spot", "futures"}:
            raise ValueError(f"Unsupported market_type/mode: {self.market_type!r}")
        if self.lookback <= 0:
            raise ValueError("lookback must be > 0")


class TradingEnv(gym.Env):
    """
    Simple trading environment:
      - Observations: lookback window of features (default uses close returns only)
      - Actions:
          spot: 0=hold, 1=long, 2=flat
          futures: 0=hold, 1=long, 2=short, 3=flat
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: dict[str, np.ndarray],
        cfg: TradingEnvConfig,
        seed: int | None = None,
        feature_list: list[str] | None = None,
    ) -> None:
        super().__init__()
        if seed is None:
            seed = int(getattr(cfg, "seed", 0) or 0)
        seed = int(seed)

        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        # Required arrays
        self.open_time = np.asarray(data["open_time_ms"], dtype=np.int64)
        self.close = np.asarray(data["close"], dtype=np.float64)

        # Optional features matrix
        self.features = None
        if "features" in data:
            self.features = np.asarray(data["features"], dtype=np.float64)

        self.feature_list = feature_list

        # Boundaries
        if cfg.end_index <= 0 or cfg.end_index > self.close.size:
            self.cfg.end_index = int(self.close.size)
        if not (0 <= cfg.start_index < cfg.end_index):
            self.cfg.start_index = 0

        self.t = int(self.cfg.start_index)
        self.done = False

        # State
        self.equity = float(cfg.initial_equity)
        self.cash = float(cfg.initial_equity)
        self.asset_qty = 0.0  # spot holdings
        self.futures_pos = 0.0  # futures position qty (+ long, - short)
        self.entry_price = 0.0
        self.fee_total = 0.0
        self.slip_total = 0.0

        # Action space
        if self.cfg.market_type == "spot":
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Discrete(4)

        # Observation space (lookback x dim)
        obs_dim = 1
        if self.features is not None:
            obs_dim = int(self.features.shape[1])

        self.obs_dim = obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.cfg.lookback, obs_dim), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        self.t = int(self.cfg.start_index)
        self.done = False

        self.equity = float(self.cfg.initial_equity)
        self.cash = float(self.cfg.initial_equity)
        self.asset_qty = 0.0
        self.futures_pos = 0.0
        self.entry_price = 0.0
        self.fee_total = 0.0
        self.slip_total = 0.0

        obs = self._get_obs()
        info: dict[str, Any] = {"equity": float(self.equity), "t": int(self.t)}
        return obs, info

    def _price(self) -> float:
        # Clamp to avoid out-of-bounds when algorithms step exactly at the boundary.
        idx = min(max(int(self.t), 0), int(len(self.close) - 1))
        return float(self.close[idx])

    def _mark_to_market(self, price: float) -> float:
        if self.cfg.market_type == "spot":
            return float(self.cash + self.asset_qty * price)
        # futures equity: collateral = cash; pnl = pos*(price-entry)
        pnl = float(self.futures_pos * (price - self.entry_price))
        return float(self.cash + pnl)

    def _apply_fee_and_slip(self, notional: float) -> tuple[float, float]:
        fee = abs(notional) * (self.cfg.fee_bps / 10_000.0)
        slip = abs(notional) * (self.cfg.slippage_bps / 10_000.0)
        self.fee_total += fee
        self.slip_total += slip
        return fee, slip

    def _get_obs(self) -> np.ndarray:
        lb = int(self.cfg.lookback)
        t0 = int(max(self.t - lb + 1, 0))
        t1 = int(self.t + 1)

        if self.features is not None:
            window = self.features[t0:t1]
        else:
            # fallback: use close-to-close log returns as 1D feature
            closes = self.close[t0:t1]
            if closes.size <= 1:
                window = np.zeros((t1 - t0, 1), dtype=np.float64)
            else:
                r = np.diff(np.log(np.maximum(closes, 1e-12)), prepend=np.log(np.maximum(closes[0], 1e-12)))
                window = r.reshape(-1, 1).astype(np.float64)

        # pad at the top if needed
        if window.shape[0] < lb:
            pad = np.zeros((lb - window.shape[0], window.shape[1]), dtype=np.float64)
            window = np.vstack([pad, window])

        return window.astype(np.float32, copy=False)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("step() called after done=True. Call reset().")

        price = self._price()
        prev_equity = self._mark_to_market(price)

        target: float | None = None
        mt = self.cfg.market_type

        if mt == "spot":
            # 0 hold, 1 long, 2 flat
            if action == 1:
                target = 1.0
            elif action == 2:
                target = 0.0
        else:
            # 0 hold, 1 long, 2 short, 3 flat
            if action == 1:
                target = 1.0
            elif action == 2:
                target = -1.0
            elif action == 3:
                target = 0.0

        if target is not None:
            if mt == "spot":
                # target in [0..1] of equity allocated to asset
                equity = self._mark_to_market(price)
                desired_notional = float(target * equity)
                current_notional = float(self.asset_qty * price)
                delta_notional = desired_notional - current_notional

                # buy/sell delta_notional worth
                fee, slip = self._apply_fee_and_slip(delta_notional)
                effective_delta = delta_notional - np.sign(delta_notional) * (fee + slip)

                self.cash -= effective_delta
                self.asset_qty += effective_delta / max(price, 1e-12)

            else:
                # futures target in [-1..1] fraction of equity as notional, scaled by leverage
                equity = self._mark_to_market(price)
                desired_notional = float(target * equity * self.cfg.futures_leverage)
                current_notional = float(self.futures_pos * price)
                delta_notional = desired_notional - current_notional

                fee, slip = self._apply_fee_and_slip(delta_notional)

                # Update position qty
                self.futures_pos += (delta_notional / max(price, 1e-12))
                if abs(self.futures_pos) < 1e-12:
                    self.futures_pos = 0.0
                    self.entry_price = 0.0
                else:
                    # if flipping or new pos, set entry to current price
                    self.entry_price = float(price)

                # fees/slip paid from collateral
                self.cash -= float(fee + slip)

        # advance time
        self.t += 1
        terminal = self.t >= self.cfg.end_index
        if terminal:
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
        info = {
            "equity": float(curr_equity),
            "t": int(self.t),
            "price": float(curr_price),
            "fee_total": float(self.fee_total),
            "slippage_total": float(self.slip_total),
        }
        return obs, float(reward), self.done, False, info
