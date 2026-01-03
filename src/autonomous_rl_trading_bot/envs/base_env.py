from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np


@dataclass(frozen=True, slots=True)
class CostModel:
    """Execution cost model.

    taker_fee_rate: fraction, e.g. 0.001 = 10 bps
    slippage_bps: bps applied on market fills.
    """

    taker_fee_rate: float = 0.001
    slippage_bps: float = 5.0


def seed_rng(seed: int | None) -> np.random.Generator:
    if seed is None:
        seed = 0
    return np.random.default_rng(int(seed))


def calc_drawdown(equity: float, peak_equity: float) -> float:
    if peak_equity <= 0.0:
        return 0.0
    return max(0.0, 1.0 - float(equity) / float(peak_equity))


class TradingEnvBase(gym.Env):
    """Base class with consistent seeding + minimal helpers.

    Concrete envs must set:
      - observation_space
      - action_space
    """

    metadata = {"render_modes": []}

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__()
        self._seed = int(seed)
        self._rng = seed_rng(self._seed)

    def reset(self, *, seed: Optional[int] = None, options=None):  # type: ignore[override]
        if seed is not None:
            self._seed = int(seed)
            self._rng = seed_rng(self._seed)
        return super().reset(seed=seed, options=options)


def window_view(x: np.ndarray, i: int, lookback: int) -> np.ndarray:
    """Return x[i-lookback+1 : i+1] (inclusive) as a view/copy.

    Assumes i >= lookback-1.
    """
    start = i - lookback + 1
    end = i + 1
    if start < 0:
        raise ValueError("window_view: i must be >= lookback-1")
    return x[start:end]


def flatten_obs(window_feats: np.ndarray, account_vec: np.ndarray) -> np.ndarray:
    """Flatten a (lookback, n_features) + account vector into 1D float32."""
    if window_feats.ndim != 2:
        raise ValueError(f"window_feats must be 2D, got shape={window_feats.shape}")
    if account_vec.ndim != 1:
        raise ValueError(f"account_vec must be 1D, got shape={account_vec.shape}")

    flat = np.concatenate([window_feats.reshape(-1), account_vec], axis=0)
    return flat.astype(np.float32, copy=False)

