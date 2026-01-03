from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

from autonomous_rl_trading_bot.envs.base_env import TradingEnvBase, calc_drawdown, flatten_obs, window_view
from autonomous_rl_trading_bot.envs.fees_slippage import execute_spot_buy, execute_spot_sell
from autonomous_rl_trading_bot.envs.rewards import log_equity_return


@dataclass(frozen=True, slots=True)
class SpotEnvConfig:
    """Config for SpotEnv.

    Action space (Discrete=4):
      0 HOLD
      1 BUY
      2 SELL
      3 CLOSE (sell all)
    """

    lookback: int = 64
    initial_cash: float = 1000.0
    order_size_quote: float = 0.0  # <=0 => all-in / sell-all
    taker_fee_rate: float = 0.001
    slippage_bps: float = 5.0

    # Termination safety
    max_drawdown: float = 0.5  # 50%
    min_equity: float = 1e-9

    # Reward shaping
    reward_type: str = "log_return"  # log_return | pnl


@dataclass
class SpotPortfolio:
    cash: float
    qty_base: float
    avg_entry_price: float
    peak_equity: float

    def equity(self, price: float) -> float:
        return float(self.cash + self.qty_base * float(price))

    def exposure(self, price: float) -> float:
        eq = self.equity(price)
        if eq <= 0.0:
            return 0.0
        return float((self.qty_base * float(price)) / eq)

    def unrealized_pnl(self, price: float) -> float:
        if self.qty_base <= 0.0:
            return 0.0
        return float((float(price) - float(self.avg_entry_price)) * self.qty_base)


def _load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    arrays = dict(np.load(npz_path, allow_pickle=False))
    if "close" not in arrays:
        raise ValueError(f"dataset.npz missing required key 'close': {npz_path}")
    return arrays


class SpotEnv(TradingEnvBase):
    """Gymnasium environment for spot trading.

    Observation:
      - flattened lookback window of chosen feature columns (default: return, log_return)
      - appended account vector: [cash_norm, pos_value_norm, equity_norm, exposure, drawdown, u_pnl_norm]
    """

    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3

    def __init__(
        self,
        *,
        close: np.ndarray,
        features: np.ndarray,
        open_time_ms: Optional[np.ndarray] = None,
        cfg: SpotEnvConfig = SpotEnvConfig(),
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed)

        close = np.asarray(close, dtype=np.float64)
        features = np.asarray(features, dtype=np.float64)

        if close.ndim != 1:
            raise ValueError(f"close must be 1D, got shape={close.shape}")
        if features.ndim != 2:
            raise ValueError(f"features must be 2D, got shape={features.shape}")
        if features.shape[0] != close.shape[0]:
            raise ValueError("features and close must have same length")
        if close.shape[0] < 2:
            raise ValueError("dataset too short")
        if cfg.lookback < 2:
            raise ValueError("lookback must be >= 2")
        if close.shape[0] < cfg.lookback + 1:
            raise ValueError(
                f"Need at least lookback+1 rows for stepping. lookback={cfg.lookback} rows={close.shape[0]}"
            )

        self.cfg = cfg
        self.close = close
        self.features = features
        self.open_time_ms = None if open_time_ms is None else np.asarray(open_time_ms, dtype=np.int64)

        self._n_features = int(features.shape[1])
        self._account_dim = 6
        self._obs_dim = int(cfg.lookback * self._n_features + self._account_dim)

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        self._i: int = 0
        self._portfolio: SpotPortfolio

    @staticmethod
    def from_dataset_dir(
        dataset_dir: Path,
        *,
        cfg: SpotEnvConfig = SpotEnvConfig(),
        seed: int = 0,
        feature_keys: Optional[List[str]] = None,
    ) -> "SpotEnv":
        """Load dataset.npz (+ optional meta.json) from a dataset directory."""
        dataset_dir = Path(dataset_dir)
        meta_path = dataset_dir / "meta.json"
        npz_path = dataset_dir / "dataset.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"dataset.npz not found: {npz_path}")

        arrays = _load_npz(npz_path)

        if feature_keys is None:
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                feature_keys = list(meta.get("features") or meta.get("columns") or [])
            else:
                feature_keys = ["return", "log_return"]

        feature_keys = [k for k in feature_keys if k not in ("open_time_ms", "open", "high", "low", "close", "volume")]
        if not feature_keys:
            feature_keys = ["log_return"] if "log_return" in arrays else []

        feats: List[np.ndarray] = []
        for k in feature_keys:
            if k not in arrays:
                continue
            v = np.asarray(arrays[k], dtype=np.float64)
            if v.ndim != 1:
                continue
            feats.append(v.reshape(-1, 1))

        if not feats:
            c = np.asarray(arrays["close"], dtype=np.float64)
            lr = np.zeros_like(c)
            lr[1:] = np.log((c[1:] + 1e-12) / (c[:-1] + 1e-12))
            feats = [lr.reshape(-1, 1)]

        features = np.concatenate(feats, axis=1)
        open_time = arrays.get("open_time_ms", None)
        return SpotEnv(close=arrays["close"], features=features, open_time_ms=open_time, cfg=cfg, seed=seed)

    def reset(self, *, seed: Optional[int] = None, options=None):  # type: ignore[override]
        super().reset(seed=seed, options=options)

        self._i = int(self.cfg.lookback - 1)
        self._portfolio = SpotPortfolio(
            cash=float(self.cfg.initial_cash),
            qty_base=0.0,
            avg_entry_price=0.0,
            peak_equity=float(self.cfg.initial_cash),
        )

        obs = self._make_obs()
        info = self._info_dict(action=None, trade=None, killed=False, reason="reset")
        return obs, info

    def step(self, action: int):  # type: ignore[override]
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        price = float(self.close[self._i])
        equity_prev = self._portfolio.equity(price)

        trade = None
        executed = False

        if action == self.BUY:
            r = execute_spot_buy(
                cash=self._portfolio.cash,
                qty_base=self._portfolio.qty_base,
                avg_entry=self._portfolio.avg_entry_price,
                mid_price=price,
                order_size_quote=float(self.cfg.order_size_quote),
                taker_fee_rate=float(self.cfg.taker_fee_rate),
                slippage_bps=float(self.cfg.slippage_bps),
            )
            if r is not None:
                executed = True
                self._portfolio.cash = r.cash_after
                self._portfolio.qty_base = r.qty_after
                self._portfolio.avg_entry_price = r.avg_entry_after
                trade = {
                    "side": "BUY",
                    "qty_base": r.qty_base,
                    "fill_price": r.fill_price,
                    "notional": r.notional_quote,
                    "fee": r.fee_quote,
                    "slippage_cost": r.slippage_cost_quote,
                }

        elif action == self.SELL:
            r = execute_spot_sell(
                cash=self._portfolio.cash,
                qty_base=self._portfolio.qty_base,
                avg_entry=self._portfolio.avg_entry_price,
                mid_price=price,
                order_size_quote=float(self.cfg.order_size_quote),
                taker_fee_rate=float(self.cfg.taker_fee_rate),
                slippage_bps=float(self.cfg.slippage_bps),
                close_all=False,
            )
            if r is not None:
                executed = True
                self._portfolio.cash = r.cash_after
                self._portfolio.qty_base = r.qty_after
                self._portfolio.avg_entry_price = r.avg_entry_after
                trade = {
                    "side": "SELL",
                    "qty_base": r.qty_base,
                    "fill_price": r.fill_price,
                    "notional": r.notional_quote,
                    "fee": r.fee_quote,
                    "slippage_cost": r.slippage_cost_quote,
                }

        elif action == self.CLOSE:
            r = execute_spot_sell(
                cash=self._portfolio.cash,
                qty_base=self._portfolio.qty_base,
                avg_entry=self._portfolio.avg_entry_price,
                mid_price=price,
                order_size_quote=float(self.cfg.order_size_quote),
                taker_fee_rate=float(self.cfg.taker_fee_rate),
                slippage_bps=float(self.cfg.slippage_bps),
                close_all=True,
            )
            if r is not None:
                executed = True
                self._portfolio.cash = r.cash_after
                self._portfolio.qty_base = r.qty_after
                self._portfolio.avg_entry_price = r.avg_entry_after
                trade = {
                    "side": "CLOSE",
                    "qty_base": r.qty_base,
                    "fill_price": r.fill_price,
                    "notional": r.notional_quote,
                    "fee": r.fee_quote,
                    "slippage_cost": r.slippage_cost_quote,
                }

        self._i += 1
        if self._i >= len(self.close):
            self._i = len(self.close) - 1

        price_next = float(self.close[self._i])
        equity_next = self._portfolio.equity(price_next)
        self._portfolio.peak_equity = max(self._portfolio.peak_equity, equity_next)

        if self.cfg.reward_type == "pnl":
            reward = float(equity_next - equity_prev)
        else:
            reward = log_equity_return(equity_prev, equity_next)

        dd = calc_drawdown(equity_next, self._portfolio.peak_equity)
        killed = bool(dd >= float(self.cfg.max_drawdown) or equity_next <= float(self.cfg.min_equity))
        reason = "drawdown" if dd >= float(self.cfg.max_drawdown) else ("min_equity" if equity_next <= float(self.cfg.min_equity) else "")

        terminated = bool(self._i >= len(self.close) - 1 or killed)
        truncated = False

        obs = self._make_obs()
        info = self._info_dict(action=int(action), trade=trade, killed=killed, reason=reason)
        info["executed"] = bool(executed)

        return obs, float(reward), terminated, truncated, info

    def _account_vec(self) -> np.ndarray:
        price = float(self.close[self._i])
        eq = self._portfolio.equity(price)
        pos_val = float(self._portfolio.qty_base * price)
        cash = float(self._portfolio.cash)
        dd = calc_drawdown(eq, self._portfolio.peak_equity)
        u = self._portfolio.unrealized_pnl(price)
        init = max(1e-12, float(self.cfg.initial_cash))
        return np.array(
            [
                cash / init,
                pos_val / init,
                eq / init,
                self._portfolio.exposure(price),
                dd,
                u / init,
            ],
            dtype=np.float32,
        )

    def _make_obs(self) -> np.ndarray:
        w = window_view(self.features, self._i, int(self.cfg.lookback))
        return flatten_obs(w, self._account_vec())

    def _info_dict(self, *, action: Optional[int], trade: Optional[Dict[str, Any]], killed: bool, reason: str) -> Dict[str, Any]:
        price = float(self.close[self._i])
        ts = None
        if self.open_time_ms is not None:
            ts = int(self.open_time_ms[self._i])

        eq = self._portfolio.equity(price)
        info: Dict[str, Any] = {
            "i": int(self._i),
            "ts_ms": ts,
            "price": float(price),
            "cash": float(self._portfolio.cash),
            "qty_base": float(self._portfolio.qty_base),
            "avg_entry": float(self._portfolio.avg_entry_price),
            "equity": float(eq),
            "exposure": float(self._portfolio.exposure(price)),
            "drawdown": float(calc_drawdown(eq, self._portfolio.peak_equity)),
            "killed": bool(killed),
            "kill_reason": str(reason or ""),
        }
        if action is not None:
            info["action"] = int(action)
        if trade is not None:
            info["trade"] = trade
        return info

