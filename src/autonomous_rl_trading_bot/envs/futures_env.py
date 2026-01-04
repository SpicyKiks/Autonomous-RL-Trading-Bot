from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from autonomous_rl_trading_bot.envs.base_env import TradingEnvBase, calc_drawdown, flatten_obs, window_view
from autonomous_rl_trading_bot.envs.fees_slippage import apply_slippage, taker_fee
from autonomous_rl_trading_bot.envs.rewards import log_equity_return


@dataclass(frozen=True, slots=True)
class FuturesEnvConfig:
    """
    USD-M futures style single-symbol env.

    Position uses signed contracts/base units:
      qty > 0 => long
      qty < 0 => short

    Actions (Discrete=5):
      0 HOLD
      1 LONG   (open/increase long)
      2 SHORT  (open/increase short)
      3 REDUCE (reduce position)
      4 CLOSE  (close all)
    """

    lookback: int = 64

    # Margin model
    initial_cash: float = 1000.0
    leverage: float = 3.0
    maintenance_margin_rate: float = 0.005
    allow_short: bool = True
    stop_on_liquidation: bool = True

    # Execution
    order_size_quote: float = 0.0  # <=0 => "all-in" using free margin
    reduce_fraction: float = 0.5   # used when order_size_quote<=0 for REDUCE
    taker_fee_rate: float = 0.001
    slippage_bps: float = 5.0

    # Optional termination (default keeps liquidation as the main terminal event)
    # Set to something like 0.30 if you want earlier safety stops.
    max_drawdown: float = 0.999
    min_equity: float = 1e-9

    # Reward
    reward_type: str = "log_return"  # log_return | pnl


@dataclass
class FuturesPortfolio:
    margin_balance: float
    qty: float
    entry_price: float
    peak_equity: float

    def unrealized_pnl(self, price: float) -> float:
        if abs(self.qty) <= 1e-18:
            return 0.0
        return float(self.qty * (float(price) - float(self.entry_price)))

    def equity(self, price: float) -> float:
        return float(self.margin_balance + self.unrealized_pnl(price))

    def position_notional(self, price: float) -> float:
        return float(abs(self.qty) * float(price))

    def used_margin(self, price: float, leverage: float) -> float:
        lev = max(1e-9, float(leverage))
        return float(self.position_notional(price) / lev)

    def maintenance_margin(self, price: float, mmr: float) -> float:
        return float(self.position_notional(price) * max(0.0, float(mmr)))


def _load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    arrays = dict(np.load(npz_path, allow_pickle=False))
    if "close" not in arrays:
        raise ValueError(f"dataset.npz missing required key 'close': {npz_path}")
    return arrays


class FuturesEnv(TradingEnvBase):
    HOLD = 0
    LONG = 1
    SHORT = 2
    REDUCE = 3
    CLOSE = 4

    def __init__(
        self,
        *,
        close: np.ndarray,
        features: np.ndarray,
        open_time_ms: Optional[np.ndarray] = None,
        cfg: FuturesEnvConfig = FuturesEnvConfig(),
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
        if close.shape[0] < cfg.lookback + 1:
            raise ValueError(
                f"Need at least lookback+1 rows. lookback={cfg.lookback} rows={close.shape[0]}"
            )
        if cfg.leverage <= 0.0:
            raise ValueError("leverage must be > 0")

        self.cfg = cfg
        self.close = close
        self.features = features
        self.open_time_ms = None if open_time_ms is None else np.asarray(open_time_ms, dtype=np.int64)

        self._n_features = int(features.shape[1])
        self._account_dim = 9
        self._obs_dim = int(cfg.lookback * self._n_features + self._account_dim)

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )

        self._i: int = 0
        self._p: FuturesPortfolio

    @staticmethod
    def from_dataset_dir(
        dataset_dir: Path,
        *,
        cfg: FuturesEnvConfig = FuturesEnvConfig(),
        seed: int = 0,
        feature_keys: Optional[List[str]] = None,
    ) -> "FuturesEnv":
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

        # remove raw candle columns from features
        feature_keys = [
            k
            for k in feature_keys
            if k not in ("open_time_ms", "open", "high", "low", "close", "volume")
        ]
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
        return FuturesEnv(
            close=arrays["close"],
            features=features,
            open_time_ms=open_time,
            cfg=cfg,
            seed=seed,
        )

    def reset(self, *, seed: Optional[int] = None, options=None):  # type: ignore[override]
        super().reset(seed=seed, options=options)

        self._i = int(self.cfg.lookback - 1)
        self._p = FuturesPortfolio(
            margin_balance=float(self.cfg.initial_cash),
            qty=0.0,
            entry_price=0.0,
            peak_equity=float(self.cfg.initial_cash),
        )

        obs = self._make_obs()
        info = self._info_dict(action=None, trade=None, liquidated=False, killed=False, reason="reset")
        return obs, info

    def step(self, action: int):  # type: ignore[override]
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        price = float(self.close[self._i])
        equity_prev = self._p.equity(price)

        executed = False
        trade: Optional[Dict[str, Any]] = None

        if action == self.LONG:
            executed, trade = self._action_open(direction=+1, price=price)
        elif action == self.SHORT:
            if self.cfg.allow_short:
                executed, trade = self._action_open(direction=-1, price=price)
        elif action == self.REDUCE:
            executed, trade = self._action_reduce(price=price)
        elif action == self.CLOSE:
            executed, trade = self._action_close(price=price)

        # advance time
        self._i += 1
        if self._i >= len(self.close):
            self._i = len(self.close) - 1

        price_next = float(self.close[self._i])
        equity_next = self._p.equity(price_next)
        self._p.peak_equity = max(self._p.peak_equity, equity_next)

        # liquidation check (mark-to-market at next candle)
        mm = self._p.maintenance_margin(price_next, float(self.cfg.maintenance_margin_rate))
        liquidated = bool(abs(self._p.qty) > 1e-18 and equity_next <= mm)

        if liquidated:
            # wipe position
            self._p.qty = 0.0
            self._p.entry_price = 0.0
            # (simple model) keep remaining margin_balance as-is
            equity_next = self._p.equity(price_next)

        # reward
        if self.cfg.reward_type == "pnl":
            reward = float(equity_next - equity_prev)
        else:
            reward = log_equity_return(equity_prev, equity_next)

        # optional kill switch
        dd = calc_drawdown(equity_next, self._p.peak_equity)
        kill_dd_enabled = float(self.cfg.max_drawdown) < 0.999
        killed = bool(
            (kill_dd_enabled and dd >= float(self.cfg.max_drawdown))
            or (equity_next <= float(self.cfg.min_equity))
        )

        reason = ""
        if liquidated:
            reason = "liquidation"
        elif kill_dd_enabled and dd >= float(self.cfg.max_drawdown):
            reason = "drawdown"
        elif equity_next <= float(self.cfg.min_equity):
            reason = "min_equity"

        terminated = bool(
            self._i >= len(self.close) - 1
            or (liquidated and self.cfg.stop_on_liquidation)
            or killed
        )
        truncated = False

        obs = self._make_obs()
        info = self._info_dict(
            action=int(action),
            trade=trade,
            liquidated=liquidated,
            killed=killed,
            reason=reason,
        )
        info["executed"] = bool(executed)

        return obs, float(reward), terminated, truncated, info

    # ---------------- actions ----------------

    def _action_open(self, *, direction: int, price: float) -> Tuple[bool, Optional[Dict[str, Any]]]:
        direction = 1 if direction >= 0 else -1
        qty0 = float(self._p.qty)

        # If flipping, close first (so we don't do weird partial flip accounting)
        if abs(qty0) > 1e-18 and np.sign(qty0) != direction:
            self._move_position_towards(target_qty=0.0, price=price)
            qty0 = 0.0

        equity = self._p.equity(price)
        if equity <= 0.0:
            return False, None

        if self.cfg.order_size_quote <= 0.0:
            # all-in using free margin
            used = self._p.used_margin(price, float(self.cfg.leverage))
            free = max(0.0, equity - used)
            target_notional = free * float(self.cfg.leverage)
            target_qty_abs = target_notional / max(1e-12, price)
            target_qty = float(direction) * target_qty_abs
        else:
            delta_qty = float(direction) * (float(self.cfg.order_size_quote) / max(1e-12, price))
            target_qty = qty0 + delta_qty

        return self._move_position_towards(target_qty=target_qty, price=price)

    def _action_reduce(self, *, price: float) -> Tuple[bool, Optional[Dict[str, Any]]]:
        qty0 = float(self._p.qty)
        if abs(qty0) <= 1e-18:
            return False, None

        notional0 = abs(qty0) * price
        if self.cfg.order_size_quote > 0.0:
            reduce_notional = min(float(self.cfg.order_size_quote), notional0)
            reduce_qty = reduce_notional / max(1e-12, price)
        else:
            frac = float(self.cfg.reduce_fraction)
            if not (0.0 < frac < 1.0):
                frac = 0.5
            reduce_qty = abs(qty0) * frac

        target_qty = qty0 - np.sign(qty0) * reduce_qty
        if abs(target_qty) <= 1e-12:
            target_qty = 0.0

        return self._move_position_towards(target_qty=target_qty, price=price)

    def _action_close(self, *, price: float) -> Tuple[bool, Optional[Dict[str, Any]]]:
        qty0 = float(self._p.qty)
        if abs(qty0) <= 1e-18:
            return False, None
        return self._move_position_towards(target_qty=0.0, price=price)

    def _move_position_towards(self, *, target_qty: float, price: float) -> Tuple[bool, Optional[Dict[str, Any]]]:
        qty0 = float(self._p.qty)
        target_qty = float(target_qty)

        if abs(target_qty - qty0) <= 1e-12:
            return False, None

        delta = target_qty - qty0
        side = "buy" if delta > 0 else "sell"
        fill = apply_slippage(price, side, float(self.cfg.slippage_bps))
        if fill <= 0.0:
            return False, None

        equity_pre = self._p.equity(price)

        # clamp by margin (single-pass)
        notional_desired = abs(delta) * fill
        fee_desired = taker_fee(notional_desired, float(self.cfg.taker_fee_rate))
        max_total_abs = max(
            0.0, (equity_pre - fee_desired) * float(self.cfg.leverage) / max(1e-12, fill)
        )

        if abs(target_qty) > max_total_abs + 1e-12:
            target_qty = float(np.sign(target_qty) * max_total_abs)
            delta = target_qty - qty0
            if abs(delta) <= 1e-12:
                return False, None
            side = "buy" if delta > 0 else "sell"
            fill = apply_slippage(price, side, float(self.cfg.slippage_bps))
            notional_desired = abs(delta) * fill
            fee_desired = taker_fee(notional_desired, float(self.cfg.taker_fee_rate))

        # Realize PnL for the closed portion
        realized = 0.0
        entry0 = float(self._p.entry_price)

        if abs(qty0) > 1e-18 and np.sign(delta) != np.sign(qty0):
            closed_abs = min(abs(qty0), abs(delta))
            closed_signed = float(np.sign(qty0) * closed_abs)
            realized = float(closed_signed * (fill - entry0))

        # update collateral
        self._p.margin_balance = float(self._p.margin_balance + realized - fee_desired)

        # update position & entry
        qty1 = float(qty0 + delta)
        if abs(qty1) <= 1e-12:
            qty1 = 0.0
            entry1 = 0.0
        else:
            # increased in same direction
            if abs(qty0) <= 1e-18 or np.sign(qty0) == np.sign(delta):
                if abs(qty0) <= 1e-18:
                    entry1 = fill
                else:
                    entry1 = (abs(qty0) * entry0 + abs(delta) * fill) / max(1e-12, abs(qty1))
            else:
                # reduced but not flipped
                entry1 = entry0

        self._p.qty = float(qty1)
        self._p.entry_price = float(entry1)

        trade = {
            "side": "BUY" if side == "buy" else "SELL",
            "qty_delta": float(delta),
            "fill_price": float(fill),
            "notional": float(abs(delta) * fill),
            "fee": float(fee_desired),
            "realized_pnl": float(realized),
            "pos_qty_after": float(self._p.qty),
            "entry_after": float(self._p.entry_price),
        }
        return True, trade

    # ---------------- obs/info ----------------

    def _account_vec(self) -> np.ndarray:
        price = float(self.close[self._i])
        eq = self._p.equity(price)
        init = max(1e-12, float(self.cfg.initial_cash))

        used = self._p.used_margin(price, float(self.cfg.leverage))
        free = max(0.0, eq - used)
        pos_notional = self._p.position_notional(price)
        lev_used = 0.0 if eq <= 0.0 else pos_notional / eq
        dd = calc_drawdown(eq, self._p.peak_equity)
        u = self._p.unrealized_pnl(price)

        pos_dir = 0.0 if abs(self._p.qty) <= 1e-18 else float(np.sign(self._p.qty))
        denom = max(1e-12, eq * float(self.cfg.leverage))
        pos_frac = float(pos_notional / denom) if denom > 0 else 0.0

        return np.array(
            [
                self._p.margin_balance / init,
                eq / init,
                used / init,
                free / init,
                pos_frac,
                pos_dir,
                lev_used,
                dd,
                u / init,
            ],
            dtype=np.float32,
        )

    def _make_obs(self) -> np.ndarray:
        w = window_view(self.features, self._i, int(self.cfg.lookback))
        return flatten_obs(w, self._account_vec())

    def _info_dict(
        self,
        *,
        action: Optional[int],
        trade: Optional[Dict[str, Any]],
        liquidated: bool,
        killed: bool,
        reason: str,
    ) -> Dict[str, Any]:
        price = float(self.close[self._i])
        ts = None
        if self.open_time_ms is not None:
            ts = int(self.open_time_ms[self._i])

        eq = self._p.equity(price)
        used = self._p.used_margin(price, float(self.cfg.leverage))
        mm = self._p.maintenance_margin(price, float(self.cfg.maintenance_margin_rate))

        info: Dict[str, Any] = {
            "i": int(self._i),
            "ts_ms": ts,
            "price": float(price),
            "margin_balance": float(self._p.margin_balance),
            "qty": float(self._p.qty),
            "entry_price": float(self._p.entry_price),
            "equity": float(eq),
            "unrealized_pnl": float(self._p.unrealized_pnl(price)),
            "used_margin": float(used),
            "maintenance_margin": float(mm),
            "drawdown": float(calc_drawdown(eq, self._p.peak_equity)),
            "liquidated": bool(liquidated),
            "killed": bool(killed),
            "end_reason": str(reason or ""),
        }
        if action is not None:
            info["action"] = int(action)
        if trade is not None:
            info["trade"] = trade
        return info

