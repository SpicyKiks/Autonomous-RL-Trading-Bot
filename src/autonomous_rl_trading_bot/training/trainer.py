from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from autonomous_rl_trading_bot.rl.dataset import LoadedDataset
from autonomous_rl_trading_bot.rl.env_trading import TradingEnv, TradingEnvConfig
from autonomous_rl_trading_bot.rl.metrics import compute_equity_metrics

from .callbacks import CallbackConfig, build_callbacks
from .checkpointing import (
    best_model_path,
    model_path,
    tensorboard_dir,
    write_json,
    write_training_manifest,
)
from .sb3_factory import SB3FactoryConfig, create_model

try:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
except ModuleNotFoundError as e:  # pragma: no cover
    raise RuntimeError(
        "stable-baselines3 is required for training. Install deps: pip install -e ."
    ) from e


@dataclass
class TrainConfig:
    algo: str = "ppo"  # ppo|dqn
    total_timesteps: int = 50_000
    lookback: int = 30
    train_split: float = 0.8

    fee_bps: float = 10.0
    slippage_bps: float = 5.0

    initial_equity: float = 1000.0
    position_fraction: float = 1.0
    futures_leverage: float = 3.0

    reward_kind: str = "log_equity"  # log_equity|delta_equity
    tensorboard: bool = True

    checkpoint_freq: int = 10_000
    eval_freq: int = 10_000

    # Optional algo hyperparams (merged into SB3 ctor kwargs)
    algo_params: Dict[str, Any] | None = None


def _split_indices(n: int, frac: float) -> Tuple[int, int, int, int]:
    frac = float(frac)
    frac = min(max(frac, 0.5), 0.95)
    split = int(n * frac)
    return 0, split, split, n


def _make_env(data: Dict[str, np.ndarray], cfg: TradingEnvConfig, seed: int, feature_list: Optional[list[str]] = None) -> TradingEnv:
    return TradingEnv(data, cfg, seed=seed, feature_list=feature_list)


def _rollout_policy(env: TradingEnv, model) -> tuple[np.ndarray, list[dict]]:
    obs, _ = env.reset()
    equity_curve = [float(env.equity)]
    done = False
    while not done:
        action, _ = model.predict(obs.astype(np.float32, copy=False), deterministic=True)
        obs, _, done, _, info = env.step(int(action))
        equity_curve.append(float(info.get("equity", env.equity)))
    return np.asarray(equity_curve, dtype=np.float64), list(env.trades)


def train_and_evaluate(
    *,
    dataset: LoadedDataset,
    market_type: str,
    seed: int,
    run_dir: Path,
    cfg: TrainConfig,
    project_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    data = dataset.data
    n = int(len(data["close"]))
    s0, s1, e0, e1 = _split_indices(n, cfg.train_split)

    train_env_cfg = TradingEnvConfig(
        market_type=market_type,
        lookback=cfg.lookback,
        fee_bps=cfg.fee_bps,
        slippage_bps=cfg.slippage_bps,
        initial_equity=cfg.initial_equity,
        position_fraction=cfg.position_fraction,
        futures_leverage=cfg.futures_leverage,
        reward_kind=cfg.reward_kind,
        start_index=s0,
        end_index=s1,
    )
    eval_env_cfg = TradingEnvConfig(
        market_type=market_type,
        lookback=cfg.lookback,
        fee_bps=cfg.fee_bps,
        slippage_bps=cfg.slippage_bps,
        initial_equity=cfg.initial_equity,
        position_fraction=cfg.position_fraction,
        futures_leverage=cfg.futures_leverage,
        reward_kind=cfg.reward_kind,
        start_index=e0,
        end_index=e1,
    )

    feature_list = getattr(dataset, 'feature_list', None)

    def make_train():
        return _make_env(data, train_env_cfg, seed, feature_list)

    def make_eval():
        return _make_env(data, eval_env_cfg, seed + 1, feature_list)

    train_vec = VecMonitor(DummyVecEnv([make_train]))
    eval_vec = VecMonitor(DummyVecEnv([make_eval]))

    tb_dir = str(tensorboard_dir(run_dir)) if cfg.tensorboard else None

    # Merge algo params from config file if present
    algo_params: Dict[str, Any] = {}
    if project_cfg:
        tb = project_cfg.get("training", {}) or {}
        sb3 = tb.get("sb3", {}) or {}
        ap = sb3.get((cfg.algo or "").strip().lower(), {}) or {}
        if isinstance(ap, dict):
            algo_params.update(ap)
    if cfg.algo_params:
        algo_params.update(cfg.algo_params)

    model = create_model(
        env=train_vec,
        cfg=SB3FactoryConfig(
            algo=cfg.algo,
            tensorboard_log=tb_dir,
            seed=seed,
            verbose=1,
            device="auto",
            algo_params=algo_params,
        ),
    )

    cb = build_callbacks(
        run_dir=run_dir,
        eval_env=eval_vec,
        cfg=CallbackConfig(
            checkpoint_freq=int(cfg.checkpoint_freq),
            eval_freq=int(cfg.eval_freq),
            n_eval_episodes=1,
            deterministic_eval=True,
            verbose=1,
        ),
    )

    model.learn(total_timesteps=int(cfg.total_timesteps), callback=cb)

    # Save final model
    mpath = model_path(run_dir)
    model.save(str(mpath))

    # Evaluate best model if it exists
    best_path = best_model_path(run_dir)
    eval_model = model
    best_model_str: Optional[str] = None
    if best_path.exists():
        best_model_str = str(best_path)
        try:
            eval_model = type(model).load(best_model_str)
        except Exception:
            eval_model = model

    # Manual evaluation to get equity curve + trades
    eval_env = _make_env(data, eval_env_cfg, seed + 123, feature_list)
    equity, trades = _rollout_policy(eval_env, eval_model)

    import pandas as pd

    pd.DataFrame({"step": np.arange(equity.size, dtype=np.int64), "equity": equity}).to_csv(
        run_dir / "eval_equity.csv", index=False
    )
    pd.DataFrame(trades).to_csv(run_dir / "eval_trades.csv", index=False)

    m = compute_equity_metrics(equity)
    metrics = {
        "final_equity": m.final_equity,
        "total_return": m.total_return,
        "max_drawdown": m.max_drawdown,
        "sharpe": m.sharpe,
        "fee_total_eval": float(eval_env.fee_total),
        "slippage_total_eval": float(eval_env.slip_total),
        "eval_steps": int(equity.size),
        "algo_params": algo_params,
    }
    write_json(run_dir / "eval_metrics.json", metrics)

    split = {"train": [s0, s1], "eval": [e0, e1]}
    write_training_manifest(
        run_dir=run_dir,
        train_params=asdict(cfg),
        split=split,
        dataset_meta=dataset.meta,
        extra={"market_type": market_type},
    )

    return {
        "model_path": str(mpath),
        "best_model_path": best_model_str,
        "tensorboard_dir": tb_dir,
        "metrics": metrics,
        "train_split": split,
    }

