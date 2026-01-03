from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .dataset import LoadedDataset
from .env_trading import TradingEnv, TradingEnvConfig
from .metrics import compute_equity_metrics

# SB3 imports inside function so project can still run even if user env is missing torch temporarily.


@dataclass
class TrainConfig:
    algo: str = "ppo"  # ppo|dqn
    total_timesteps: int = 50_000
    lookback: int = 30
    train_split: Optional[str] = None  # "train"|"val"|"test" or None for legacy float-based
    eval_split: Optional[str] = None  # "train"|"val"|"test" or None for legacy
    train_split_frac: float = 0.8  # Legacy: used if train_split is None
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    initial_equity: float = 1000.0
    position_fraction: float = 1.0
    futures_leverage: float = 3.0
    reward_kind: str = "log_equity"
    tensorboard: bool = True


def _split_indices(n: int, frac: float) -> Tuple[int, int, int, int]:
    frac = float(frac)
    frac = min(max(frac, 0.5), 0.95)
    split = int(n * frac)
    return 0, split, split, n


def _get_split_indices_from_meta(
    meta: Dict[str, Any], train_split: Optional[str], eval_split: Optional[str], n: int
) -> Tuple[int, int, int, int]:
    """
    Get split indices from meta splits or fallback to legacy fraction-based.
    Returns (train_start, train_end, eval_start, eval_end).
    """
    splits = meta.get("splits")
    
    if splits and train_split:
        # Use meta splits
        if train_split not in splits:
            raise ValueError(f"train_split={train_split} not found in dataset splits: {list(splits.keys())}")
        train_info = splits[train_split]
        s0 = train_info["start_idx"]
        s1 = train_info["end_idx"]
        
        if eval_split:
            if eval_split not in splits:
                raise ValueError(f"eval_split={eval_split} not found in dataset splits: {list(splits.keys())}")
            eval_info = splits[eval_split]
            e0 = eval_info["start_idx"]
            e1 = eval_info["end_idx"]
        else:
            # Default eval to val if available, else test
            if "val" in splits:
                eval_info = splits["val"]
            elif "test" in splits:
                eval_info = splits["test"]
            else:
                raise ValueError("No eval split available in dataset")
            e0 = eval_info["start_idx"]
            e1 = eval_info["end_idx"]
    else:
        # Legacy: use fraction-based split
        s0, s1, e0, e1 = _split_indices(n, 0.8)  # Default 0.8
    
    return s0, s1, e0, e1


def _rollout_policy(env: TradingEnv, model) -> Tuple[np.ndarray, list[dict[str, Any]]]:
    obs, _ = env.reset()
    equities = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(int(action))
        equities.append(info["equity"])
    if equities:
        eq = np.asarray(equities, dtype=np.float64)
    else:
        eq = np.asarray([env.cfg.initial_equity], dtype=np.float64)
    return eq, env.trades


def train_and_evaluate(
    dataset: LoadedDataset,
    market_type: str,
    seed: int,
    run_dir: Path,
    cfg: TrainConfig,
) -> Dict[str, Any]:
    data = dataset.data
    n = int(len(data["close"]))
    
    # Get split indices from meta or fallback to legacy
    s0, s1, e0, e1 = _get_split_indices_from_meta(
        dataset.meta, cfg.train_split, cfg.eval_split, n
    )

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

    train_env = TradingEnv(data, train_env_cfg, seed=seed, feature_list=dataset.feature_list)
    eval_env = TradingEnv(data, eval_env_cfg, seed=seed + 1, feature_list=dataset.feature_list)

    algo = cfg.algo.lower().strip()
    tb_dir = run_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)

    try:
        from stable_baselines3 import PPO, DQN
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as ex:
        raise RuntimeError(
            "stable-baselines3 import failed. Ensure dependencies are installed (torch + stable-baselines3)."
        ) from ex

    venv = DummyVecEnv([lambda: train_env])

    if algo == "ppo":
        model = PPO(
            "MlpPolicy",
            venv,
            verbose=0,
            seed=seed,
            tensorboard_log=str(tb_dir) if cfg.tensorboard else None,
        )
    elif algo == "dqn":
        model = DQN(
            "MlpPolicy",
            venv,
            verbose=0,
            seed=seed,
            tensorboard_log=str(tb_dir) if cfg.tensorboard else None,
        )
    else:
        raise ValueError(f"Unsupported algo: {cfg.algo}. Use ppo|dqn.")

    model.learn(total_timesteps=int(cfg.total_timesteps))

    model_path = run_dir / "model.zip"
    model.save(str(model_path))

    equity, trades = _rollout_policy(eval_env, model)
    m = compute_equity_metrics(equity)

    # Write eval outputs
    import pandas as pd

    pd.DataFrame(
        {
            "step": np.arange(equity.size, dtype=np.int64),
            "equity": equity,
        }
    ).to_csv(run_dir / "eval_equity.csv", index=False)

    if trades:
        pd.DataFrame(trades).to_csv(run_dir / "eval_trades.csv", index=False)
    else:
        pd.DataFrame([]).to_csv(run_dir / "eval_trades.csv", index=False)

    metrics = {
        "final_equity": m.final_equity,
        "total_return": m.total_return,
        "max_drawdown": m.max_drawdown,
        "sharpe": m.sharpe,
        "fee_total_eval": float(eval_env.fee_total),
        "slippage_total_eval": float(eval_env.slip_total),
        "eval_steps": int(equity.size),
    }
    (run_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "model_path": str(model_path),
        "tensorboard_dir": str(tb_dir),
        "metrics": metrics,
        "train_split": {"train": [s0, s1], "eval": [e0, e1]},
    }

