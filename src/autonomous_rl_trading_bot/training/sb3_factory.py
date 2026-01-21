from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from stable_baselines3 import DQN, PPO  # type: ignore
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import VecEnv
except ModuleNotFoundError as e:  # pragma: no cover
    raise RuntimeError(
        "stable-baselines3 is required for training. Install deps: pip install -e ."
    ) from e


@dataclass(frozen=True)
class SB3FactoryConfig:
    algo: str  # ppo|dqn
    tensorboard_log: str | None
    seed: int = 0
    verbose: int = 1
    device: str = "cpu"  # Force CPU for MLP policies to avoid GPU warnings
    algo_params: dict[str, Any] | None = None


def create_model(*, env: VecEnv, cfg: SB3FactoryConfig):
    """
    Create an SB3 model with sensible defaults. Hyperparameters can be supplied via cfg.algo_params.
    """
    algo = (cfg.algo or "").strip().lower()
    params = dict(cfg.algo_params or {})
    set_random_seed(int(cfg.seed))

    # Force CPU if device is "auto" to avoid GPU warnings for MLP policies
    device = "cpu" if cfg.device == "auto" else str(cfg.device)
    
    common = dict(
        policy="MlpPolicy",
        env=env,
        verbose=int(cfg.verbose),
        tensorboard_log=cfg.tensorboard_log,
        seed=int(cfg.seed),
        device=device,
    )

    if algo == "ppo":
        return PPO(**common, **params)
    if algo == "dqn":
        return DQN(**common, **params)

    raise ValueError(f"Unknown algo={algo!r}. Expected ppo|dqn.")

