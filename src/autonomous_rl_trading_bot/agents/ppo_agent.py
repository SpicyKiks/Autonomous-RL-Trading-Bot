from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.utils import set_random_seed


def create_ppo(
    env: Union[Any, VecEnv],
    *,
    seed: int = 42,
    tensorboard_log_dir: Optional[str] = None,
    n_steps: int = 2048,
    batch_size: int = 64,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    learning_rate: float = 3e-4,
    clip_range: float = 0.2,
    n_epochs: int = 10,
    **kwargs: Any,
) -> PPO:
    """
    Create and configure PPO agent for time-series RL.
    
    Args:
        env: Gymnasium environment or VecEnv (if VecEnv, used directly; otherwise wrapped)
        seed: Random seed for reproducibility
        tensorboard_log_dir: Directory for TensorBoard logs
        n_steps: Steps per update
        batch_size: Minibatch size
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        learning_rate: Learning rate
        clip_range: PPO clip range
        n_epochs: Number of epochs per update
        **kwargs: Additional PPO parameters
    
    Returns:
        Configured PPO model
    """
    # Set seeds for reproducibility
    set_random_seed(seed)
    np.random.seed(seed)
    
    # If env is already a VecEnv, use it directly; otherwise wrap it
    if isinstance(env, VecEnv):
        vec_env = env
    else:
        # This case should not happen in train_pipeline (we pass vec_env directly)
        # But kept for backward compatibility
        from stable_baselines3.common.vec_env import DummyVecEnv
        vec_env = DummyVecEnv([lambda: env])
    
    # Create PPO model (force CPU for MLP policies to avoid GPU warnings)
    device = kwargs.pop("device", "cpu")
    if device == "auto":
        device = "cpu"  # Force CPU for MLP policies
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        learning_rate=learning_rate,
        clip_range=clip_range,
        n_epochs=n_epochs,
        tensorboard_log=tensorboard_log_dir,
        verbose=1,
        seed=seed,
        device=device,
        **kwargs,
    )
    
    return model
