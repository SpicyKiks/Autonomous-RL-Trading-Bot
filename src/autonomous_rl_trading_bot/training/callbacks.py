from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    from stable_baselines3.common.callbacks import (
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.vec_env import VecEnv
except ModuleNotFoundError as e:  # pragma: no cover
    raise RuntimeError(
        "stable-baselines3 is required for training. Install deps: pip install -e ."
    ) from e


@dataclass(frozen=True)
class CallbackConfig:
    checkpoint_freq: int = 10_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 1
    deterministic_eval: bool = True
    verbose: int = 1


def build_callbacks(
    *,
    run_dir: Path,
    eval_env: VecEnv | None,
    cfg: CallbackConfig,
) -> CallbackList:
    """
    SB3 callbacks used during training:
      - periodic checkpoints to run_dir/checkpoints/
      - periodic evaluation (if eval_env provided) saving best_model.zip
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = []

    if int(cfg.checkpoint_freq) > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=int(cfg.checkpoint_freq),
                save_path=str(ckpt_dir),
                name_prefix="ckpt",
                verbose=int(cfg.verbose),
            )
        )

    if eval_env is not None and int(cfg.eval_freq) > 0:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir),
                log_path=str(run_dir / "eval_cb"),
                eval_freq=int(cfg.eval_freq),
                n_eval_episodes=int(cfg.n_eval_episodes),
                deterministic=bool(cfg.deterministic_eval),
                render=False,
                verbose=int(cfg.verbose),
            )
        )

    return CallbackList(callbacks)

