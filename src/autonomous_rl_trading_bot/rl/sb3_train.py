from __future__ import annotations

"""Compatibility wrapper.

Historically, SB3 training lived under `autonomous_rl_trading_bot.rl.sb3_train`.
For a cleaner architecture, implementation now lives in `autonomous_rl_trading_bot.training`.

Public API is kept identical to avoid breaking run_train.py imports.
"""

from autonomous_rl_trading_bot.training.trainer import TrainConfig, train_and_evaluate

__all__ = ["TrainConfig", "train_and_evaluate"]
