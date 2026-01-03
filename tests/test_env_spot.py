from __future__ import annotations

import numpy as np


def test_spot_env_reset_step_and_equity_up_in_uptrend() -> None:
    from autonomous_rl_trading_bot.envs.spot_env import SpotEnv, SpotEnvConfig

    n = 256
    close = np.linspace(100.0, 200.0, num=n).astype(np.float64)

    lr = np.zeros_like(close)
    lr[1:] = np.log((close[1:] + 1e-12) / (close[:-1] + 1e-12))
    feats = lr.reshape(-1, 1)

    cfg = SpotEnvConfig(lookback=32, initial_cash=1000.0, order_size_quote=0.0, taker_fee_rate=0.0, slippage_bps=0.0)
    env = SpotEnv(close=close, features=feats, cfg=cfg, seed=123)

    try:
        from stable_baselines3.common.env_checker import check_env
        check_env(env, warn=True)
    except Exception:
        pass

    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert isinstance(info["equity"], float)

    obs, r, terminated, truncated, info = env.step(SpotEnv.BUY)
    assert truncated is False
    assert not np.isnan(r)

    eq_after_buy = info["equity"]

    for _ in range(25):
        obs, r, terminated, truncated, info = env.step(SpotEnv.HOLD)
        if terminated:
            break

    assert info["equity"] >= eq_after_buy


def test_spot_env_close_sells_inventory() -> None:
    from autonomous_rl_trading_bot.envs.spot_env import SpotEnv, SpotEnvConfig

    close = np.linspace(100.0, 110.0, num=128).astype(np.float64)
    feats = np.zeros((128, 1), dtype=np.float64)
    cfg = SpotEnvConfig(lookback=16, initial_cash=1000.0, order_size_quote=0.0, taker_fee_rate=0.0, slippage_bps=0.0)
    env = SpotEnv(close=close, features=feats, cfg=cfg)
    env.reset()

    env.step(SpotEnv.BUY)
    _, _, _, _, info = env.step(SpotEnv.CLOSE)
    assert info["qty_base"] == 0.0

