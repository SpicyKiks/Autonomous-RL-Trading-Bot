from __future__ import annotations

import numpy as np


def test_futures_env_short_profits_in_downtrend() -> None:
    from autonomous_rl_trading_bot.envs.futures_env import FuturesEnv, FuturesEnvConfig

    close = np.linspace(100.0, 50.0, num=400).astype(np.float64)
    lr = np.zeros_like(close)
    lr[1:] = np.log((close[1:] + 1e-12) / (close[:-1] + 1e-12))
    feats = lr.reshape(-1, 1)

    cfg = FuturesEnvConfig(
        lookback=32,
        initial_cash=1000.0,
        leverage=5.0,
        maintenance_margin_rate=0.01,
        allow_short=True,
        order_size_quote=0.0,     # all-in
        taker_fee_rate=0.0,
        slippage_bps=0.0,
        max_drawdown=0.999,       # don't stop early
        stop_on_liquidation=True,
    )

    env = FuturesEnv(close=close, features=feats, cfg=cfg, seed=0)
    env.reset()

    # open short all-in
    env.step(FuturesEnv.SHORT)

    # hold for a while
    for _ in range(50):
        _, _, terminated, _, info = env.step(FuturesEnv.HOLD)
        if terminated:
            break

    assert info["equity"] > cfg.initial_cash


def test_futures_env_liquidates_high_leverage_long_on_crash() -> None:
    from autonomous_rl_trading_bot.envs.futures_env import FuturesEnv, FuturesEnvConfig

    close = np.concatenate(
        [
            np.full(80, 100.0),
            np.linspace(100.0, 50.0, num=40),
            np.full(80, 50.0),
        ]
    ).astype(np.float64)

    lr = np.zeros_like(close)
    lr[1:] = np.log((close[1:] + 1e-12) / (close[:-1] + 1e-12))
    feats = lr.reshape(-1, 1)

    cfg = FuturesEnvConfig(
        lookback=16,
        initial_cash=1000.0,
        leverage=10.0,
        maintenance_margin_rate=0.01,
        allow_short=True,
        order_size_quote=0.0,
        taker_fee_rate=0.0,
        slippage_bps=0.0,
        max_drawdown=0.999,  # do not stop before liquidation
        stop_on_liquidation=True,
    )

    env = FuturesEnv(close=close, features=feats, cfg=cfg, seed=1)
    env.reset()

    env.step(FuturesEnv.LONG)  # all-in long at ~100

    liquidated = False
    for _ in range(150):
        _, _, terminated, _, info = env.step(FuturesEnv.HOLD)
        liquidated = liquidated or bool(info.get("liquidated"))
        if terminated:
            break

    assert liquidated is True
    assert info.get("end_reason") in ("liquidation", "")


def test_futures_env_respects_allow_short_false() -> None:
    from autonomous_rl_trading_bot.envs.futures_env import FuturesEnv, FuturesEnvConfig

    close = np.full(200, 100.0).astype(np.float64)
    feats = np.zeros((200, 1), dtype=np.float64)

    cfg = FuturesEnvConfig(
        lookback=16,
        initial_cash=1000.0,
        leverage=3.0,
        maintenance_margin_rate=0.01,
        allow_short=False,
        order_size_quote=0.0,
        taker_fee_rate=0.0,
        slippage_bps=0.0,
        max_drawdown=0.999,
    )

    env = FuturesEnv(close=close, features=feats, cfg=cfg, seed=2)
    env.reset()

    _, _, _, _, info = env.step(FuturesEnv.SHORT)
    assert info["qty"] == 0.0
    assert info.get("executed") is False

