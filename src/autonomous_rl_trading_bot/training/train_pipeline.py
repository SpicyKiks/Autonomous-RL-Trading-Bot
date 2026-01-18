from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from autonomous_rl_trading_bot.agents.ppo_agent import create_ppo
from autonomous_rl_trading_bot.envs.trading_env import TradingEnv


def load_dataset(
    symbol: str,
    interval: str,
    processed_dir: str = "data/processed",
) -> pd.DataFrame:
    """
    Load Day-1 dataset parquet.
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        interval: Timeframe (e.g., "1m")
        processed_dir: Directory containing processed datasets
    
    Returns:
        DataFrame with columns: timestamp_ms, datetime_utc, close, next_log_return, state
    """
    dataset_path = Path(processed_dir) / f"{symbol.upper()}_{interval}_dataset.parquet"
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Run: python scripts/make_dataset.py --symbol {symbol} --interval {interval}"
        )
    
    df = pd.read_parquet(dataset_path)
    
    # Validate required columns
    required = ["timestamp_ms", "close", "next_log_return", "state"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    
    return df


def split_dataset(
    df: pd.DataFrame,
    train_split: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and test sets chronologically.
    
    Args:
        df: Full dataset DataFrame
        train_split: Fraction for training (0.0 to 1.0)
    
    Returns:
        Tuple of (train_df, test_df)
    """
    if not (0.0 < train_split < 1.0):
        raise ValueError(f"train_split must be between 0 and 1, got {train_split}")
    
    # Ensure sorted by timestamp
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    
    split_idx = int(len(df) * train_split)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df


def train_ppo(
    train_df: pd.DataFrame,
    *,
    timesteps: int = 500000,
    seed: int = 42,
    tensorboard_log_dir: Optional[str] = None,
    model_out: str = "models/ppo_trader.zip",
    taker_fee: float = 0.0004,
    slippage_bps: float = 1.0,
    risk_penalty: float = 0.1,
    position_fraction: float = 1.0,
    initial_balance: float = 10000.0,
    **ppo_kwargs: Any,
) -> str:
    """
    Train PPO agent on training dataset.
    
    Args:
        train_df: Training dataset DataFrame
        timesteps: Total training timesteps
        seed: Random seed
        tensorboard_log_dir: Directory for TensorBoard logs
        model_out: Output path for saved model
        taker_fee: Taker fee rate (default 0.0004 = 0.04%)
        slippage_bps: Slippage in basis points (default 1.0 bps = 0.01%)
        risk_penalty: Risk penalty coefficient
        position_fraction: Fraction of balance to use for positions
        initial_balance: Initial account balance
        **ppo_kwargs: Additional PPO hyperparameters
    
    Returns:
        Path to saved model
    """
    # Create make_env function for vectorized environment
    # NO Monitor here - VecMonitor will handle statistics
    def make_env():
            return TradingEnv(
                train_df,
                initial_balance=initial_balance,
                taker_fee=taker_fee,
                slippage_bps=slippage_bps,
                risk_penalty=risk_penalty,
                position_fraction=position_fraction,
                cooldown_steps=10,  # Block trades for 10 steps after each trade
                hold_threshold=0.1,  # Force HOLD if action confidence < 10%
                cost_penalty=0.01,  # Penalty for fees
                position_change_penalty=0.005,  # Penalty for position changes
                seed=seed,
            )
    
    # Create vectorized environment with VecMonitor
    vec_env = DummyVecEnv([make_env])
    # VecMonitor aggregates stats from all envs in the vector and logs to TensorBoard
    monitor_dir = Path(tensorboard_log_dir) / "monitor" if tensorboard_log_dir else None
    if monitor_dir:
        monitor_dir.mkdir(parents=True, exist_ok=True)
    vec_env = VecMonitor(vec_env, filename=str(monitor_dir / "train.monitor.csv") if monitor_dir else None)
    
    # Create PPO model (pass the wrapped vec_env directly)
    model = create_ppo(
        vec_env,  # Pass vec_env directly instead of raw env
        seed=seed,
        tensorboard_log_dir=tensorboard_log_dir,
        **ppo_kwargs,
    )
    
    # Train
    model.learn(total_timesteps=timesteps)
    
    # Save model
    model_path = Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    
    # Save training config (both next to model and in run directory)
    config_path = model_path.parent / f"{model_path.stem}_config.json"
    config = {
        "timesteps": timesteps,
        "seed": seed,
        "taker_fee": taker_fee,
        "slippage_bps": slippage_bps,
        "risk_penalty": risk_penalty,
        "position_fraction": position_fraction,
        "initial_balance": initial_balance,
        "train_rows": len(train_df),
        **ppo_kwargs,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # Also save run_config.json in tensorboard log directory if provided
    if tensorboard_log_dir:
        run_dir = Path(tensorboard_log_dir).parent
        run_config_path = run_dir / "run_config.json"
        run_config_path.parent.mkdir(parents=True, exist_ok=True)
        # Extract symbol/interval from dataset if available (will be added by caller)
        run_config = config.copy()
        with open(run_config_path, "w") as f:
            json.dump(run_config, f, indent=2)
    
    return str(model_path)


def evaluate_ppo(
    model_path: str,
    test_df: pd.DataFrame,
    *,
    seed: int = 42,
    taker_fee: float = 0.0004,
    slippage_bps: float = 1.0,
    risk_penalty: float = 0.1,
    position_fraction: float = 1.0,
    initial_balance: float = 10000.0,
    report_out: str = "reports/training_metrics.json",
) -> Dict[str, Any]:
    """
    Evaluate trained PPO model on test dataset.
    
    Args:
        model_path: Path to saved PPO model
        test_df: Test dataset DataFrame
        seed: Random seed
        taker_fee: Taker fee rate
        slippage_bps: Slippage in basis points
        risk_penalty: Risk penalty coefficient
        position_fraction: Fraction of balance to use for positions
        initial_balance: Initial account balance
        report_out: Output path for metrics JSON
    
    Returns:
        Dictionary of evaluation metrics
    """
    from stable_baselines3 import PPO
    
    # Load model (force CPU to avoid GPU warnings)
    model = PPO.load(model_path, device="cpu")
    
    # Create test environment with Monitor
    test_env_raw = TradingEnv(
        test_df,
        initial_balance=initial_balance,
        taker_fee=taker_fee,
        slippage_bps=slippage_bps,
        risk_penalty=risk_penalty,
        position_fraction=position_fraction,
        seed=seed,
    )
    
    # Wrap with Monitor for evaluation statistics
    monitor_dir = Path(report_out).parent / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    test_env = Monitor(test_env_raw, filename=str(monitor_dir / "eval.monitor.csv"))
    
    # Run evaluation episode
    obs, info = test_env.reset()
    done = False
    
    equity_history: list[float] = [initial_balance]
    rewards: list[float] = []
    trades: list[Dict[str, Any]] = []
    last_trade_count = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(int(action))
        done = terminated or truncated
        
        equity_history.append(float(info["equity"]))
        rewards.append(float(reward))
        
        # Track trades
        current_trade_count = info.get("trade_count", 0)
        if current_trade_count > last_trade_count:
            trades.append({
                "price": float(info["price"]),
                "position": int(info["position"]),
                "equity": float(info["equity"]),
            })
            last_trade_count = current_trade_count
    
    # Compute metrics
    equity_array = np.array(equity_history)
    returns = np.diff(np.log(np.maximum(equity_array, 1e-12)))
    
    total_return = (equity_array[-1] / equity_array[0]) - 1.0
    
    sharpe = 0.0
    if len(returns) > 1 and np.std(returns) > 1e-12:
        sharpe = np.mean(returns) / np.std(returns) * math.sqrt(252 * 24 * 60)  # Annualized for 1m data
    
    peak = np.maximum.accumulate(equity_array)
    drawdown = (peak - equity_array) / peak
    max_drawdown = float(np.max(drawdown))
    
    # Access unwrapped environment for TradingEnv attributes
    unwrapped_env = test_env.unwrapped
    
    win_rate = 0.0
    if unwrapped_env.win_count + unwrapped_env.loss_count > 0:
        win_rate = unwrapped_env.win_count / (unwrapped_env.win_count + unwrapped_env.loss_count)
    
    avg_trade_pnl = 0.0
    if unwrapped_env.trade_count > 0:
        avg_trade_pnl = unwrapped_env.realized_pnl / unwrapped_env.trade_count
    
    metrics = {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "trades": int(unwrapped_env.trade_count),
        "win_rate": float(win_rate),
        "avg_trade_pnl": float(avg_trade_pnl),
        "final_equity": float(equity_array[-1]),
        "initial_equity": float(equity_array[0]),
        "realized_pnl": float(unwrapped_env.realized_pnl),
        "win_count": int(unwrapped_env.win_count),
        "loss_count": int(unwrapped_env.loss_count),
    }
    
    # Save metrics
    report_path = Path(report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics
