from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from stable_baselines3 import PPO

from autonomous_rl_trading_bot.backtest.metrics import compute_backtest_metrics
from autonomous_rl_trading_bot.envs.trading_env import TradingEnv
from autonomous_rl_trading_bot.training.train_pipeline import split_dataset


def _run_baselines_comparison(
    symbol: str,
    interval: str,
    dataset_path: str,
    train_split: float,
    initial_balance: float,
    taker_fee: float,
    slippage_bps: float,
) -> dict[str, Any]:
    """
    Run baseline strategies and return their metrics.
    
    Returns:
        Dictionary with baseline strategy metrics
    """
    try:
        # Import baseline runner
        from autonomous_rl_trading_bot.evaluation.baselines import BuyAndHold, SMACrossover
        
        # Load dataset
        df = pd.read_parquet(dataset_path)
        _, test_df = split_dataset(df, train_split=train_split)
        
        baselines_metrics = {}
        
        # Buy & Hold baseline
        buyhold_strategy = BuyAndHold(allow_short=False)
        buyhold_metrics = _run_baseline_strategy(
            test_df, buyhold_strategy, initial_balance, taker_fee, slippage_bps
        )
        baselines_metrics["buy_and_hold"] = buyhold_metrics
        
        # SMA Crossover baseline
        sma_strategy = SMACrossover(fast=20, slow=50, allow_short=False)
        sma_metrics = _run_baseline_strategy(
            test_df, sma_strategy, initial_balance, taker_fee, slippage_bps
        )
        baselines_metrics["sma_crossover"] = sma_metrics
        
        return baselines_metrics
    except Exception as e:
        # If baselines fail, return empty dict
        return {"error": str(e)}


def _run_baseline_strategy(
    df: pd.DataFrame,
    strategy,
    initial_balance: float,
    taker_fee: float,
    slippage_bps: float,
) -> dict[str, Any]:
    """Run a baseline strategy and compute metrics."""
    # Generate positions
    positions = strategy.generate_positions(df)

    # Simulate trading
    balance = initial_balance
    position = 0
    qty = 0.0
    entry_price = 0.0
    equity_curve = [initial_balance]
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row["close"])
        target_position = int(positions.iloc[i]) if i < len(positions) else 0

        # Execute position change
        if target_position != position:
            # Close current position
            if position != 0:
                execution_price = (
                    price * (1.0 - slippage_bps / 10000.0)
                    if position == 1
                    else price * (1.0 + slippage_bps / 10000.0)
                )
                notional = execution_price * qty
                fee = notional * taker_fee
                pnl = (
                    (execution_price - entry_price) * qty
                    if position == 1
                    else (entry_price - execution_price) * qty
                )
                pnl -= fee
                balance += notional - fee
                trades.append({"pnl": float(pnl)})
                position = 0
                qty = 0.0

            # Open new position
            if target_position != 0:
                execution_price = (
                    price * (1.0 + slippage_bps / 10000.0)
                    if target_position == 1
                    else price * (1.0 - slippage_bps / 10000.0)
                )
                available = balance * 1.0  # Use full balance
                fee = available * taker_fee
                cost_after_fee = available - fee
                qty = cost_after_fee / execution_price
                balance -= available
                entry_price = execution_price
                position = target_position

        # Update equity
        if position == 1:
            equity = balance + (price - entry_price) * qty
        elif position == -1:
            equity = balance + (entry_price - price) * qty
        else:
            equity = balance
        equity_curve.append(float(equity))

    # Compute metrics
    return compute_backtest_metrics(equity_curve, trades, initial_balance)


def load_run_config(run_id: str) -> dict[str, Any]:
    """
    Load run configuration from run directory.
    
    Looks for run_config.json in:
    1. logs/runs/<run_id>/run_config.json
    2. logs/tensorboard/<run_id>/../run_config.json
    3. models/<model_name>_config.json (if run_id matches model name pattern)
    
    Args:
        run_id: Run identifier
    
    Returns:
        Dictionary of run configuration
    """
    # Try logs/runs/<run_id>/run_config.json
    run_dir = Path("logs") / "runs" / run_id
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    
    # Try logs/tensorboard/<run_id>/../run_config.json
    tb_dir = Path("logs") / "tensorboard" / run_id
    config_path = tb_dir.parent / "run_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    
    # Try models directory (for backward compatibility)
    models_dir = Path("models")
    if models_dir.exists():
        for config_file in models_dir.glob("*_config.json"):
            with open(config_file) as f:
                config = json.load(f)
                # Check if this config matches the run_id pattern
                if run_id in config_file.stem or config_file.stem.startswith("ppo_trader"):
                    return config
    
    raise FileNotFoundError(f"Run config not found for run_id: {run_id}")


def run_backtest(
    *,
    mode: str,
    policy: str = "ppo",
    model_path: str | None = None,
    dataset_path: str | None = None,
    symbol: str | None = None,
    interval: str | None = None,
    run_id: str | None = None,
    train_split: float = 0.8,
    output_dir: str = "reports",
) -> dict[str, Any]:
    """
    Run backtest evaluation on test split.
    
    Args:
        mode: Market mode ("spot" or "futures")
        policy: Policy to backtest ("ppo", "buyhold", "sma", or "rsi")
        model_path: Path to saved PPO model (.zip file) - required if policy="ppo"
        dataset_path: Optional explicit dataset path (parquet)
        symbol: Symbol (e.g., "BTCUSDT") - required if dataset_path not provided
        interval: Interval (e.g., "1m") - required if dataset_path not provided
        run_id: Optional run ID to load config from
        train_split: Train/test split ratio (default 0.8)
        output_dir: Output directory for reports
    
    Returns:
        Dictionary with metrics and output paths
    """
    # Load run config if run_id provided
    config: dict[str, Any] = {}
    if run_id:
        try:
            config = load_run_config(run_id)
            # Extract params from config
            symbol = config.get("symbol", symbol)
            interval = config.get("interval", interval)
            train_split = config.get("train_split", train_split)
            dataset_path = config.get("dataset_path", dataset_path)
            # Use model path from config if available
            if not model_path:
                model_path = config.get("model_path", "models/ppo_trader.zip")
        except FileNotFoundError:
            # If run config not found, continue with provided params
            pass
    
    # Determine dataset path
    if not dataset_path:
        if not symbol or not interval:
            raise ValueError("Either dataset_path or both symbol and interval must be provided")
        dataset_path = f"data/processed/{symbol.upper()}_{interval}_dataset.parquet"
    
    dataset_path_obj = Path(dataset_path)
    if not dataset_path_obj.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Load dataset
    df = pd.read_parquet(dataset_path_obj)
    
    # Split dataset (use test split only for backtest)
    _, test_df = split_dataset(df, train_split=train_split)
    
    # Load environment config from run config or use defaults
    initial_balance = config.get("initial_balance", 10000.0)
    taker_fee = config.get("taker_fee", 0.0004)
    slippage_bps = config.get("slippage_bps", 1.0)
    risk_penalty = config.get("risk_penalty", 0.1)
    position_fraction = config.get("position_fraction", 1.0)
    seed = config.get("seed", 42)
    
    # Route to appropriate backtest implementation
    if policy == "ppo":
        # Default model path if not provided
        if not model_path:
            model_path = "models/ppo_trader.zip"
        
        # Verify model path exists
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        return _run_ppo_backtest(
            test_df=test_df,
            model_path=model_path,
            initial_balance=initial_balance,
            taker_fee=taker_fee,
            slippage_bps=slippage_bps,
            risk_penalty=risk_penalty,
            position_fraction=position_fraction,
            seed=seed,
            train_split=train_split,
            output_dir=output_dir,
            symbol=symbol,
            interval=interval,
        )
    else:
        # Baseline strategy backtest
        return _run_baseline_backtest(
            test_df=test_df,
            policy=policy,
            initial_balance=initial_balance,
            taker_fee=taker_fee,
            slippage_bps=slippage_bps,
            risk_penalty=risk_penalty,
            position_fraction=position_fraction,
            seed=seed,
            train_split=train_split,
            output_dir=output_dir,
            symbol=symbol,
            interval=interval,
        )


def _run_ppo_backtest(
    *,
    test_df: pd.DataFrame,
    model_path: str,
    initial_balance: float,
    taker_fee: float,
    slippage_bps: float,
    risk_penalty: float,
    position_fraction: float,
    seed: int,
    train_split: float,
    output_dir: str,
    symbol: str | None,
    interval: str | None,
) -> dict[str, Any]:
    """Run PPO model backtest (existing implementation)."""
    # Create test environment (use same constraints as training)
    test_env = TradingEnv(
        test_df,
        initial_balance=initial_balance,
        taker_fee=taker_fee,
        slippage_bps=slippage_bps,
        risk_penalty=risk_penalty,
        position_fraction=position_fraction,
        cooldown_steps=10,  # Match training constraints
        hold_threshold=0.1,
        cost_penalty=0.01,
        position_change_penalty=0.005,
        seed=seed,
    )
    
    # Load model
    model = PPO.load(model_path, device="cpu")
    
    # Run backtest
    obs, info = test_env.reset()
    done = False
    
    # Logging structures
    equity_curve: list[float] = [initial_balance]
    step_logs: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    
    last_position = 0
    last_trade_count = 0
    current_trade: dict[str, Any] | None = None
    
    while not done:
        # Get current state (before step, so we log the state before action)
        current_row = test_env.df.iloc[test_env.current_idx]
        current_price = float(current_row["close"])
        
        # Predict action
        action, _ = model.predict(obs, deterministic=True)
        action_int = int(action)
        
        # Get action probabilities for hold threshold (if available)
        action_probs = None
        try:
            # Try to get probabilities from policy
            obs_tensor = model.policy.obs_to_tensor(obs)[0]
            distribution = model.policy.get_distribution(obs_tensor)
            if hasattr(distribution, 'distribution'):
                probs = distribution.distribution.probs
                if probs is not None:
                    action_probs = probs.detach().cpu().numpy().flatten()
        except Exception:
            # If we can't get probabilities, continue without hold threshold
            pass
        
        # Execute step (pass action_probs for hold threshold)
        obs, reward, terminated, truncated, info = test_env.step(action_int, action_probs=action_probs)
        done = terminated or truncated
        
        # Log step
        step_logs.append({
            "timestamp_ms": int(info["timestamp_ms"]),
            "price": float(current_price),
            "action": action_int,
            "position": int(info["position"]),
            "equity": float(info["equity"]),
            "reward": float(reward),
            "balance": float(info["balance"]),
            "qty": float(info["qty"]),
            "entry_price": float(info["entry_price"]) if info["entry_price"] > 0 else 0.0,
        })
        
        # Track trades (position changes)
        if info["position"] != last_position:
            # Position changed
            if last_position != 0:
                # Close previous trade
                if current_trade is not None:
                    # Calculate gross PnL
                    gross_pnl = 0.0
                    if last_position == 1:  # Long
                        gross_pnl = (current_price - current_trade["entry_price"]) * current_trade["qty"]
                    elif last_position == -1:  # Short
                        gross_pnl = (current_trade["entry_price"] - current_price) * current_trade["qty"]
                    
                    # Calculate fees and slippage
                    notional_entry = current_trade["entry_price"] * current_trade["qty"]
                    notional_exit = current_price * current_trade["qty"]
                    
                    # Entry fee
                    entry_fee = notional_entry * taker_fee
                    # Exit fee
                    exit_fee = notional_exit * taker_fee
                    total_fee = entry_fee + exit_fee
                    
                    # Slippage cost (difference between mid price and execution price)
                    # For long exit: sell at lower price (negative slippage)
                    # For short exit: buy at higher price (negative slippage)
                    if last_position == 1:  # Long exit: sell at price * (1 - slippage_bps/10000)
                        execution_price = current_price * (1.0 - slippage_bps / 10000.0)
                        mid_price = current_price
                        slippage_cost = abs(mid_price - execution_price) * current_trade["qty"]
                    else:  # Short exit: buy at price * (1 + slippage_bps/10000)
                        execution_price = current_price * (1.0 + slippage_bps / 10000.0)
                        mid_price = current_price
                        slippage_cost = abs(execution_price - mid_price) * current_trade["qty"]
                    
                    # Net PnL
                    net_pnl = gross_pnl - total_fee - slippage_cost
                    
                    current_trade["exit_price"] = float(execution_price)  # Use execution price (with slippage)
                    current_trade["exit_timestamp_ms"] = int(current_row["timestamp_ms"])
                    current_trade["gross_pnl"] = float(gross_pnl)
                    current_trade["fee_paid"] = float(total_fee)
                    current_trade["slippage_cost"] = float(slippage_cost)
                    current_trade["net_pnl"] = float(net_pnl)
                    current_trade["pnl"] = float(net_pnl)  # Legacy alias
                    trades.append(current_trade)
                    current_trade = None
            
            # Open new trade if position is not flat
            if info["position"] != 0:
                current_trade = {
                    "entry_timestamp_ms": int(current_row["timestamp_ms"]),
                    "entry_price": float(current_price),
                    "position": int(info["position"]),
                    "qty": float(info["qty"]),
                }
        
        last_position = info["position"]
        equity_curve.append(float(info["equity"]))
    
    # Close any open trade at end
    if current_trade is not None:
        final_row = test_env.df.iloc[min(test_env.current_idx, len(test_env.df) - 1)]
        final_price = float(final_row["close"])
        
        # Get execution price (with slippage)
        if current_trade["position"] == 1:  # Long exit
            execution_price = final_price * (1.0 - slippage_bps / 10000.0)
            mid_price = final_price
        else:  # Short exit
            execution_price = final_price * (1.0 + slippage_bps / 10000.0)
            mid_price = final_price
        
        # Calculate gross PnL (using mid prices)
        entry_mid_price = current_trade["entry_price"] / (1.0 + slippage_bps / 10000.0) if current_trade["position"] == 1 else current_trade["entry_price"] / (1.0 - slippage_bps / 10000.0)
        gross_pnl = 0.0
        if current_trade["position"] == 1:  # Long
            gross_pnl = (mid_price - entry_mid_price) * current_trade["qty"]
        elif current_trade["position"] == -1:  # Short
            gross_pnl = (entry_mid_price - mid_price) * current_trade["qty"]
        
        # Calculate fees (on execution prices)
        notional_entry = current_trade["entry_price"] * current_trade["qty"]
        notional_exit = execution_price * current_trade["qty"]
        
        entry_fee = notional_entry * taker_fee
        exit_fee = notional_exit * taker_fee
        total_fee = entry_fee + exit_fee
        
        # Slippage cost
        entry_slippage = abs(current_trade["entry_price"] - entry_mid_price) * current_trade["qty"]
        exit_slippage = abs(execution_price - mid_price) * current_trade["qty"]
        slippage_cost = entry_slippage + exit_slippage
        
        # Net PnL
        net_pnl = gross_pnl - total_fee - slippage_cost
        
        current_trade["exit_price"] = float(execution_price)
        current_trade["exit_timestamp_ms"] = int(final_row["timestamp_ms"])
        current_trade["gross_pnl"] = float(gross_pnl)
        current_trade["fee_paid"] = float(total_fee)
        current_trade["slippage_cost"] = float(slippage_cost)
        current_trade["net_pnl"] = float(net_pnl)
        current_trade["pnl"] = float(net_pnl)  # Legacy alias
        trades.append(current_trade)
    
    # Compute metrics
    metrics = compute_backtest_metrics(equity_curve, trades, initial_balance)
    metrics["policy"] = "ppo"
    
    # Run baselines comparison (only for PPO backtests)
    baselines_metrics = {}
    try:
        from autonomous_rl_trading_bot.evaluation.baselines import BuyAndHold, SMACrossover
        
        # Buy & Hold baseline
        buyhold_strategy = BuyAndHold(allow_short=False)
        buyhold_metrics = _run_baseline_strategy(
            test_df, buyhold_strategy, initial_balance, taker_fee, slippage_bps
        )
        baselines_metrics["buy_and_hold"] = buyhold_metrics
        
        # SMA Crossover baseline
        sma_strategy = SMACrossover(fast=20, slow=50, allow_short=False)
        sma_metrics = _run_baseline_strategy(
            test_df, sma_strategy, initial_balance, taker_fee, slippage_bps
        )
        baselines_metrics["sma_crossover"] = sma_metrics
    except Exception as e:
        # If baselines fail, log error but continue
        baselines_metrics = {"error": str(e)}
    
    # Add baselines to metrics
    metrics["baselines"] = baselines_metrics
    
    return _save_backtest_results(metrics, trades, step_logs, equity_curve, output_dir)


def _run_baseline_backtest(
    *,
    test_df: pd.DataFrame,
    policy: str,
    initial_balance: float,
    taker_fee: float,
    slippage_bps: float,
    risk_penalty: float,
    position_fraction: float,
    seed: int,
    train_split: float,
    output_dir: str,
    symbol: str | None,
    interval: str | None,
) -> dict[str, Any]:
    """Run baseline strategy backtest."""
    from autonomous_rl_trading_bot.evaluation.baselines import (
        BuyAndHold,
        RSIReversion,
        SMACrossover,
    )
    
    # Create baseline strategy
    if policy == "buyhold":
        strategy = BuyAndHold(allow_short=False)
    elif policy == "sma":
        strategy = SMACrossover(fast=20, slow=50, allow_short=False)
    elif policy == "rsi":
        strategy = RSIReversion(allow_short=False)
    else:
        raise ValueError(f"Unknown baseline policy: {policy}")
    
    # Generate position signals
    positions = strategy.generate_positions(test_df)
    
    # Simulate trading with same execution rules as PPO backtest
    balance = initial_balance
    position = 0
    qty = 0.0
    entry_price = 0.0
    equity_curve: list[float] = [initial_balance]
    step_logs: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    
    current_trade: dict[str, Any] | None = None
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        price = float(row["close"])
        timestamp_ms = int(row["timestamp_ms"])
        target_position = int(positions.iloc[i]) if i < len(positions) else 0
        
        # Execute position changes (same logic as TradingEnv)
        if target_position != position:
            # Close current position
            if position != 0:
                if current_trade is not None:
                    # Calculate execution price with slippage
                    if position == 1:  # Long exit
                        execution_price = price * (1.0 - slippage_bps / 10000.0)
                        mid_price = price
                    else:  # Short exit
                        execution_price = price * (1.0 + slippage_bps / 10000.0)
                        mid_price = price
                    
                    # Calculate gross PnL (using mid prices)
                    entry_mid_price = entry_price / (1.0 + slippage_bps / 10000.0) if position == 1 else entry_price / (1.0 - slippage_bps / 10000.0)
                    gross_pnl = 0.0
                    if position == 1:  # Long
                        gross_pnl = (mid_price - entry_mid_price) * qty
                    else:  # Short
                        gross_pnl = (entry_mid_price - mid_price) * qty
                    
                    # Calculate fees
                    notional_entry = entry_price * qty
                    notional_exit = execution_price * qty
                    entry_fee = notional_entry * taker_fee
                    exit_fee = notional_exit * taker_fee
                    total_fee = entry_fee + exit_fee
                    
                    # Slippage cost
                    entry_slippage = abs(entry_price - entry_mid_price) * qty
                    exit_slippage = abs(execution_price - mid_price) * qty
                    slippage_cost = entry_slippage + exit_slippage
                    
                    # Net PnL
                    net_pnl = gross_pnl - total_fee - slippage_cost
                    
                    # Update balance
                    if position == 1:  # Long exit
                        balance += notional_exit - exit_fee
                    else:  # Short exit
                        balance -= (execution_price * qty + exit_fee)
                    
                    current_trade["exit_price"] = float(execution_price)
                    current_trade["exit_timestamp_ms"] = timestamp_ms
                    current_trade["gross_pnl"] = float(gross_pnl)
                    current_trade["fee_paid"] = float(total_fee)
                    current_trade["slippage_cost"] = float(slippage_cost)
                    current_trade["net_pnl"] = float(net_pnl)
                    current_trade["pnl"] = float(net_pnl)
                    trades.append(current_trade)
                    current_trade = None
                
                position = 0
                qty = 0.0
                entry_price = 0.0
            
            # Open new position
            if target_position != 0:
                # Calculate execution price with slippage
                if target_position == 1:  # Long entry
                    execution_price = price * (1.0 + slippage_bps / 10000.0)
                else:  # Short entry
                    execution_price = price * (1.0 - slippage_bps / 10000.0)
                
                available = balance * position_fraction
                fee = available * taker_fee
                cost_after_fee = available - fee
                
                if cost_after_fee > 0:
                    qty = cost_after_fee / execution_price
                    balance -= available
                    entry_price = execution_price
                    position = target_position
                    
                    current_trade = {
                        "entry_timestamp_ms": timestamp_ms,
                        "entry_price": float(execution_price),
                        "position": int(target_position),
                        "qty": float(qty),
                    }
        
        # Update equity (mark-to-market)
        if position == 1:  # Long
            equity = balance + (price - entry_price) * qty
        elif position == -1:  # Short
            equity = balance + (entry_price - price) * qty
        else:
            equity = balance
        
        equity_curve.append(float(equity))
        step_logs.append({
            "timestamp_ms": timestamp_ms,
            "price": float(price),
            "action": target_position,  # Position signal
            "position": int(position),
            "equity": float(equity),
            "reward": 0.0,  # Not applicable for baselines
            "balance": float(balance),
            "qty": float(qty),
            "entry_price": float(entry_price) if entry_price > 0 else 0.0,
        })
    
    # Close any open trade at end
    if current_trade is not None and position != 0:
        final_price = float(test_df.iloc[-1]["close"])
        final_timestamp_ms = int(test_df.iloc[-1]["timestamp_ms"])
        
        if position == 1:  # Long exit
            execution_price = final_price * (1.0 - slippage_bps / 10000.0)
            mid_price = final_price
        else:  # Short exit
            execution_price = final_price * (1.0 + slippage_bps / 10000.0)
            mid_price = final_price
        
        entry_mid_price = entry_price / (1.0 + slippage_bps / 10000.0) if position == 1 else entry_price / (1.0 - slippage_bps / 10000.0)
        gross_pnl = 0.0
        if position == 1:
            gross_pnl = (mid_price - entry_mid_price) * qty
        else:
            gross_pnl = (entry_mid_price - mid_price) * qty
        
        notional_entry = entry_price * qty
        notional_exit = execution_price * qty
        entry_fee = notional_entry * taker_fee
        exit_fee = notional_exit * taker_fee
        total_fee = entry_fee + exit_fee
        
        entry_slippage = abs(entry_price - entry_mid_price) * qty
        exit_slippage = abs(execution_price - mid_price) * qty
        slippage_cost = entry_slippage + exit_slippage
        
        net_pnl = gross_pnl - total_fee - slippage_cost
        
        current_trade["exit_price"] = float(execution_price)
        current_trade["exit_timestamp_ms"] = final_timestamp_ms
        current_trade["gross_pnl"] = float(gross_pnl)
        current_trade["fee_paid"] = float(total_fee)
        current_trade["slippage_cost"] = float(slippage_cost)
        current_trade["net_pnl"] = float(net_pnl)
        current_trade["pnl"] = float(net_pnl)
        trades.append(current_trade)
    
    # Compute metrics
    metrics = compute_backtest_metrics(equity_curve, trades, initial_balance)
    metrics["policy"] = policy
    
    return _save_backtest_results(metrics, trades, step_logs, equity_curve, output_dir)


def _save_backtest_results(
    metrics: dict[str, Any],
    trades: list[dict[str, Any]],
    step_logs: list[dict[str, Any]],
    equity_curve: list[float],
    output_dir: str,
) -> dict[str, Any]:
    """Save backtest results to files."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save reports
    report_json_path = output_path / "backtest_report.json"
    with open(report_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Ensure all trades have required columns
    if trades:
        normalized_trades = []
        for trade in trades:
            normalized = {
                "entry_timestamp_ms": trade.get("entry_timestamp_ms", trade.get("entry_timestamp", 0)),
                "exit_timestamp_ms": trade.get("exit_timestamp_ms", trade.get("exit_timestamp", 0)),
                "entry_price": float(trade.get("entry_price", 0.0)),
                "exit_price": float(trade.get("exit_price", 0.0)),
                "position": int(trade.get("position", 0)),
                "qty": float(trade.get("qty", 0.0)),
                "gross_pnl": float(trade.get("gross_pnl", trade.get("pnl", 0.0))),
                "fee_paid": float(trade.get("fee_paid", 0.0)),
                "slippage_cost": float(trade.get("slippage_cost", 0.0)),
                "net_pnl": float(trade.get("net_pnl", trade.get("pnl", 0.0))),
                "pnl": float(trade.get("net_pnl", trade.get("pnl", 0.0))),
            }
            normalized_trades.append(normalized)
        trades_df = pd.DataFrame(normalized_trades)
    else:
        trades_df = pd.DataFrame(columns=[
            "entry_timestamp_ms", "exit_timestamp_ms", "entry_price", "exit_price",
            "position", "qty", "gross_pnl", "fee_paid", "slippage_cost", "net_pnl", "pnl"
        ])
    
    trades_csv_path = output_path / "trades.csv"
    trades_df.to_csv(trades_csv_path, index=False)
    
    # Calculate drawdown for each step
    initial_balance_val = equity_curve[0] if equity_curve else 10000.0
    peak_equity = initial_balance_val
    drawdown_curve = []
    for equity_val in equity_curve[1:]:  # Skip initial balance
        peak_equity = max(peak_equity, equity_val)
        drawdown = (peak_equity - equity_val) / peak_equity if peak_equity > 0 else 0.0
        drawdown_curve.append(float(drawdown))
    
    equity_df = pd.DataFrame({
        "timestamp_ms": [log["timestamp_ms"] for log in step_logs],
        "equity": equity_curve[1:],  # Skip initial balance
        "drawdown": drawdown_curve,
    })
    equity_csv_path = output_path / "equity_curve.csv"
    equity_df.to_csv(equity_csv_path, index=False)
    
    return {
        "metrics": metrics,
        "report_json": str(report_json_path),
        "trades_csv": str(trades_csv_path),
        "equity_csv": str(equity_csv_path),
        "num_steps": len(step_logs),
        "num_trades": len(trades),
    }
