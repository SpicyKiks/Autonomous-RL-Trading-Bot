from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Import lazily to avoid circular imports
# from autonomous_rl_trading_bot.backtest.runner import run_backtest
# from autonomous_rl_trading_bot.training.train_pipeline import (
#     load_dataset,
#     split_dataset,
#     train_ppo,
#     evaluate_ppo,
# )


def _utc_now_compact() -> str:
    """Generate compact UTC timestamp for run IDs."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _download_data(
    symbol: str,
    interval: str,
    days: int,
    mode: str,
) -> Path:
    """Download OHLCV data from Binance."""
    script_path = Path("scripts/download_binance_futures.py")
    
    if not script_path.exists():
        raise FileNotFoundError(f"Download script not found: {script_path}")
    
    # Determine exchange based on mode
    exchange_name = "binanceusdm" if mode == "futures" else "binance"
    
    # Run download script
    cmd = [
        sys.executable,
        str(script_path),
        "--symbol", symbol,
        "--interval", interval,
        "--days", str(days),
        "--exchange", exchange_name,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Data download failed: {result.stderr}")
    
    # Verify output file exists
    raw_dir = Path("data/raw")
    stem = f"{symbol.upper()}_{interval}"
    raw_file = raw_dir / f"{stem}.parquet"
    
    if not raw_file.exists():
        raw_file = raw_dir / f"{stem}.csv"
    
    if not raw_file.exists():
        raise FileNotFoundError(f"Downloaded data file not found: {raw_file}")
    
    return raw_file


def _build_features(symbol: str, interval: str) -> Path:
    """Build features from raw data."""
    script_path = Path("scripts/build_features.py")
    
    if not script_path.exists():
        raise FileNotFoundError(f"Build features script not found: {script_path}")
    
    cmd = [
        sys.executable,
        str(script_path),
        "--symbol", symbol,
        "--interval", interval,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Feature building failed: {result.stderr}")
    
    # Verify output
    out_dir = Path("data/processed")
    stem = f"{symbol.upper()}_{interval}"
    features_file = out_dir / f"{stem}_features.parquet"
    
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    
    return features_file


def _make_dataset(symbol: str, interval: str, window: int) -> Path:
    """Make final dataset with windowed states."""
    script_path = Path("scripts/make_dataset.py")
    
    if not script_path.exists():
        raise FileNotFoundError(f"Make dataset script not found: {script_path}")
    
    cmd = [
        sys.executable,
        str(script_path),
        "--symbol", symbol,
        "--interval", interval,
        "--window", str(window),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Dataset creation failed: {result.stderr}")
    
    # Verify output
    out_dir = Path("data/processed")
    stem = f"{symbol.upper()}_{interval}"
    dataset_file = out_dir / f"{stem}_dataset.parquet"
    
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    
    return dataset_file


def run_repro(
    *,
    symbol: str,
    interval: str,
    days: int,
    window: int,
    timesteps: int,
    mode: str,
    seed: int = 42,
    output_dir: str = "reports/repro",
) -> Dict[str, Any]:
    """
    Run complete reproducibility pipeline: download -> features -> dataset -> train -> backtest.
    
    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        interval: Timeframe interval (e.g., 1m)
        days: Number of days of data to download
        window: State window size
        timesteps: Training timesteps
        mode: Market mode (spot or futures)
        seed: Random seed
        output_dir: Base output directory
    
    Returns:
        Dictionary with run_id and output paths
    """
    run_id = f"{_utc_now_compact()}_{mode}_{symbol}_{interval}_w{window}_t{timesteps}_s{seed}"
    repro_dir = Path(output_dir) / run_id
    repro_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[REPRO] Starting reproducibility run: {run_id}")
    print(f"[REPRO] Output directory: {repro_dir}")
    
    # Step 1: Download data
    print("\n[1/6] Downloading data...")
    raw_file = _download_data(symbol, interval, days, mode)
    print(f"  ✓ Downloaded: {raw_file}")
    
    # Step 2: Build features
    print("\n[2/6] Building features...")
    features_file = _build_features(symbol, interval)
    print(f"  ✓ Features built: {features_file}")
    
    # Step 3: Make dataset
    print("\n[3/6] Making dataset...")
    dataset_file = _make_dataset(symbol, interval, window)
    print(f"  ✓ Dataset created: {dataset_file}")
    
    # Step 4: Train PPO
    print("\n[4/6] Training PPO...")
    # Lazy import to avoid circular dependencies
    from autonomous_rl_trading_bot.training.train_pipeline import (
        load_dataset,
        split_dataset,
        train_ppo,
        evaluate_ppo,
    )
    
    df = load_dataset(symbol, interval)
    train_df, test_df = split_dataset(df, train_split=0.8)
    
    # Create unique tensorboard log dir for this run
    tb_log_dir = repro_dir / "training" / "tensorboard"
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = repro_dir / "ppo_model.zip"
    
    train_ppo(
        train_df,
        timesteps=timesteps,
        seed=seed,
        tensorboard_log_dir=str(tb_log_dir),
        model_out=str(model_path),
        taker_fee=0.0004,
        slippage_bps=1.0,
        risk_penalty=0.1,
        position_fraction=1.0,
        initial_balance=10000.0,
    )
    print(f"  ✓ Model trained: {model_path}")
    
    # Evaluate PPO (for training metrics)
    print("\n[5/6] Evaluating PPO...")
    eval_report_path = repro_dir / "training_metrics.json"
    eval_metrics = evaluate_ppo(
        str(model_path),
        test_df,
        seed=seed,
        taker_fee=0.0004,
        slippage_bps=1.0,
        risk_penalty=0.1,
        position_fraction=1.0,
        initial_balance=10000.0,
        report_out=str(eval_report_path),
    )
    print(f"  ✓ Evaluation complete: {eval_report_path}")
    
    # Step 5: Backtest PPO
    print("\n[6/6] Running backtests...")
    # Lazy import to avoid circular dependencies
    from autonomous_rl_trading_bot.backtest.runner import run_backtest
    
    ppo_backtest_dir = repro_dir / "ppo"
    ppo_backtest_dir.mkdir(parents=True, exist_ok=True)
    
    ppo_result = run_backtest(
        mode=mode,
        policy="ppo",
        model_path=str(model_path),
        symbol=symbol,
        interval=interval,
        train_split=0.8,
        output_dir=str(ppo_backtest_dir),
    )
    print(f"  ✓ PPO backtest complete: {ppo_backtest_dir}")
    
    # Backtest SMA baseline
    sma_backtest_dir = repro_dir / "sma"
    sma_backtest_dir.mkdir(parents=True, exist_ok=True)
    
    sma_result = run_backtest(
        mode=mode,
        policy="sma",
        symbol=symbol,
        interval=interval,
        train_split=0.8,
        output_dir=str(sma_backtest_dir),
    )
    print(f"  ✓ SMA backtest complete: {sma_backtest_dir}")
    
    # Generate comparison
    print("\n[COMPARISON] Generating comparison report...")
    comparison = _generate_comparison(
        ppo_metrics=ppo_result["metrics"],
        sma_metrics=sma_result["metrics"],
        run_id=run_id,
        symbol=symbol,
        interval=interval,
        timesteps=timesteps,
        seed=seed,
    )
    
    comparison_json_path = repro_dir / "comparison.json"
    with open(comparison_json_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    comparison_md_path = repro_dir / "comparison.md"
    _write_comparison_md(comparison, comparison_md_path)
    
    print(f"  ✓ Comparison report: {comparison_json_path}")
    print(f"  ✓ Comparison markdown: {comparison_md_path}")
    
    # Save run config
    run_config = {
        "run_id": run_id,
        "symbol": symbol,
        "interval": interval,
        "days": days,
        "window": window,
        "timesteps": timesteps,
        "mode": mode,
        "seed": seed,
        "dataset_path": str(dataset_file),
        "model_path": str(model_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    
    run_config_path = repro_dir / "run_config.json"
    with open(run_config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    
    print(f"\n[REPRO] Complete! Run ID: {run_id}")
    print(f"  Run config: {run_config_path}")
    print(f"  PPO backtest: {ppo_backtest_dir}")
    print(f"  SMA backtest: {sma_backtest_dir}")
    print(f"  Comparison: {comparison_json_path}")
    
    return {
        "run_id": run_id,
        "run_dir": str(repro_dir),
        "run_config": str(run_config_path),
        "ppo_backtest": str(ppo_backtest_dir),
        "sma_backtest": str(sma_backtest_dir),
        "comparison": str(comparison_json_path),
        "comparison_md": str(comparison_md_path),
    }


def _generate_comparison(
    ppo_metrics: Dict[str, Any],
    sma_metrics: Dict[str, Any],
    run_id: str,
    symbol: str,
    interval: str,
    timesteps: int,
    seed: int,
) -> Dict[str, Any]:
    """Generate comparison between PPO and SMA."""
    # Determine winner
    ppo_return = ppo_metrics.get("total_return", 0.0)
    sma_return = sma_metrics.get("total_return", 0.0)
    ppo_sharpe = ppo_metrics.get("sharpe", 0.0)
    sma_sharpe = sma_metrics.get("sharpe", 0.0)
    
    winner = "ppo"
    if sma_return > ppo_return and sma_sharpe > ppo_sharpe:
        winner = "sma"
    elif sma_return > ppo_return:
        winner = "sma"  # Prefer return over sharpe
    elif abs(sma_return - ppo_return) < 0.01 and sma_sharpe > ppo_sharpe:
        winner = "sma"
    
    return {
        "run_id": run_id,
        "symbol": symbol,
        "interval": interval,
        "timesteps": timesteps,
        "seed": seed,
        "ppo": {
            "total_return": float(ppo_metrics.get("total_return", 0.0)),
            "sharpe": float(ppo_metrics.get("sharpe", 0.0)),
            "sortino": float(ppo_metrics.get("sortino", 0.0)),
            "max_drawdown": float(ppo_metrics.get("max_drawdown", 0.0)),
            "calmar": float(ppo_metrics.get("calmar", 0.0)),
            "num_trades": int(ppo_metrics.get("num_trades", 0)),
            "win_rate": float(ppo_metrics.get("win_rate", 0.0)),
            "avg_trade_pnl": float(ppo_metrics.get("avg_trade_pnl", 0.0)),
            "profit_factor": float(ppo_metrics.get("profit_factor", 0.0)),
            "final_equity": float(ppo_metrics.get("final_equity", 0.0)),
        },
        "sma": {
            "total_return": float(sma_metrics.get("total_return", 0.0)),
            "sharpe": float(sma_metrics.get("sharpe", 0.0)),
            "sortino": float(sma_metrics.get("sortino", 0.0)),
            "max_drawdown": float(sma_metrics.get("max_drawdown", 0.0)),
            "calmar": float(sma_metrics.get("calmar", 0.0)),
            "num_trades": int(sma_metrics.get("num_trades", 0)),
            "win_rate": float(sma_metrics.get("win_rate", 0.0)),
            "avg_trade_pnl": float(sma_metrics.get("avg_trade_pnl", 0.0)),
            "profit_factor": float(sma_metrics.get("profit_factor", 0.0)),
            "final_equity": float(sma_metrics.get("final_equity", 0.0)),
        },
        "winner": winner,
        "ppo_advantage": {
            "return_diff": float(ppo_return - sma_return),
            "sharpe_diff": float(ppo_sharpe - sma_sharpe),
        },
    }


def _write_comparison_md(comparison: Dict[str, Any], output_path: Path) -> None:
    """Write comparison markdown report."""
    ppo = comparison["ppo"]
    sma = comparison["sma"]
    winner = comparison["winner"]
    
    md = f"""# Backtest Comparison Report

**Run ID:** `{comparison['run_id']}`  
**Symbol:** {comparison['symbol']}  
**Interval:** {comparison['interval']}  
**Training Timesteps:** {comparison['timesteps']:,}  
**Seed:** {comparison['seed']}  
**Generated:** {datetime.now(timezone.utc).isoformat()}

## Summary

**Winner:** {winner.upper()}

PPO vs SMA:
- Return difference: {comparison['ppo_advantage']['return_diff']:.4f} ({'+' if comparison['ppo_advantage']['return_diff'] > 0 else ''}{comparison['ppo_advantage']['return_diff']*100:.2f}%)
- Sharpe difference: {comparison['ppo_advantage']['sharpe_diff']:.4f}

## Metrics Comparison

| Metric | PPO | SMA | Winner |
|--------|-----|-----|--------|
| **Total Return** | {ppo['total_return']:.4f} ({ppo['total_return']*100:.2f}%) | {sma['total_return']:.4f} ({sma['total_return']*100:.2f}%) | {'PPO' if ppo['total_return'] > sma['total_return'] else 'SMA'} |
| **Sharpe Ratio** | {ppo['sharpe']:.4f} | {sma['sharpe']:.4f} | {'PPO' if ppo['sharpe'] > sma['sharpe'] else 'SMA'} |
| **Sortino Ratio** | {ppo['sortino']:.4f} | {sma['sortino']:.4f} | {'PPO' if ppo['sortino'] > sma['sortino'] else 'SMA'} |
| **Max Drawdown** | {ppo['max_drawdown']:.4f} ({ppo['max_drawdown']*100:.2f}%) | {sma['max_drawdown']:.4f} ({sma['max_drawdown']*100:.2f}%) | {'PPO' if ppo['max_drawdown'] < sma['max_drawdown'] else 'SMA'} |
| **Calmar Ratio** | {ppo['calmar']:.4f} | {sma['calmar']:.4f} | {'PPO' if ppo['calmar'] > sma['calmar'] else 'SMA'} |
| **Num Trades** | {ppo['num_trades']} | {sma['num_trades']} | - |
| **Win Rate** | {ppo['win_rate']:.4f} ({ppo['win_rate']*100:.2f}%) | {sma['win_rate']:.4f} ({sma['win_rate']*100:.2f}%) | {'PPO' if ppo['win_rate'] > sma['win_rate'] else 'SMA'} |
| **Avg Trade PnL** | {ppo['avg_trade_pnl']:.4f} | {sma['avg_trade_pnl']:.4f} | {'PPO' if ppo['avg_trade_pnl'] > sma['avg_trade_pnl'] else 'SMA'} |
| **Profit Factor** | {ppo['profit_factor']:.4f} | {sma['profit_factor']:.4f} | {'PPO' if ppo['profit_factor'] > sma['profit_factor'] else 'SMA'} |
| **Final Equity** | {ppo['final_equity']:.2f} | {sma['final_equity']:.2f} | {'PPO' if ppo['final_equity'] > sma['final_equity'] else 'SMA'} |

## Detailed Results

### PPO Model
- **Total Return:** {ppo['total_return']:.4f} ({ppo['total_return']*100:.2f}%)
- **Sharpe Ratio:** {ppo['sharpe']:.4f}
- **Max Drawdown:** {ppo['max_drawdown']:.4f} ({ppo['max_drawdown']*100:.2f}%)
- **Trades:** {ppo['num_trades']}
- **Win Rate:** {ppo['win_rate']:.4f} ({ppo['win_rate']*100:.2f}%)

### SMA Baseline
- **Total Return:** {sma['total_return']:.4f} ({sma['total_return']*100:.2f}%)
- **Sharpe Ratio:** {sma['sharpe']:.4f}
- **Max Drawdown:** {sma['max_drawdown']:.4f} ({sma['max_drawdown']*100:.2f}%)
- **Trades:** {sma['num_trades']}
- **Win Rate:** {sma['win_rate']:.4f} ({sma['win_rate']*100:.2f}%)

## Conclusion

The **{winner.upper()}** strategy performed better overall based on total return and Sharpe ratio.
"""
    
    output_path.write_text(md, encoding="utf-8")
