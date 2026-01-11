from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RunModel:
    """Model for runs table."""
    run_id: str
    kind: str
    mode: str
    status: str
    created_utc: str
    config_hash: str
    seed: int
    run_dir: Optional[str] = None
    run_json_path: Optional[str] = None
    run_log_path: Optional[str] = None
    global_log_path: Optional[str] = None


@dataclass
class BacktestModel:
    """Model for backtests table."""
    backtest_id: str
    run_id: str
    mode: str
    dataset_id: str
    market_type: str
    symbol: str
    interval: str
    started_utc: str
    finished_utc: Optional[str]
    status: str
    initial_cash: float
    final_equity: float
    total_return: float
    max_drawdown: float
    trade_count: int
    fee_total: float
    slippage_total: float


@dataclass
class TrainJobModel:
    """Model for train_jobs table."""
    train_id: str
    run_id: str
    mode: str
    market_type: str
    dataset_id: str
    algo: str
    total_timesteps: int
    seed: int
    started_utc: str
    finished_utc: Optional[str]
    status: str
    model_path: Optional[str] = None

