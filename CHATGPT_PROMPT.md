# Comprehensive Codebase Explanation for ChatGPT

## Project Overview

This is an **Autonomous Reinforcement Learning Trading Bot** - a complete, production-ready system for algorithmic trading on Binance using reinforcement learning. The system supports both **spot trading** and **futures trading** with comprehensive features including data management, model training, backtesting, live trading, and risk management.

## Core Purpose

The bot uses **Proximal Policy Optimization (PPO)** and other RL algorithms from Stable-Baselines3 to learn optimal trading strategies. It processes market data (OHLCV candles), extracts technical features, trains RL agents in simulated trading environments, evaluates performance through backtesting, and can execute live trades with built-in safeguards.

## System Architecture

### 1. Data Pipeline (`src/autonomous_rl_trading_bot/data/`)

**Components:**
- `binance_spot.py` / `binance_futures.py`: Fetch OHLCV (Open, High, Low, Close, Volume) data from Binance via CCXT library
- `candles_store.py`: SQLite database storage for historical candles
- `dataset_builder.py`: Builds datasets with **leakage-free** train/validation/test splits (default: 75%/10%/15%)
- `dataset_loader.py`: Unified dataset loading with format detection (Parquet or NPZ)

**Key Features:**
- Time-based splits prevent data leakage (train before val before test chronologically)
- Supports 1-minute timeframe candles
- Stores data in SQLite for fast querying
- Exports processed datasets to Parquet format

### 2. Feature Engineering (`src/autonomous_rl_trading_bot/features/`)

**Components:**
- `indicators.py`: Technical indicators (returns, EMA, RSI, ATR, volatility, volume_delta)
- `scaling.py`: RobustScaler with leakage prevention (fit on train only, transform on val/test)
- `feature_pipeline.py`: Deterministic feature computation pipeline
- `alignment.py`: Time-aligned feature computation using `asof_align()` to prevent lookahead bias

**Features Computed:**
1. **log_return**: Logarithmic returns (log(close[t] / close[t-1]))
2. **return**: Simple returns
3. **close_norm**: Normalized close price
4. **vol_norm**: Normalized volume (log1p transformation)
5. **atr_norm**: Average True Range normalized by price
6. **volatility**: Rolling standard deviation of returns
7. **volume_delta**: Volume change relative to rolling average

**Critical Design:**
- Features are fit/scaled on training data ONLY
- Validation and test data are transformed using the scaler fit on training data
- This prevents data leakage that would inflate performance metrics

### 3. Trading Environments (`src/autonomous_rl_trading_bot/envs/`)

**Base Environment (`base_env.py`):**
- Common functionality for both spot and futures
- Drawdown calculation
- Window views for feature lookback
- Observation flattening

**Spot Environment (`spot_env.py`):**
- **Actions (Discrete=3):**
  - 0: HOLD - Maintain current position
  - 1: BUY - Open/increase long position
  - 2: SELL/CLOSE - Close position or sell
- Uses cash and asset holdings
- Applies taker fees and slippage

**Futures Environment (`futures_env.py`):**
- **Actions (Discrete=5):**
  - 0: HOLD - Maintain current position
  - 1: LONG - Open/increase long position
  - 2: SHORT - Open/increase short position
  - 3: REDUCE - Reduce position by fraction
  - 4: CLOSE - Close all positions
- Uses margin and leverage (default 3x)
- Supports both long and short positions
- Includes maintenance margin requirements
- Stops on liquidation if enabled

**State Space (219 dimensions):**
- **Feature Window (210 dims):** Lookback × Features = 30 × 7 = 210
  - 30 timesteps of historical data
  - 7 features per timestep
- **Account State (9 dims):**
  1. Margin balance (normalized)
  2. Equity (normalized)
  3. Used margin (normalized)
  4. Free margin (normalized)
  5. Position fraction (0-1)
  6. Position direction (-1/0/+1)
  7. Leverage used
  8. Drawdown (0-1)
  9. Unrealized PnL (normalized)

**Reward Function:**
```
reward[t] = log(equity[t] / equity[t-1])
```
- Logarithmic equity return
- Additive: total return = sum of rewards
- Scale-invariant
- Naturally penalizes drawdowns

**RL Wrapper (`rl/env_trading.py`):**
- Gymnasium-compatible wrapper for Stable-Baselines3
- Converts internal environment to SB3 format
- Handles action/observation spaces

### 4. Training Pipeline (`src/autonomous_rl_trading_bot/training/`)

**Components:**
- `train_pipeline.py`: Main training pipeline (Day-2, uses Parquet datasets)
- `trainer.py`: Legacy trainer (uses NPZ datasets)
- `sb3_factory.py`: Factory for creating SB3 models (PPO, DQN)
- `checkpointing.py`: Model checkpoint management
- `callbacks.py`: Training callbacks for logging

**Process:**
1. Load dataset from Parquet
2. Create trading environment with train split
3. Initialize PPO agent from Stable-Baselines3
4. Train on train split
5. Evaluate on validation split
6. Save model checkpoints
7. Generate training metrics and reports

**Outputs:**
- Trained model files (`.zip` format, SB3 compatible)
- Training metrics (TensorBoard logs)
- Evaluation reports (HTML/PDF)
- Reproducibility artifacts (`repro.json`)

### 5. Backtesting (`src/autonomous_rl_trading_bot/backtest/` and `evaluation/`)

**Components:**
- `backtest/runner.py`: Modern backtest runner (Day-4, uses Parquet)
- `evaluation/backtester.py`: Legacy backtester (uses NPZ)
- `evaluation/baselines.py`: Baseline strategies for comparison
- `evaluation/metrics.py`: Performance metrics calculation
- `evaluation/plots.py`: Visualization (equity curves, drawdowns)
- `evaluation/reporting.py`: HTML/PDF report generation

**Baseline Strategies:**
1. **Buy-and-Hold**: Simple buy and hold strategy
2. **SMA Crossover**: Moving average crossover (fast SMA crosses slow SMA)
3. **EMA Crossover**: Exponential moving average crossover
4. **RSI Reversion**: Mean reversion based on RSI (buy when RSI < low threshold, sell when RSI > high threshold)

**Metrics Calculated:**
- Total return
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- Maximum drawdown
- Win rate
- Average trade duration
- Profit factor

**Outputs:**
- Equity curve plots
- Drawdown plots
- Trade logs (CSV)
- Metrics reports (JSON, Markdown, HTML, PDF)
- Comparison reports ranking strategies

### 6. Live Trading (`src/autonomous_rl_trading_bot/live/`)

**Components:**
- `live_runner.py`: Main live trading loop
- `safeguards.py`: Risk management and rate limiting
- `candle_sync.py`: Synchronizes live candles with database
- `position_sync.py`: Position tracking and synchronization

**Execution Flow:**
1. Poll latest closed candle from exchange
2. Compute features using same pipeline as training
3. Get action from trained policy
4. Check risk limits and safeguards
5. Execute trade via broker (if allowed)
6. Update position tracking
7. Log results and metrics

**Safeguards:**
- **Rate Limiting:**
  - Minimum seconds between trades
  - Maximum trades per hour
  - Maximum orders per minute
  - Uses rolling windows (auto-reset as time passes)
- **Drawdown Limits:** Kill switch if drawdown exceeds threshold
- **Position Limits:** Maximum position size
- **Slippage Checks:** Validates execution prices

**Safety:**
- Requires `ALLOW_NETWORK=1` environment variable for live trading
- Paper trading mode available for testing
- Comprehensive logging and monitoring

### 7. Brokers (`src/autonomous_rl_trading_bot/broker/`)

**Components:**
- `base_broker.py`: Abstract base class for brokers
- `spot_broker.py`: CCXT-based spot trading broker
- `futures_broker.py`: CCXT-based futures trading broker
- `execution.py`: Order sizing and execution helpers
- `paper/`: Paper trading implementations (simulated trading)

**Functionality:**
- Order placement (market orders)
- Position management
- Balance queries
- Trade execution with fees and slippage modeling

### 8. Risk Management (`src/autonomous_rl_trading_bot/risk/`)

**Components:**
- `risk_manager.py`: Main risk management system
- `spot_risk.py`: Spot-specific risk checks
- `futures_risk.py`: Futures-specific risk checks (leverage, margin)
- `kill_switch.py`: Emergency stop mechanism

**Risk Checks:**
- Maximum position size
- Maximum leverage (futures)
- Drawdown limits
- Daily loss limits
- Position concentration limits

### 9. Configuration System (`src/autonomous_rl_trading_bot/common/config.py`)

**YAML-Based Configuration:**
- `configs/base.yaml`: Base configuration with defaults
- Mode-specific overrides (spot.yaml, futures.yaml)
- Hierarchical merging (overrides extend base)
- Dot notation access: `cfg.get("exchange.name")`

**Configuration Includes:**
- Exchange settings (Binance API endpoints)
- Trading parameters (symbols, intervals, leverage)
- Model hyperparameters (PPO learning rate, batch size)
- Risk limits
- Feature engineering parameters
- Backtest settings

**Reproducibility:**
- Config hash computed deterministically
- Stored in `repro.json` for reproducibility
- Ensures same config = same results

### 10. Storage (`src/autonomous_rl_trading_bot/storage/`)

**Components:**
- `sqlite_store.py`: SQLite database for runs, backtests, training jobs
- `parquet_store.py`: Parquet file storage for datasets
- `artifact_store.py`: Artifact management (reports, models, plots)
- `models.py`: Data models (RunModel, BacktestModel, TrainJobModel)

**Database Schema:**
- `runs`: Training run metadata
- `backtests`: Backtest results
- `train_jobs`: Training job tracking
- `candles`: Historical OHLCV data (in separate database)

### 11. Modes System (`src/autonomous_rl_trading_bot/modes/`)

**Purpose:** Unified abstraction for spot vs futures trading

**Components:**
- `registry.py`: Mode registry mapping "spot" or "futures" to mode definitions
- `mode_defs.py`: Mode definitions with factories for:
  - Market data clients
  - Brokers
  - Risk managers
  - Environments
- `schemas.py`: Configuration schemas per mode

**Benefits:**
- Single codebase supports both spot and futures
- Mode-specific configuration validation
- Easy to add new trading modes

### 12. CLI (`src/autonomous_rl_trading_bot/cli.py`)

**Unified Command-Line Interface:**

```bash
# Dataset operations
arbt dataset fetch --mode spot --symbol BTCUSDT --interval 1m --minutes 120
arbt dataset build --mode spot --symbol BTCUSDT --interval 1m

# Training
arbt train --mode spot --timesteps 100000

# Backtesting
arbt backtest --mode spot --policy ppo --model models/ppo_trader.zip --symbol BTCUSDT --interval 1m
arbt backtest --mode spot --policy buyhold --symbol BTCUSDT --interval 1m

# Live trading
arbt live --mode spot

# Dashboard
arbt dashboard

# Baselines
arbt baselines --mode spot

# Reproducibility pipeline
arbt repro --symbol BTCUSDT --interval 1m --days 30 --mode spot --seed 42
```

**Commands:**
- `dataset fetch`: Download market data
- `dataset build`: Build dataset with splits
- `train`: Train RL model
- `backtest`: Run backtest evaluation
- `live`: Start live trading
- `dashboard`: Launch web dashboard
- `baselines`: Run baseline strategies
- `repro`: Complete reproducibility pipeline
- `verify`: Run verification tests

### 13. Dashboard (`src/autonomous_rl_trading_bot/dashboard/`)

**Web-Based Dashboard (Dash/Plotly):**
- Real-time monitoring of live trading
- Equity curve visualization
- Trade history
- Risk metrics
- Performance analytics
- Can run as desktop app (`app_desktop.py`)

**Components:**
- `app.py`: Main Dash application
- `layout.py`: UI layout
- `callbacks.py`: Interactive callbacks
- `components.py`: Reusable UI components
- `data_api.py`: Data API for dashboard
- `live_session.py`: Live trading session management

### 14. Reproducibility (`src/autonomous_rl_trading_bot/repro/`)

**Complete Reproducibility Pipeline:**
- Downloads data
- Builds dataset
- Trains model
- Runs backtests
- Compares baselines
- Generates comprehensive reports

**Artifacts:**
- `repro.json`: Contains seed, dataset_id, dataset_hash, config_hash
- All outputs are deterministic given same inputs
- Can reproduce exact results

### 15. Common Utilities (`src/autonomous_rl_trading_bot/common/`)

**Key Utilities:**
- `reproducibility.py`: Seed management (numpy, random, torch)
- `hashing.py`: Deterministic hashing (SHA256 of files, datasets)
- `types.py`: Common type definitions (MarketType, OrderRequest, etc.)
- `time.py`: Time utilities (timeframe parsing, timestamp conversion)
- `logging.py`: Structured logging
- `db.py`: Database connection management
- `paths.py`: Path resolution utilities
- `exceptions.py`: Custom exceptions

## Key Design Principles

### 1. Leakage-Free Design
- Train/val/test splits are chronological (no future data in training)
- Features are fit on train only, transformed on val/test
- No lookahead bias in feature computation

### 2. Deterministic Reproducibility
- Fixed random seeds (numpy, random, torch)
- Deterministic algorithms
- Hash-based dataset identification
- Config hashing for reproducibility

### 3. Modular Architecture
- Clear separation of concerns
- Mode system for spot/futures abstraction
- Pluggable components (brokers, risk managers, environments)

### 4. Risk-First Approach
- Comprehensive safeguards
- Rate limiting
- Drawdown limits
- Position limits
- Kill switch mechanism

### 5. Research-Grade Outputs
- Detailed metrics
- Visualization (plots, charts)
- HTML/PDF reports
- Reproducibility artifacts
- Comparison reports

## Data Flow

```
1. DATA INGESTION
   Binance API → CCXT Client → SQLite Storage

2. DATASET BUILDING
   SQLite Candles → Feature Computation → Train/Val/Test Splits → Parquet

3. FEATURE ENGINEERING
   Raw OHLCV → Indicators → Scaling (fit on train) → Feature Matrix

4. TRAINING
   Feature Matrix → TradingEnv → PPO Agent → Trained Model

5. EVALUATION
   Trained Model → Backtester → Metrics → Reports

6. LIVE TRADING
   Live Candles → Feature Pipeline → Policy → Risk Checks → Broker → Execution
```

## File Structure

```
src/autonomous_rl_trading_bot/
├── agents/          # RL agents (PPO, DQN)
├── backtest/        # Backtest engine
├── broker/          # Broker adapters (spot, futures, paper)
├── commands/        # CLI command implementations
├── common/          # Shared utilities
├── dashboard/       # Web dashboard
├── data/            # Data fetching & dataset building
├── envs/            # Gymnasium trading environments
├── evaluation/      # Backtesting & baselines
├── features/        # Feature engineering
├── live/            # Live trading runner
├── modes/           # Mode registry (spot/futures)
├── repro/           # Reproducibility pipeline
├── risk/            # Risk management
├── rl/              # RL utilities
├── storage/         # Database & artifact storage
├── training/        # Training pipeline
└── cli.py           # Unified CLI entrypoint

configs/             # YAML configuration files
tests/               # Pytest test suite
scripts/             # Utility scripts
```

## Dependencies

**Core:**
- numpy, pandas, pyarrow (data processing)
- PyYAML (configuration)
- gymnasium (RL environments)
- stable-baselines3 (RL algorithms)
- torch (neural networks)

**Trading:**
- ccxt (exchange API)

**ML:**
- scikit-learn (scaling, preprocessing)

**Visualization:**
- matplotlib, plotly (plotting)
- dash (web dashboard)

**Reporting:**
- reportlab (PDF generation)

**Web:**
- fastapi, uvicorn (API server)
- requests (HTTP client)

## Testing

**Test Suite (`tests/`):**
- Unit tests for individual components
- Integration tests for pipelines
- Smoke tests for end-to-end workflows
- Determinism tests for reproducibility

**Run Tests:**
```bash
pytest
arbt verify  # Runs pytest + smoke repro test
```

## Usage Examples

### Complete Workflow

```bash
# 1. Fetch data
arbt dataset fetch --mode spot --symbol BTCUSDT --interval 1m --minutes 120

# 2. Build dataset
arbt dataset build --mode spot --symbol BTCUSDT --interval 1m

# 3. Train model
arbt train --mode spot --timesteps 100000

# 4. Backtest
arbt backtest --mode spot --policy ppo --model models/ppo_trader.zip --symbol BTCUSDT --interval 1m

# 5. Compare with baselines
arbt baselines --mode spot

# 6. Live trading (requires ALLOW_NETWORK=1)
ALLOW_NETWORK=1 arbt live --mode spot
```

### Reproducibility Pipeline

```bash
arbt repro --symbol BTCUSDT --interval 1m --days 30 --mode spot --seed 42 --timesteps 200000
```

This runs the complete pipeline and generates reproducible results.

## Important Notes

1. **No Network Calls at Import Time**: All network code is lazy-loaded to avoid import-time side effects
2. **Deterministic by Default**: Seeds set, hashing for reproducibility
3. **Leakage-Free Splits**: Train/val/test split by time, features fit on train only
4. **Safety First**: Live trading requires explicit `ALLOW_NETWORK=1` env var
5. **Rolling Windows**: Rate limiting uses rolling windows (auto-reset as time passes)
6. **Mode Abstraction**: Single codebase supports both spot and futures trading
7. **Research-Grade**: Suitable for academic research and evaluation

## Code Style

- Python 3.11+ (type hints, dataclasses)
- Black formatting (line length 100)
- Ruff linting (E, F, I, B rules)
- Docstrings for public functions/classes

## Summary

This is a **complete, production-ready RL trading system** that:
- Fetches and processes market data with leakage-free splits
- Trains RL agents (PPO/DQN) in realistic trading environments
- Evaluates performance through comprehensive backtesting
- Supports live trading with robust risk management
- Generates research-grade reports and visualizations
- Ensures reproducibility through deterministic design
- Supports both spot and futures trading on Binance

The system is designed for both **practical trading** and **academic research**, with emphasis on correctness, reproducibility, and risk management.
