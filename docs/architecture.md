# System Architecture Documentation
## Autonomous RL Trading Bot

---

## 1. Core Components

### 1.1 MarketDataLoader
**Location:** `src/autonomous_rl_trading_bot/data/`

**Components:**
- `binance_futures.py` - Binance Futures API client
- `binance_spot.py` - Binance Spot API client  
- `candles_store.py` - SQLite storage layer
- `dataset_builder.py` - Dataset construction with splits

**Functionality:**
- Fetches OHLCV data from Binance
- Stores in SQLite database
- Builds datasets with train/val/test splits (75/10/15)
- Supports 1-minute timeframe
- Leakage-free feature engineering

### 1.2 FeatureEngine
**Location:** `src/autonomous_rl_trading_bot/features/`

**Components:**
- `indicators.py` - Technical indicators (returns, EMA, RSI, ATR, volatility, volume_delta)
- `scaling.py` - RobustScaler with leakage prevention
- `feature_pipeline.py` - Feature computation pipeline
- `alignment.py` - Time-aligned feature computation

**Features Computed:**
1. Returns (simple and logarithmic)
2. Normalized close price
3. Normalized volume (log1p)
4. ATR (Average True Range) - normalized by price
5. Volatility (rolling std of returns)
6. Volume delta (change relative to rolling average)

**Scaling:**
- Fit on training data only
- Transform validation and test data
- Prevents data leakage

### 1.3 TradingEnv (Gymnasium-Compatible)
**Location:** `src/autonomous_rl_trading_bot/rl/`

**File:** `env_trading.py`

**State Space:**
- **Feature Window:** Lookback × Feature Dim (default: 30 × 7 = 210)
- **Account State:** 9 dimensions (equity, margin, position, drawdown, etc.)
- **Total:** 219 dimensions

**Action Space (Discrete):**
- **Spot:** 0=HOLD, 1=LONG, 2=FLAT
- **Futures:** 0=HOLD, 1=LONG, 2=SHORT, 3=FLAT

**Reward Function:**
```
reward[t] = log(equity[t] / equity[t-1])
```
- Risk-adjusted log return
- Encourages consistent growth
- Penalizes drawdowns naturally

**Episode Structure:**
- Episodes correspond to dataset splits
- Terminate at end of split or on liquidation
- Reset resets portfolio and time index

### 1.4 PPO Agent
**Location:** `src/autonomous_rl_trading_bot/training/`

**Components:**
- `trainer.py` - Training orchestration
- `sb3_factory.py` - Model creation
- `callbacks.py` - Evaluation callbacks
- `checkpointing.py` - Model checkpointing

**Hyperparameters:**
- Algorithm: PPO (Proximal Policy Optimization)
- Learning rate: 0.0003
- Batch size: 64
- N steps: 2048
- N epochs: 10
- Clip range: 0.2
- Gamma: 0.99

**Policy Network:**
- MLP (Multi-Layer Perceptron)
- Input: Flattened observation (219 dims)
- Output: Action probabilities (discrete)

### 1.5 Reward Function
**Design:** Log equity return

**Formula:**
```
r[t] = log(equity[t] / equity[t-1])
```

**Properties:**
- Additive over time
- Scale-invariant
- Risk-sensitive (drawdowns yield negative rewards)

**Rationale:**
- Better signal-to-noise than PnL
- Encourages consistent growth
- Implicit risk adjustment

### 1.6 Risk Manager
**Location:** `src/autonomous_rl_trading_bot/risk/` and `live/safeguards.py`

**Mechanisms:**
1. **Drawdown Limits**
   - Maximum drawdown: 30% (configurable)
   - Kill switch on breach

2. **Position Limits**
   - Maximum position size
   - Leverage limits (default: 3x)
   - Maintenance margin requirements

3. **Rate Limiting**
   - Min seconds between trades
   - Max trades per hour
   - Max orders per minute

4. **Slippage & Fees**
   - Configurable slippage (default: 5 bps)
   - Taker fee modeling

### 1.7 Execution Engine
**Location:** `src/autonomous_rl_trading_bot/broker/`

**Components:**
- `futures_broker.py` - Futures broker adapter
- `spot_broker.py` - Spot broker adapter
- `execution.py` - Order execution helpers
- `paper/` - Paper trading implementations

**Features:**
- CCXT-based exchange integration
- Paper trading support
- Demo/testnet mode
- Realistic execution modeling

---

## 2. Data Flow

```
1. DATA INGESTION
   Binance API → CCXT Client → SQLite Storage

2. DATASET BUILDING
   SQLite Candles → Feature Computation → Train/Val/Test Splits

3. FEATURE ENGINEERING
   Raw OHLCV → Indicators → Scaling → Feature Matrix

4. TRAINING
   Feature Matrix → TradingEnv → PPO Agent → Trained Model

5. EVALUATION
   Trained Model → Backtester → Metrics → Reports

6. LIVE TRADING
   Live Candles → Policy → Risk Checks → Broker → Execution
```

---

## 3. State Space Definition

### 3.1 Observation Components

**Feature Window (210 dims):**
- Lookback: 30 timesteps
- Features per timestep: 7
  - log_return
  - return
  - close_norm
  - vol_norm
  - atr_norm
  - volatility
  - volume_delta

**Account State (9 dims):**
1. Margin balance (normalized by initial equity)
2. Equity (normalized)
3. Used margin (normalized)
4. Free margin (normalized)
5. Position fraction (0-1)
6. Position direction (-1/0/+1)
7. Leverage used
8. Drawdown (0-1)
9. Unrealized PnL (normalized)

**Total:** 219 dimensions

### 3.2 Normalization

- Features: RobustScaler (fit on train, transform on val/test)
- Account state: Normalized by initial equity
- Prevents scale issues in neural network

---

## 4. Action Space Definition

### 4.1 Futures Trading (5 actions)

- **0: HOLD** - Maintain current position
- **1: LONG** - Open/increase long position
- **2: SHORT** - Open/increase short position
- **3: REDUCE** - Reduce position by fraction
- **4: CLOSE** - Close all positions

### 4.2 Position Sizing

- Uses available margin with leverage
- Configurable position fraction
- Respects maintenance margin requirements
- Prevents over-leveraging

---

## 5. Reward Function Design

### 5.1 Primary Reward

```
r[t] = log(equity[t] / equity[t-1])
```

### 5.2 Properties

1. **Additive:** Total return = sum of rewards
2. **Scale-Invariant:** Works across equity levels
3. **Risk-Sensitive:** Drawdowns yield negative rewards

### 5.3 Risk Adjustment

- Implicit through equity changes
- Large drawdowns → large negative rewards
- Encourages consistent, low-volatility growth

---

## 6. Episode Structure

### 6.1 Episode Definition

- Episodes correspond to dataset splits
- Train episodes: Training split
- Eval episodes: Validation split
- Test episodes: Test split

### 6.2 Termination Conditions

1. End of split reached
2. Liquidation (futures)
3. Maximum drawdown exceeded
4. Minimum equity breached

### 6.3 Reset Behavior

- Reset portfolio state (equity, cash, position)
- Reset time index to start of split
- Reset peak equity tracking
- New random seed (if stochastic)

---

## 7. Training Pipeline

### 7.1 Dataset Preparation

1. Fetch OHLCV data from Binance
2. Build dataset with features
3. Split chronologically (75/10/15)
4. Scale features (fit on train)

### 7.2 Environment Setup

1. Create training environment (train split)
2. Create evaluation environment (val split)
3. Configure reward function
4. Set initial equity and risk limits

### 7.3 Model Training

1. Initialize PPO agent
2. Train for N timesteps
3. Periodic evaluation on validation set
4. Save checkpoints
5. Early stopping (optional)

### 7.4 Evaluation

1. Load trained model
2. Evaluate on test split
3. Compute metrics (Sharpe, drawdown, win rate)
4. Generate reports (HTML/PDF)
5. Compare to baselines

---

## 8. Backtesting Framework

### 8.1 Deterministic Design

- Fixed random seeds
- Deterministic execution
- Reproducible results
- Hash-based tracking

### 8.2 Metrics Computed

- Total return
- Sharpe ratio (annualized)
- Maximum drawdown
- Win rate
- Profit factor
- Sortino ratio
- Calmar ratio

### 8.3 Baseline Comparisons

- Buy-and-hold
- SMA crossover
- EMA crossover
- RSI mean reversion

---

## 9. Live Trading

### 9.1 Execution Flow

1. Poll latest closed candle
2. Compute features
3. Get action from policy
4. Check risk limits
5. Execute trade (if allowed)
6. Update position tracking
7. Log results

### 9.2 Safeguards

- Rate limiting
- Drawdown limits
- Position limits
- Kill switch

### 9.3 Monitoring

- Real-time dashboard
- Equity curve
- Trade log
- Risk metrics

---

## 10. Reproducibility

### 10.1 Deterministic Design

- Fixed seeds (numpy, random, torch)
- Deterministic algorithms
- Hash-based dataset identification

### 10.2 Artifacts

- `repro.json` with:
  - Random seed
  - Dataset hash
  - Config hash
  - Model checkpoint path

### 10.3 Configuration

- YAML-based configuration
- Hierarchical merging
- Mode-specific overrides
- Validated per mode

---

## 11. File Structure

```
src/autonomous_rl_trading_bot/
├── data/          # Data ingestion & dataset building
├── features/      # Feature engineering
├── rl/            # RL environment (Gymnasium)
├── training/      # PPO training pipeline
├── evaluation/    # Backtesting & metrics
├── live/          # Live trading runner
├── broker/        # Execution engine
├── risk/          # Risk management
├── dashboard/     # Web interface
├── common/        # Shared utilities
├── storage/       # Database persistence
└── modes/         # Mode registry (spot/futures)
```

---

## 12. Key Design Principles

1. **Leakage-Free:** Features fit on train only
2. **Deterministic:** Fixed seeds, reproducible results
3. **Modular:** Clear separation of concerns
4. **Risk-First:** Comprehensive safeguards
5. **Academic:** Suitable for research/evaluation

---

**Last Updated:** January 2025
