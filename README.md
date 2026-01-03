# Autonomous RL Trading Bot (v1)

## Quickstart (Windows PowerShell)

```powershell
cd "Autonomous RL Trading Bot"
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
pytest
python -c "import autonomous_rl_trading_bot; print(autonomous_rl_trading_bot.__version__)"
```

## Demo: Steps 3-5 (Quick Pipeline)

Run these commands in sequence to build a dataset, train a model, and compare baselines:

### Step 3: Fetch Data
```powershell
# Fetch spot data (or use --mode futures for futures)
python run_fetch_data.py --mode spot --symbol BTCUSDT --interval 1m --minutes 120
```

### Step 4: Build Dataset (with leakage-free splits)
```powershell
# Build dataset with train/val/test splits (default: 75/10/15)
python run_build_dataset.py --mode spot --symbol BTCUSDT --interval 1m --minutes 120

# Or customize splits:
python run_build_dataset.py --mode spot --train-frac 0.8 --val-frac 0.1 --test-frac 0.1
```

### Step 5: Run Baseline Strategies
```powershell
# Run all baselines on TEST split (default)
python run_baselines.py --mode spot

# Or specify dataset:
python run_baselines.py --mode spot --dataset-id <dataset_id>

# Customize strategy parameters:
python run_baselines.py --mode spot --sma-fast 5 --sma-slow 20 --rsi-period 14 --rsi-low 25 --rsi-high 75
```

Outputs are saved to `artifacts/baselines/{dataset_id}/{timestamp_run_id}/`:
- `baseline_comparison.md` - Ranked comparison by Sharpe and total return
- `{strategy}/metrics.json` - Full metrics (Sharpe, Sortino, Calmar, win rate, etc.)
- `{strategy}/equity.csv` - Equity curve over time
- `{strategy}/trades.csv` - All trades
- `{strategy}/equity_drawdown.png` - Equity and drawdown plots
- `{strategy}/trades_price.png` - Trades overlaid on price chart

### Optional: Train RL Model
```powershell
# Train on train split, evaluate on val split
python run_train.py --mode spot --train-split train --eval-split val --timesteps 50000
```

## Dev commands
- Format: `python -m black .`
- Lint: `python -m ruff check .`
- Test: `pytest`
