# Validation Pack

**Generated:** 2026-01-18T18:21:09.913449

## Dataset Statistics

- **Rows:** 2,818
- **Columns:** 5
- **Date Range:** 2026-01-16T19:23:00 to 2026-01-18T18:20:00
- **NaN Values:** None detected

## Integrity Check

✓ **Integrity Check PASSED**
- Final equity (report): 9908.83
- Final equity (curve): 9908.83
- Difference: 0.00

## Backtest Metrics

**Policy:** sma

| Metric | Value |
|--------|-------|
| total_return | -0.0091 |
| sharpe | -0.0013 |
| sortino | -0.0008 |
| max_drawdown | 1.0019 |
| calmar | -0.0091 |
| num_trades | 7.0000 |
| win_rate | 0.1429 |
| avg_trade_pnl | -13.0225 |
| profit_factor | 0.0010 |
| final_equity | 9908.8317 |

## Commands Used

```bash
# Download data
python scripts/download_binance_futures.py --symbol BTCUSDT --interval 1m --days 30

# Build features
python scripts/build_features.py --symbol BTCUSDT --interval 1m

# Make dataset
python scripts/make_dataset.py --symbol BTCUSDT --interval 1m --window 30

# Train PPO
arbt train --symbol BTCUSDT --interval 1m --timesteps 200000

# Backtest
arbt backtest --symbol BTCUSDT --interval 1m --mode spot --policy ppo --model models/ppo_trader.zip
arbt backtest --symbol BTCUSDT --interval 1m --mode spot --policy sma
```
