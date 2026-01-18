# Backtest Comparison Report

**Run ID:** `20260118T182124Z_spot_BTCUSDT_1m_w30_t2000_s42`  
**Symbol:** BTCUSDT  
**Interval:** 1m  
**Training Timesteps:** 2,000  
**Seed:** 42  
**Generated:** 2026-01-18T18:21:34.688975+00:00

## Summary

**Winner:** SMA

PPO vs SMA:
- Return difference: -0.0128 (-1.28%)
- Sharpe difference: -0.1352

## Metrics Comparison

| Metric | PPO | SMA | Winner |
|--------|-----|-----|--------|
| **Total Return** | -0.0222 (-2.22%) | -0.0093 (-0.93%) | SMA |
| **Sharpe Ratio** | -0.1366 | -0.0013 | SMA |
| **Sortino Ratio** | -0.1079 | -0.0008 | SMA |
| **Max Drawdown** | 0.5112 (51.12%) | 1.0021 (100.21%) | PPO |
| **Calmar Ratio** | -0.0433 | -0.0093 | SMA |
| **Num Trades** | 26 | 7 | - |
| **Win Rate** | 0.0769 (7.69%) | 0.1429 (14.29%) | SMA |
| **Avg Trade PnL** | -7.7183 | -13.2944 | PPO |
| **Profit Factor** | 0.0681 | 0.0010 | PPO |
| **Final Equity** | 9778.48 | 9906.93 | SMA |

## Detailed Results

### PPO Model
- **Total Return:** -0.0222 (-2.22%)
- **Sharpe Ratio:** -0.1366
- **Max Drawdown:** 0.5112 (51.12%)
- **Trades:** 26
- **Win Rate:** 0.0769 (7.69%)

### SMA Baseline
- **Total Return:** -0.0093 (-0.93%)
- **Sharpe Ratio:** -0.0013
- **Max Drawdown:** 1.0021 (100.21%)
- **Trades:** 7
- **Win Rate:** 0.1429 (14.29%)

## Conclusion

The **SMA** strategy performed better overall based on total return and Sharpe ratio.
