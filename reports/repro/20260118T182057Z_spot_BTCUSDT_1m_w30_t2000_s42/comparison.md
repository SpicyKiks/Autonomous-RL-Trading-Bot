# Backtest Comparison Report

**Run ID:** `20260118T182057Z_spot_BTCUSDT_1m_w30_t2000_s42`  
**Symbol:** BTCUSDT  
**Interval:** 1m  
**Training Timesteps:** 2,000  
**Seed:** 42  
**Generated:** 2026-01-18T18:21:07.865672+00:00

## Summary

**Winner:** SMA

PPO vs SMA:
- Return difference: -0.0197 (-1.97%)
- Sharpe difference: -0.1724

## Metrics Comparison

| Metric | PPO | SMA | Winner |
|--------|-----|-----|--------|
| **Total Return** | -0.0288 (-2.88%) | -0.0091 (-0.91%) | SMA |
| **Sharpe Ratio** | -0.1737 | -0.0013 | SMA |
| **Sortino Ratio** | -0.1426 | -0.0008 | SMA |
| **Max Drawdown** | 0.5144 (51.44%) | 1.0019 (100.19%) | PPO |
| **Calmar Ratio** | -0.0561 | -0.0091 | SMA |
| **Num Trades** | 29 | 7 | - |
| **Win Rate** | 0.0345 (3.45%) | 0.1429 (14.29%) | SMA |
| **Avg Trade PnL** | -8.9599 | -13.0225 | PPO |
| **Profit Factor** | 0.0574 | 0.0010 | PPO |
| **Final Equity** | 9711.53 | 9908.83 | SMA |

## Detailed Results

### PPO Model
- **Total Return:** -0.0288 (-2.88%)
- **Sharpe Ratio:** -0.1737
- **Max Drawdown:** 0.5144 (51.44%)
- **Trades:** 29
- **Win Rate:** 0.0345 (3.45%)

### SMA Baseline
- **Total Return:** -0.0091 (-0.91%)
- **Sharpe Ratio:** -0.0013
- **Max Drawdown:** 1.0019 (100.19%)
- **Trades:** 7
- **Win Rate:** 0.1429 (14.29%)

## Conclusion

The **SMA** strategy performed better overall based on total return and Sharpe ratio.
