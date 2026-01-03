PRAGMA foreign_keys = ON;

-- Step 5 extension: futures backtest storage (spot tables remain unchanged)

CREATE TABLE IF NOT EXISTS backtest_futures_equity (
  backtest_id TEXT NOT NULL,
  open_time_ms INTEGER NOT NULL,
  price REAL NOT NULL,

  collateral REAL NOT NULL,
  position_qty REAL NOT NULL,
  entry_price REAL NOT NULL,

  unrealized_pnl REAL NOT NULL,
  equity REAL NOT NULL,
  drawdown REAL NOT NULL,

  notional REAL NOT NULL,
  margin_used REAL NOT NULL,
  leverage_used REAL NOT NULL,
  exposure REAL NOT NULL,

  PRIMARY KEY (backtest_id, open_time_ms),
  FOREIGN KEY(backtest_id) REFERENCES backtests(backtest_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_fut_equity_time ON backtest_futures_equity(backtest_id, open_time_ms);

CREATE TABLE IF NOT EXISTS backtest_futures_trades (
  trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
  backtest_id TEXT NOT NULL,
  open_time_ms INTEGER NOT NULL,

  side TEXT NOT NULL,               -- BUY | SELL | CLOSE
  qty REAL NOT NULL,                -- signed delta qty applied to position
  fill_price REAL NOT NULL,
  notional REAL NOT NULL,

  fee REAL NOT NULL,
  slippage_cost REAL NOT NULL,
  realized_pnl REAL NOT NULL,

  reason TEXT,

  collateral_after REAL NOT NULL,
  position_qty_after REAL NOT NULL,
  entry_price_after REAL NOT NULL,
  equity_after REAL NOT NULL,

  FOREIGN KEY(backtest_id) REFERENCES backtests(backtest_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_fut_trades_time ON backtest_futures_trades(backtest_id, open_time_ms);

