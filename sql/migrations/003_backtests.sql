PRAGMA foreign_keys = ON;

-- Step 5: Deterministic offline backtesting storage (v1)
--
-- We keep tables minimal but normalized:
-- - backtests: one row per backtest run (backtest_id == run_id)
-- - backtest_equity: per-step equity/account snapshots
-- - backtest_trades: executed trades ledger

CREATE TABLE IF NOT EXISTS backtests (
  backtest_id TEXT PRIMARY KEY,        -- use run_id as backtest_id for traceability
  run_id TEXT NOT NULL UNIQUE,
  mode TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  market_type TEXT NOT NULL,
  symbol TEXT NOT NULL,
  interval TEXT NOT NULL,
  started_utc TEXT NOT NULL,
  finished_utc TEXT,
  status TEXT NOT NULL,
  initial_cash REAL NOT NULL,
  final_equity REAL,
  total_return REAL,
  max_drawdown REAL,
  trade_count INTEGER,
  fee_total REAL,
  slippage_total REAL,
  params_json TEXT NOT NULL,
  metrics_json TEXT,
  FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_backtests_mode_started ON backtests(mode, started_utc);
CREATE INDEX IF NOT EXISTS idx_backtests_symbol_interval ON backtests(symbol, interval);

CREATE TABLE IF NOT EXISTS backtest_equity (
  backtest_id TEXT NOT NULL,
  open_time_ms INTEGER NOT NULL,
  price REAL NOT NULL,
  cash REAL NOT NULL,
  qty_base REAL NOT NULL,
  equity REAL NOT NULL,
  drawdown REAL NOT NULL,
  exposure REAL NOT NULL,
  PRIMARY KEY (backtest_id, open_time_ms),
  FOREIGN KEY(backtest_id) REFERENCES backtests(backtest_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_backtest_equity_time ON backtest_equity(backtest_id, open_time_ms);

CREATE TABLE IF NOT EXISTS backtest_trades (
  trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
  backtest_id TEXT NOT NULL,
  open_time_ms INTEGER NOT NULL,
  side TEXT NOT NULL,                 -- BUY | SELL
  qty_base REAL NOT NULL,
  price REAL NOT NULL,                -- fill price
  notional REAL NOT NULL,
  fee REAL NOT NULL,
  slippage_cost REAL NOT NULL,
  reason TEXT,
  cash_after REAL NOT NULL,
  qty_after REAL NOT NULL,
  equity_after REAL NOT NULL,
  FOREIGN KEY(backtest_id) REFERENCES backtests(backtest_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_time ON backtest_trades(backtest_id, open_time_ms);

