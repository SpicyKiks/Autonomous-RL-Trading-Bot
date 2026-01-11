-- Reference schema (consolidated from migrations)
-- NOTE: Migrations remain the runtime source of truth.
-- This file is for examiner readability only.

PRAGMA foreign_keys = ON;

-- Schema migrations tracking
CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  applied_utc TEXT NOT NULL
);

-- Runs table (all run types: backtest, train, live)
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  mode TEXT NOT NULL,
  created_utc TEXT NOT NULL,
  config_hash TEXT NOT NULL,
  seed INTEGER NOT NULL,
  status TEXT NOT NULL,
  run_dir TEXT NOT NULL,
  run_json_path TEXT NOT NULL,
  run_log_path TEXT,
  global_log_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_created_utc ON runs(created_utc);
CREATE INDEX IF NOT EXISTS idx_runs_mode ON runs(mode);

-- Candles storage
CREATE TABLE IF NOT EXISTS candles (
  exchange TEXT NOT NULL,
  market_type TEXT NOT NULL,
  symbol TEXT NOT NULL,
  interval TEXT NOT NULL,
  open_time_ms INTEGER NOT NULL,
  open REAL NOT NULL,
  high REAL NOT NULL,
  low REAL NOT NULL,
  close REAL NOT NULL,
  volume REAL NOT NULL,
  close_time_ms INTEGER NOT NULL,
  quote_asset_volume REAL,
  number_of_trades INTEGER,
  taker_buy_base_asset_volume REAL,
  taker_buy_quote_asset_volume REAL,
  ignore REAL,
  PRIMARY KEY (exchange, market_type, symbol, interval, open_time_ms)
);

CREATE INDEX IF NOT EXISTS idx_candles_symbol_interval_time
  ON candles(symbol, interval, open_time_ms);
CREATE INDEX IF NOT EXISTS idx_candles_market_time
  ON candles(market_type, open_time_ms);

-- Backtests (spot + futures)
CREATE TABLE IF NOT EXISTS backtests (
  backtest_id TEXT PRIMARY KEY,
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

CREATE UNIQUE INDEX IF NOT EXISTS ux_backtests_backtest_id ON backtests(backtest_id);
CREATE UNIQUE INDEX IF NOT EXISTS ux_backtests_run_id ON backtests(run_id);
CREATE INDEX IF NOT EXISTS idx_backtests_mode_started ON backtests(mode, started_utc);
CREATE INDEX IF NOT EXISTS idx_backtests_symbol_interval ON backtests(symbol, interval);

-- Spot backtest equity
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

-- Spot backtest trades
CREATE TABLE IF NOT EXISTS backtest_trades (
  trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
  backtest_id TEXT NOT NULL,
  open_time_ms INTEGER NOT NULL,
  side TEXT NOT NULL,
  qty_base REAL NOT NULL,
  price REAL NOT NULL,
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

-- Futures backtest equity
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

-- Futures backtest trades
CREATE TABLE IF NOT EXISTS backtest_futures_trades (
  trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
  backtest_id TEXT NOT NULL,
  open_time_ms INTEGER NOT NULL,
  side TEXT NOT NULL,
  qty REAL NOT NULL,
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

-- Training jobs
CREATE TABLE IF NOT EXISTS train_jobs (
  train_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL UNIQUE,
  mode TEXT NOT NULL,
  market_type TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  algo TEXT NOT NULL,
  total_timesteps INTEGER NOT NULL,
  seed INTEGER NOT NULL,
  started_utc TEXT NOT NULL,
  finished_utc TEXT,
  status TEXT NOT NULL,
  params_json TEXT NOT NULL,
  metrics_json TEXT,
  model_path TEXT,
  FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_train_jobs_started ON train_jobs(started_utc);
CREATE INDEX IF NOT EXISTS idx_train_jobs_dataset ON train_jobs(dataset_id);

-- Live trading sessions
CREATE TABLE IF NOT EXISTS live_sessions (
  live_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  mode TEXT NOT NULL,
  market_type TEXT NOT NULL,
  symbol TEXT NOT NULL,
  interval TEXT NOT NULL,
  started_utc TEXT NOT NULL,
  finished_utc TEXT,
  status TEXT NOT NULL,
  initial_equity REAL NOT NULL,
  final_equity REAL,
  fee_total REAL NOT NULL DEFAULT 0,
  slippage_total REAL NOT NULL DEFAULT 0,
  params_json TEXT,
  metrics_json TEXT
);

-- Live equity snapshots
CREATE TABLE IF NOT EXISTS live_equity (
  live_id TEXT NOT NULL,
  open_time_ms INTEGER NOT NULL,
  price REAL NOT NULL,
  cash REAL NOT NULL,
  position_qty REAL NOT NULL,
  entry_price REAL NOT NULL,
  unrealized_pnl REAL NOT NULL,
  equity REAL NOT NULL,
  drawdown REAL NOT NULL,
  notional REAL NOT NULL,
  margin_used REAL NOT NULL,
  leverage_used REAL NOT NULL,
  exposure REAL NOT NULL,
  PRIMARY KEY (live_id, open_time_ms),
  FOREIGN KEY (live_id) REFERENCES live_sessions(live_id)
);

CREATE INDEX IF NOT EXISTS idx_live_equity_live_id ON live_equity(live_id);

-- Live trades
CREATE TABLE IF NOT EXISTS live_trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  live_id TEXT NOT NULL,
  open_time_ms INTEGER NOT NULL,
  side TEXT NOT NULL,
  qty REAL NOT NULL,
  fill_price REAL NOT NULL,
  notional REAL NOT NULL,
  fee REAL NOT NULL,
  slippage_cost REAL NOT NULL,
  realized_pnl REAL NOT NULL,
  reason TEXT,
  cash_after REAL NOT NULL,
  position_qty_after REAL NOT NULL,
  entry_price_after REAL NOT NULL,
  equity_after REAL NOT NULL,
  FOREIGN KEY (live_id) REFERENCES live_sessions(live_id)
);

CREATE INDEX IF NOT EXISTS idx_live_trades_live_id ON live_trades(live_id);
CREATE INDEX IF NOT EXISTS idx_live_trades_time ON live_trades(open_time_ms);

