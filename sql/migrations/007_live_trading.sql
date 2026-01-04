-- Live (paper) trading persistence tables

BEGIN;

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

CREATE INDEX IF NOT EXISTS idx_live_equity_live_id ON live_equity(live_id);
CREATE INDEX IF NOT EXISTS idx_live_trades_live_id ON live_trades(live_id);
CREATE INDEX IF NOT EXISTS idx_live_trades_time ON live_trades(open_time_ms);

COMMIT;

