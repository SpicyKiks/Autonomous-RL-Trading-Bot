PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS candles (
  exchange TEXT NOT NULL,
  market_type TEXT NOT NULL,          -- spot | futures
  symbol TEXT NOT NULL,               -- e.g. BTCUSDT
  interval TEXT NOT NULL,             -- e.g. 1m, 1h
  open_time_ms INTEGER NOT NULL,      -- kline open time (ms)
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

