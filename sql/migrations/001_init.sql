PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  applied_utc TEXT NOT NULL
);

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

