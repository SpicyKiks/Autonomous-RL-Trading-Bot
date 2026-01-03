PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS train_jobs (
  train_id TEXT PRIMARY KEY,              -- equals run_id for simplicity
  run_id TEXT NOT NULL UNIQUE,
  mode TEXT NOT NULL,                     -- spot | futures (requested)
  market_type TEXT NOT NULL,              -- spot | futures (from dataset meta)
  dataset_id TEXT NOT NULL,
  algo TEXT NOT NULL,                     -- ppo | dqn
  total_timesteps INTEGER NOT NULL,
  seed INTEGER NOT NULL,
  started_utc TEXT NOT NULL,
  finished_utc TEXT,
  status TEXT NOT NULL,                   -- CREATED | RUNNING | DONE | FAILED
  params_json TEXT NOT NULL,
  metrics_json TEXT,
  model_path TEXT,
  FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_train_jobs_started ON train_jobs(started_utc);
CREATE INDEX IF NOT EXISTS idx_train_jobs_dataset ON train_jobs(dataset_id);

