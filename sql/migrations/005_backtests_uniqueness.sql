PRAGMA foreign_keys = ON;

BEGIN;

-- If you accidentally inserted duplicates before constraints existed,
-- keep the most recently inserted row (largest rowid) per backtest_id.
DELETE FROM backtests
WHERE rowid NOT IN (
  SELECT MAX(rowid) FROM backtests GROUP BY backtest_id
);

-- Also enforce uniqueness on run_id the same way (defensive).
DELETE FROM backtests
WHERE rowid NOT IN (
  SELECT MAX(rowid) FROM backtests GROUP BY run_id
);

-- Enforce uniqueness going forward (idempotent by index name).
CREATE UNIQUE INDEX IF NOT EXISTS ux_backtests_backtest_id ON backtests(backtest_id);
CREATE UNIQUE INDEX IF NOT EXISTS ux_backtests_run_id ON backtests(run_id);

COMMIT;

