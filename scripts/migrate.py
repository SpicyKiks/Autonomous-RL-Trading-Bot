from __future__ import annotations

import argparse
from pathlib import Path

from autonomous_rl_trading_bot.common.config import load_config
from autonomous_rl_trading_bot.common.db import apply_migrations, connect, migrate
from autonomous_rl_trading_bot.common.paths import repo_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply SQLite migrations.")
    parser.add_argument("--mode", default=None, help="Mode id (spot/futures).")
    parser.add_argument(
        "--db", default=None, help="Override DB path (absolute or relative to repo root)."
    )
    parser.add_argument(
        "--migrations",
        default=None,
        help="Override migrations directory (absolute or relative to repo root).",
    )
    args = parser.parse_args()

    loaded = load_config(mode=args.mode)
    cfg = dict(loaded.config)

    if args.db:
        cfg.setdefault("db", {})
        cfg["db"]["path"] = args.db

    migrations_dir = None
    if args.migrations:
        p = Path(args.migrations)
        if not p.is_absolute():
            p = repo_root() / p
        migrations_dir = p

    # If migrations_dir override is set, apply explicitly (and still ensure db exists)
    if migrations_dir is not None:
        db_path = migrate(cfg)  # creates + ensures schema_migrations table exists via 001 too
        with connect(db_path) as conn:
            applied = apply_migrations(conn, migrations_dir)
        print(f"OK: migrated {db_path} (applied {len(applied)} migrations)")
        return 0

    db_path = migrate(cfg)
    print(f"OK: migrated {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
