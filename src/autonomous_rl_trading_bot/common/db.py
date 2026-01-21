from __future__ import annotations

import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .exceptions import ConfigError
from .paths import artifacts_dir, repo_root

_MIGRATION_RE = re.compile(r"^(?P<ver>\d+)_.*\.sql$")


def _utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def default_db_path(cfg: dict[str, Any]) -> Path:
    """
    If cfg['db']['path'] is set:
      - absolute path is used directly
      - relative path is resolved relative to repo root
    Else defaults to artifacts/db/<filename>.
    """
    db_cfg = cfg.get("db", {}) or {}
    raw_path = str(db_cfg.get("path", "") or "").strip()
    filename = str(db_cfg.get("filename", "bot.db") or "bot.db").strip() or "bot.db"

    if raw_path:
        p = Path(raw_path)
        if not p.is_absolute():
            p = repo_root() / p
        if p.suffix.lower() != ".db":
            # allow specifying a folder path; append filename
            if p.exists() and p.is_dir():
                p = p / filename
            elif str(p).endswith(("/", "\\")):
                p = p / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    p = artifacts_dir() / "db" / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def ensure_schema_migrations(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
          version INTEGER PRIMARY KEY,
          name TEXT NOT NULL,
          applied_utc TEXT NOT NULL
        );
        """
    )


def _list_migration_files(migrations_dir: Path) -> list[tuple[int, Path]]:
    if not migrations_dir.exists():
        raise ConfigError(f"Migrations directory not found: {migrations_dir}")

    items: list[tuple[int, Path]] = []
    for p in migrations_dir.glob("*.sql"):
        m = _MIGRATION_RE.match(p.name)
        if not m:
            raise ConfigError(f"Invalid migration filename (must start with digits_): {p.name}")
        ver = int(m.group("ver"))
        items.append((ver, p))

    items.sort(key=lambda x: (x[0], x[1].name))
    return items


def applied_migrations(conn: sqlite3.Connection) -> dict[int, str]:
    ensure_schema_migrations(conn)
    rows = conn.execute("SELECT version, name FROM schema_migrations ORDER BY version").fetchall()
    return {int(r["version"]): str(r["name"]) for r in rows}


def apply_migrations(conn: sqlite3.Connection, migrations_dir: Path) -> list[str]:
    """
    Applies any pending migrations in-order.
    Returns list of applied migration filenames.
    """
    ensure_schema_migrations(conn)
    already = applied_migrations(conn)
    to_apply = _list_migration_files(migrations_dir)

    applied: list[str] = []
    try:
        conn.execute("BEGIN;")
        for ver, path in to_apply:
            if ver in already:
                continue
            sql = path.read_text(encoding="utf-8")
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_migrations(version, name, applied_utc) VALUES(?, ?, ?);",
                (ver, path.name, _utc_iso()),
            )
            applied.append(path.name)
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise

    return applied


def migrate(cfg: dict[str, Any], migrations_dir: Path | None = None) -> Path:
    """
    Ensures DB exists and all migrations are applied.
    Returns db_path.
    """
    db_path = default_db_path(cfg)
    mig_dir = migrations_dir or (repo_root() / "sql" / "migrations")
    with connect(db_path) as conn:
        apply_migrations(conn, mig_dir)
    return db_path


def insert_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    kind: str,
    mode: str,
    created_utc: str,
    config_hash: str,
    seed: int,
    status: str,
    run_dir: str,
    run_json_path: str,
    run_log_path: str | None,
    global_log_path: str | None,
) -> None:
    """
    Insert a run row. If run_id already exists, update it.
    """
    conn.execute(
        """
        INSERT INTO runs(
          run_id, kind, mode, created_utc, config_hash, seed, status,
          run_dir, run_json_path, run_log_path, global_log_path
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            run_id,
            kind,
            mode,
            created_utc,
            config_hash,
            seed,
            status,
            run_dir,
            run_json_path,
            run_log_path,
            global_log_path,
        ),
    )


def upsert_run(conn: sqlite3.Connection, **kwargs: Any) -> None:
    """
    Upsert wrapper for insert_run: tries insert, falls back to update on conflict.
    """
    try:
        insert_run(conn, **kwargs)
    except sqlite3.IntegrityError:
        conn.execute(
            """
            UPDATE runs SET
              kind=?,
              mode=?,
              created_utc=?,
              config_hash=?,
              seed=?,
              status=?,
              run_dir=?,
              run_json_path=?,
              run_log_path=?,
              global_log_path=?
            WHERE run_id=?;
            """,
            (
                kwargs["kind"],
                kwargs["mode"],
                kwargs["created_utc"],
                kwargs["config_hash"],
                kwargs["seed"],
                kwargs["status"],
                kwargs["run_dir"],
                kwargs["run_json_path"],
                kwargs.get("run_log_path"),
                kwargs.get("global_log_path"),
                kwargs["run_id"],
            ),
        )
