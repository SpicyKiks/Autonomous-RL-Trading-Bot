import sqlite3
from pathlib import Path

db = Path(r"artifacts\db\bot.db")
print("exists=", db.exists(), db.resolve())

conn = sqlite3.connect(db)
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
print("tables=", [r[0] for r in cur.fetchall()])

cur.execute("PRAGMA table_info('candles')")
print("candles_cols=", [r[1] for r in cur.fetchall()])

conn.close()
