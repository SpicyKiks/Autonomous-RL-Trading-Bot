from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    """UTC now as ISO-8601 with 'Z'."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_now_compact() -> str:
    """UTC now as compact timestamp e.g. 20260111T160633Z."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
