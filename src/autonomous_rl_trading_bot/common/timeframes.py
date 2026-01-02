from __future__ import annotations

_INTERVAL_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def interval_to_ms(interval: str) -> int:
    i = (interval or "").strip()
    if i not in _INTERVAL_MS:
        raise ValueError(
            f"Unsupported interval: {interval!r}. Supported: {sorted(_INTERVAL_MS.keys())}"
        )
    return _INTERVAL_MS[i]
