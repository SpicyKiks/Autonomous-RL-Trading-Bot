from __future__ import annotations

import hashlib
import json
from typing import Any


def _stable_json(obj: Any) -> str:
    """
    Deterministic JSON representation: sorted keys, no whitespace noise.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_of_obj(obj: Any) -> str:
    payload = _stable_json(obj).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def short_hash(hex_hash: str, n: int = 10) -> str:
    return hex_hash[:n]
