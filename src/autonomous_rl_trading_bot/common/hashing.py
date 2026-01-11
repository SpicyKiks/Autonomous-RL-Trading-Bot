from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any


def _to_jsonable(obj: Any) -> Any:
    """
    Convert common Python objects to something JSON-serializable and stable.
    """
    if obj is None:
        return None

    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, bytes):
        # stable textual form
        return obj.hex()

    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]

    if isinstance(obj, dict):
        # sort keys for stability
        return {str(k): _to_jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}

    # common pandas/numpy-ish
    if hasattr(obj, "to_dict"):
        try:
            return _to_jsonable(obj.to_dict())
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        try:
            return _to_jsonable(vars(obj))
        except Exception:
            pass

    # fallback
    return str(obj)


def sha256_of_obj(obj: Any) -> str:
    """
    Stable SHA256 hex digest for arbitrary Python objects by canonical JSON encoding.
    """
    payload = _to_jsonable(obj)
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def short_hash(obj: Any, length: int = 8) -> str:
    """
    Short stable hash of any object.
    """
    if length <= 0:
        raise ValueError("length must be > 0")
    return sha256_of_obj(obj)[:length]


def dataset_hash(dataset: Any, length: int = 10) -> str:
    """
    Stable id for a dataset object / dataset metadata.

    Tries common attributes (id, meta, path, symbol, timeframe, etc.) before
    falling back to hashing the object itself.
    """
    candidates: dict[str, Any] = {}

    for key in ("dataset_id", "id", "name", "symbol", "timeframe", "market_type"):
        if hasattr(dataset, key):
            try:
                candidates[key] = getattr(dataset, key)
            except Exception:
                pass

    for key in ("meta", "metadata", "info", "config"):
        if hasattr(dataset, key):
            try:
                val = getattr(dataset, key)
                if isinstance(val, dict):
                    candidates[key] = val
                else:
                    candidates[key] = str(val)
            except Exception:
                pass

    for key in ("path", "file_path", "parquet_path", "csv_path"):
        if hasattr(dataset, key):
            try:
                candidates[key] = str(getattr(dataset, key))
            except Exception:
                pass

    base = candidates if candidates else dataset
    return sha256_of_obj(base)[:length]
