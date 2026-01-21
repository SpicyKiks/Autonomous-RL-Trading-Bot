from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)


def _env_subset(prefixes: tuple[str, ...] = ("ARBT_", "PYTHON", "CUDA", "CUBLAS", "CUDNN")) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in os.environ.items():
        if k.startswith(prefixes):
            out[k] = v
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _pkg_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    # standard library metadata (py3.11)
    try:
        from importlib.metadata import version as _version
    except Exception:  # pragma: no cover
        return versions

    for name in [
        "autonomous-rl-trading-bot",
        "numpy",
        "pandas",
        "pyarrow",
        "gymnasium",
        "stable-baselines3",
        "torch",
        "matplotlib",
        "plotly",
        "dash",
        "ccxt",
        "scikit-learn",
    ]:
        try:
            versions[name] = _version(name)
        except Exception:
            pass
    return versions


def _torch_info() -> dict[str, Any]:
    if torch is None:
        return {"available": False}

    info: dict[str, Any] = {"available": True}
    try:
        info["version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["cuda_device_count"] = int(torch.cuda.device_count())
            info["cuda_device_name_0"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def _numpy_info() -> dict[str, Any]:
    if np is None:
        return {"available": False}
    return {"available": True, "version": getattr(np, "__version__", None)}


def _pandas_info() -> dict[str, Any]:
    if pd is None:
        return {"available": False}
    return {"available": True, "version": getattr(pd, "__version__", None)}


def _asdict_if_needed(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def build_repro_payload(config: Any | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Build a reproducibility payload saved alongside artifacts.

    The trainer imports this as:
      from autonomous_rl_trading_bot.repro.repro import build_repro_payload

    Keep it dependency-light and never crash training.
    """
    payload: dict[str, Any] = {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cwd": _safe_str(Path.cwd()),
        "env": _env_subset(),
        "packages": _pkg_versions(),
        "numpy": _numpy_info(),
        "pandas": _pandas_info(),
        "torch": _torch_info(),
    }

    if config is not None:
        payload["config"] = _asdict_if_needed(config)

    if extra:
        payload["extra"] = extra

    return payload
