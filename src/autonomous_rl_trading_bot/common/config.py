from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

from .exceptions import ConfigError
from .hashing import sha256_of_obj
from .paths import configs_dir

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _expand_env_vars(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    if isinstance(obj, str):

        def repl(m: re.Match) -> str:
            key = m.group(1)
            return os.getenv(key, "")

        return _ENV_PATTERN.sub(repl, obj)
    return obj


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        raise ConfigError(f"Failed to parse YAML: {path} :: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"Top-level YAML must be a mapping (dict): {path}")
    return data


@dataclass(frozen=True)
class LoadedConfig:
    config: Dict[str, Any]
    config_hash: str


def load_config(mode: str | None = None, base_name: str = "base.yaml") -> LoadedConfig:
    """
    Loads configs/base.yaml and optionally configs/modes/<mode>.yaml, deep-merges, expands ${ENV_VAR}.
    Also loads .env if present (for local dev) via python-dotenv.
    """
    load_dotenv(override=False)

    base_path = configs_dir() / base_name
    base_cfg = _load_yaml(base_path)

    resolved_mode = mode or (base_cfg.get("mode", {}) or {}).get("id", None)
    merged = base_cfg

    if resolved_mode:
        mode_path = configs_dir() / "modes" / f"{resolved_mode}.yaml"
        mode_cfg = _load_yaml(mode_path)
        merged = _deep_merge(merged, mode_cfg)

    merged = _expand_env_vars(merged)

    # Basic sanity checks
    if "project" not in merged or "run" not in merged:
        raise ConfigError("Config must contain 'project' and 'run' sections.")
    if not isinstance((merged.get("run", {}) or {}).get("seed", None), int):
        raise ConfigError("Config 'run.seed' must be an integer.")
    if "mode" not in merged or not (merged.get("mode", {}) or {}).get("id"):
        raise ConfigError("Config must define 'mode.id' (e.g., spot).")

    h = sha256_of_obj(merged)
    return LoadedConfig(config=merged, config_hash=h)
