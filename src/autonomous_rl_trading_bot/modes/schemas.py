from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from autonomous_rl_trading_bot.common.exceptions import ConfigError


def _get(m: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return m.get(key, default) if isinstance(m, Mapping) else default


def _require_map(obj: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise ConfigError(f"Config '{path}' must be a mapping (dict).")
    return obj


def _require_str(obj: Any, path: str) -> str:
    if not isinstance(obj, str) or not obj.strip():
        raise ConfigError(f"Config '{path}' must be a non-empty string.")
    return obj.strip()


def _require_int(obj: Any, path: str) -> int:
    if not isinstance(obj, int):
        raise ConfigError(f"Config '{path}' must be an int.")
    return int(obj)


def _require_bool(obj: Any, path: str) -> bool:
    if not isinstance(obj, bool):
        raise ConfigError(f"Config '{path}' must be a bool.")
    return bool(obj)


@dataclass(frozen=True, slots=True)
class ModeRuntimeConfig:
    """Small, validated subset of the root YAML config used by mode factories."""
    mode_id: str
    seed: int
    timezone: str

    exchange_name: str
    exchange_demo: bool

    default_symbol: str
    default_interval: str
    default_limit: int

    db_path: Optional[str] = None


def validate_root_config(cfg: Mapping[str, Any]) -> ModeRuntimeConfig:
    cfg = _require_map(cfg, "<root>")

    run = _require_map(_get(cfg, "run", {}), "run")
    seed = _require_int(_get(run, "seed", None), "run.seed")
    timezone = _require_str(_get(run, "timezone", "UTC"), "run.timezone")

    mode = _require_map(_get(cfg, "mode", {}), "mode")
    mode_id = _require_str(_get(mode, "id", None), "mode.id").lower()

    exch = _require_map(_get(cfg, "exchange", {}), "exchange")
    exchange_name = _require_str(_get(exch, "name", "binance"), "exchange.name").lower()
    exchange_demo = _require_bool(_get(exch, "demo", True), "exchange.demo")

    defaults = _require_map(_get(exch, "defaults", {}), "exchange.defaults")
    default_symbol = _require_str(_get(defaults, "symbol", "BTCUSDT"), "exchange.defaults.symbol")
    default_interval = _require_str(_get(defaults, "interval", "1m"), "exchange.defaults.interval")
    default_limit = _require_int(_get(defaults, "limit", 1000), "exchange.defaults.limit")

    db = _require_map(_get(cfg, "db", {}), "db")
    db_path = _get(db, "path", None)
    if db_path is not None and not isinstance(db_path, str):
        raise ConfigError("Config 'db.path' must be a string if provided.")

    return ModeRuntimeConfig(
        mode_id=mode_id,
        seed=seed,
        timezone=timezone,
        exchange_name=exchange_name,
        exchange_demo=exchange_demo,
        default_symbol=default_symbol,
        default_interval=default_interval,
        default_limit=default_limit,
        db_path=(db_path.strip() if isinstance(db_path, str) and db_path.strip() else None),
    )

