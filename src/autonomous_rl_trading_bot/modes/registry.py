from __future__ import annotations

from typing import Any, Dict, Mapping

from autonomous_rl_trading_bot.common.exceptions import ConfigError
from autonomous_rl_trading_bot.modes.mode_defs import ModeDefinition, builtin_modes
from autonomous_rl_trading_bot.modes.schemas import ModeRuntimeConfig, validate_root_config


class ModeRegistry:
    """Registry of mode definitions.

    Everything must resolve mode through this registry (CLI/dashboard/training/live).
    """

    def __init__(self) -> None:
        self._modes: Dict[str, ModeDefinition] = {}

    def register(self, mode: ModeDefinition) -> None:
        mid = (mode.mode_id or "").strip().lower()
        if not mid:
            raise ConfigError("ModeDefinition.mode_id must be a non-empty string")
        if mid in self._modes:
            raise ConfigError(f"Mode already registered: {mid}")
        self._modes[mid] = mode

    def get(self, mode_id: str) -> ModeDefinition:
        mid = (mode_id or "").strip().lower()
        if mid not in self._modes:
            raise ConfigError(f"Unknown mode: {mode_id!r}. Available: {sorted(self._modes)}")
        return self._modes[mid]

    def list_modes(self) -> list[str]:
        return sorted(self._modes.keys())

    def resolve(self, cfg: Mapping[str, Any]) -> tuple[ModeDefinition, ModeRuntimeConfig]:
        runtime = validate_root_config(cfg)
        mode = self.get(runtime.mode_id)
        mode.validate_config(runtime, cfg)
        return mode, runtime


_REGISTRY: ModeRegistry | None = None


def get_registry() -> ModeRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        reg = ModeRegistry()
        for m in builtin_modes():
            reg.register(m)
        _REGISTRY = reg
    return _REGISTRY

