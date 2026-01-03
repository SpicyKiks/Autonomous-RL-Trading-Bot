from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from autonomous_rl_trading_bot.common.exceptions import ConfigError
from autonomous_rl_trading_bot.common.types import MarketType
from autonomous_rl_trading_bot.modes.schemas import ModeRuntimeConfig


MarketDataFactory = Callable[[ModeRuntimeConfig, Mapping[str, Any]], Any]
BrokerFactory = Callable[[ModeRuntimeConfig, Mapping[str, Any]], Any]
RiskFactory = Callable[[ModeRuntimeConfig, Mapping[str, Any]], Any]
EnvFactory = Callable[[ModeRuntimeConfig, Mapping[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class ModeDefinition:
    """A mode is a self-contained configuration + set of factories."""
    mode_id: str
    market_type: MarketType
    description: str

    validate_config: Callable[[ModeRuntimeConfig, Mapping[str, Any]], None]

    create_market_data_client: MarketDataFactory
    create_broker: BrokerFactory
    create_risk_manager: RiskFactory
    create_env: EnvFactory


def _validate_common(runtime: ModeRuntimeConfig, cfg: Mapping[str, Any]) -> None:
    if runtime.exchange_name not in ("binance",):
        raise ConfigError(f"Unsupported exchange.name={runtime.exchange_name!r} (Step-2 supports binance only)")
    if runtime.mode_id not in ("spot", "futures"):
        raise ConfigError(f"Unknown mode.id={runtime.mode_id!r}")
    if not isinstance(cfg, Mapping):
        raise ConfigError("Root config must be a mapping (dict).")


def _validate_spot(runtime: ModeRuntimeConfig, cfg: Mapping[str, Any]) -> None:
    _validate_common(runtime, cfg)


def _validate_futures(runtime: ModeRuntimeConfig, cfg: Mapping[str, Any]) -> None:
    _validate_common(runtime, cfg)


def _make_binance_spot_client(runtime: ModeRuntimeConfig, cfg: Mapping[str, Any]):
    from autonomous_rl_trading_bot.data.binance_spot import BinanceSpotClient
    return BinanceSpotClient()


def _make_binance_futures_client(runtime: ModeRuntimeConfig, cfg: Mapping[str, Any]):
    from autonomous_rl_trading_bot.data.binance_futures import BinanceFuturesClient
    return BinanceFuturesClient()


def _spot_broker(runtime: ModeRuntimeConfig, cfg: Mapping[str, Any]):
    raise NotImplementedError("Spot broker adapter not implemented yet. This will be added in Step 3/8.")


def _futures_broker(runtime: ModeRuntimeConfig, cfg: Mapping[str, Any]):
    raise NotImplementedError("Futures broker adapter not implemented yet. This will be added in Step 3/8.")


def _spot_risk(runtime: ModeRuntimeConfig, cfg: Mapping[str, Any]):
    raise NotImplementedError("Spot risk manager not implemented yet. This will be added in Step 3.")


def _futures_risk(runtime: ModeRuntimeConfig, cfg: Mapping[str, Any]):
    raise NotImplementedError("Futures risk manager not implemented yet. This will be added in Step 4.")


def _spot_env(runtime: ModeRuntimeConfig, cfg: Mapping[str, Any]):
    raise NotImplementedError("SpotEnv not implemented yet. This will be added in Step 3.")


def _futures_env(runtime: ModeRuntimeConfig, cfg: Mapping[str, Any]):
    raise NotImplementedError("FuturesEnv not implemented yet. This will be added in Step 4.")


def builtin_modes() -> list[ModeDefinition]:
    return [
        ModeDefinition(
            mode_id="spot",
            market_type="spot",
            description="Spot trading mode (cash account).",
            validate_config=_validate_spot,
            create_market_data_client=_make_binance_spot_client,
            create_broker=_spot_broker,
            create_risk_manager=_spot_risk,
            create_env=_spot_env,
        ),
        ModeDefinition(
            mode_id="futures",
            market_type="futures",
            description="USD-M futures mode (margin + leverage).",
            validate_config=_validate_futures,
            create_market_data_client=_make_binance_futures_client,
            create_broker=_futures_broker,
            create_risk_manager=_futures_risk,
            create_env=_futures_env,
        ),
    ]

