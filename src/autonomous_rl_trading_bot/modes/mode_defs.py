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
    from autonomous_rl_trading_bot.broker.paper import PaperSpotBroker, PaperSpotConfig

    bt = (cfg.get("evaluation", {}) or {}).get("backtest", {}) or {}
    initial_cash = float(bt.get("initial_cash", 1000.0))
    taker_fee_rate = float(bt.get("taker_fee_rate", 0.001))
    slippage_bps = float(bt.get("slippage_bps", 5.0))

    return PaperSpotBroker(
        symbol=runtime.default_symbol,
        cfg=PaperSpotConfig(
            initial_cash=initial_cash,
            taker_fee_rate=taker_fee_rate,
            slippage_bps=slippage_bps,
        ),
    )


def _futures_broker(runtime: ModeRuntimeConfig, cfg: Mapping[str, Any]):
    raise NotImplementedError("Futures broker adapter not implemented yet. This will be added in Step 3/8.")


def _spot_risk(runtime: ModeRuntimeConfig, cfg: Mapping[str, Any]):
    from autonomous_rl_trading_bot.risk import SpotRiskConfig, SpotRiskManager

    live = (cfg.get("live", {}) or {})
    r = (live.get("risk", {}) or {})
    max_drawdown = float(r.get("max_drawdown", 0.3))
    min_equity = float(r.get("min_equity", 1e-9))
    max_order_quote = float(r.get("max_order_quote", 0.0))
    max_exposure = float(r.get("max_exposure", 1.0))

    return SpotRiskManager(
        SpotRiskConfig(
            max_drawdown=max_drawdown,
            min_equity=min_equity,
            max_order_quote=max_order_quote,
            max_exposure=max_exposure,
        )
    )


def _futures_risk(runtime: ModeRuntimeConfig, cfg: Mapping[str, Any]):
    raise NotImplementedError("Futures risk manager not implemented yet. This will be added in Step 4.")


def _spot_env(runtime: ModeRuntimeConfig, cfg: Mapping[str, Any]):
    from autonomous_rl_trading_bot.common.paths import artifacts_dir
    from autonomous_rl_trading_bot.envs import SpotEnv, SpotEnvConfig
    from autonomous_rl_trading_bot.rl.dataset import load_dataset_npz, select_latest_dataset

    artifacts_base = artifacts_dir()
    datasets_dir = artifacts_base / "datasets"

    env_cfg = (cfg.get("env", {}) or {}).get("spot", {}) or {}
    lookback = int(env_cfg.get("lookback", 64))
    reward_type = str(env_cfg.get("reward_type", "log_return"))

    bt = (cfg.get("evaluation", {}) or {}).get("backtest", {}) or {}
    initial_cash = float(bt.get("initial_cash", 1000.0))
    order_size_quote = float(bt.get("order_size_quote", 0.0))
    taker_fee_rate = float(bt.get("taker_fee_rate", 0.001))
    slippage_bps = float(bt.get("slippage_bps", 5.0))

    max_drawdown = float(env_cfg.get("max_drawdown", 0.5))
    min_equity = float(env_cfg.get("min_equity", 1e-9))

    dataset_id = env_cfg.get("dataset_id", None)
    if isinstance(dataset_id, str) and dataset_id.strip():
        ds_dir = datasets_dir / dataset_id.strip()
        ds = load_dataset_npz(ds_dir)
    else:
        ds = select_latest_dataset(datasets_dir, "spot")

    feature_keys = None
    data_cfg = (cfg.get("data", {}) or {}).get("dataset", {}) or {}
    feats = data_cfg.get("features", None)
    if isinstance(feats, list) and feats:
        feature_keys = [str(x) for x in feats]

    scfg = SpotEnvConfig(
        lookback=lookback,
        initial_cash=initial_cash,
        order_size_quote=order_size_quote,
        taker_fee_rate=taker_fee_rate,
        slippage_bps=slippage_bps,
        max_drawdown=max_drawdown,
        min_equity=min_equity,
        reward_type=reward_type,
    )

    return SpotEnv.from_dataset_dir(ds.dataset_dir, cfg=scfg, seed=runtime.seed, feature_keys=feature_keys)


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

