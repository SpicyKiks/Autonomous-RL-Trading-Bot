from __future__ import annotations

from autonomous_rl_trading_bot.common.timeframes import interval_to_ms
from autonomous_rl_trading_bot.exchange.binance_public import Candle


def test_interval_to_ms() -> None:
    assert interval_to_ms("1m") == 60_000
    assert interval_to_ms("1h") == 3_600_000


def test_candle_parsing_shape_without_network() -> None:
    # We can't call fetch_klines in tests (network), so validate Candle fields directly.
    c = Candle(
        exchange="binance",
        market_type="spot",
        symbol="BTCUSDT",
        interval="1m",
        open_time_ms=1,
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume=10.0,
        close_time_ms=2,
        quote_asset_volume=None,
        number_of_trades=None,
        taker_buy_base_asset_volume=None,
        taker_buy_quote_asset_volume=None,
        ignore=None,
    )
    assert c.symbol == "BTCUSDT"
    assert c.interval == "1m"
    assert c.open_time_ms == 1
