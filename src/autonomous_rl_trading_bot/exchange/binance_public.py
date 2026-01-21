from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

from autonomous_rl_trading_bot.common.http import get_json
from autonomous_rl_trading_bot.common.timeframes import interval_to_ms


@dataclass(frozen=True)
class Candle:
    exchange: str
    market_type: str
    symbol: str
    interval: str
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time_ms: int
    quote_asset_volume: float | None
    number_of_trades: int | None
    taker_buy_base_asset_volume: float | None
    taker_buy_quote_asset_volume: float | None
    ignore: float | None


def _base_url(cfg: dict[str, Any], market_type: str) -> str:
    exch = cfg.get("exchange", {}) or {}
    demo = bool(exch.get("demo", True))
    market_type = market_type.strip().lower()
    if market_type == "spot":
        spot = exch.get("spot", {}) or {}
        return str(spot["base_url_demo" if demo else "base_url_live"]).rstrip("/")
    if market_type == "futures":
        fut = exch.get("futures", {}) or {}
        return str(fut["base_url_demo" if demo else "base_url_live"]).rstrip("/")
    raise ValueError(f"Unknown market_type: {market_type!r}")


def _path_for(market_type: str) -> str:
    mt = market_type.strip().lower()
    if mt == "spot":
        return "/api/v3/klines"
    if mt == "futures":
        return "/fapi/v1/klines"
    raise ValueError(f"Unknown market_type: {market_type!r}")


def fetch_klines(
    cfg: dict[str, Any],
    *,
    market_type: str,
    symbol: str,
    interval: str,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
    limit: int = 1000,
) -> list[Candle]:
    """
    Fetch a single batch of klines (max 1000).
    Returns parsed Candle objects.
    """
    base = _base_url(cfg, market_type)
    path = _path_for(market_type)

    params: dict[str, Any] = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": int(limit),
    }
    if start_time_ms is not None:
        params["startTime"] = int(start_time_ms)
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)

    url = f"{base}{path}?{urlencode(params)}"
    res = get_json(url)
    raw = res.data

    if not isinstance(raw, list):
        raise ValueError(f"Unexpected klines payload type: {type(raw)}")

    out: list[Candle] = []
    for row in raw:
        # Binance kline format: list with 12 items
        # [ openTime, open, high, low, close, volume, closeTime, quoteAssetVolume, numTrades, takerBuyBase, takerBuyQuote, ignore ]
        if not isinstance(row, list) or len(row) < 6:
            continue

        def _f(x: Any) -> float:
            return float(x)

        def _i(x: Any) -> int:
            return int(x)

        open_time = _i(row[0])
        close_time = _i(row[6]) if len(row) > 6 else open_time + interval_to_ms(interval) - 1

        qav = float(row[7]) if len(row) > 7 and row[7] is not None else None
        ntr = int(row[8]) if len(row) > 8 and row[8] is not None else None
        tbb = float(row[9]) if len(row) > 9 and row[9] is not None else None
        tbq = float(row[10]) if len(row) > 10 and row[10] is not None else None
        ign = float(row[11]) if len(row) > 11 and row[11] is not None else None

        out.append(
            Candle(
                exchange="binance",
                market_type=market_type.lower(),
                symbol=symbol.upper(),
                interval=interval,
                open_time_ms=open_time,
                open=_f(row[1]),
                high=_f(row[2]),
                low=_f(row[3]),
                close=_f(row[4]),
                volume=_f(row[5]),
                close_time_ms=close_time,
                quote_asset_volume=qav,
                number_of_trades=ntr,
                taker_buy_base_asset_volume=tbb,
                taker_buy_quote_asset_volume=tbq,
                ignore=ign,
            )
        )

    return out
