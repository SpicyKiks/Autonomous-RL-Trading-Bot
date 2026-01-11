from __future__ import annotations

import os
from typing import Iterable, Optional

from autonomous_rl_trading_bot.broker.base_broker import BrokerAdapter
from autonomous_rl_trading_bot.common.types import AccountSnapshot, Fill, OrderAck, OrderRequest, Position
from autonomous_rl_trading_bot.data.exchange_client import to_ccxt_symbol


def _truthy(name: str) -> bool:
    v = (os.getenv(name, "") or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


class SpotBroker(BrokerAdapter):
    """
    CCXT-based spot broker adapter (Binance Spot Testnet supported).

    Safety:
      - Network calls are hard-gated by ALLOW_NETWORK=1 (or allow_network_env).
      - Testnet can be enabled by demo=True or USE_TESTNET=1.
    """

    def __init__(
        self,
        *,
        exchange_id: str = "binance",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        demo: Optional[bool] = None,
        base_url_demo: Optional[str] = None,
        allow_network_env: str = "ALLOW_NETWORK",
    ) -> None:
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.demo = bool(_truthy("USE_TESTNET") if demo is None else demo)
        self.base_url_demo = base_url_demo
        self.allow_network_env = allow_network_env

        self._exchange = None
        self._allow_network = (os.getenv(self.allow_network_env, "") or "").strip() == "1"

    def _require_network(self) -> None:
        if not self._allow_network:
            raise RuntimeError(
                f"Network calls disabled. Set {self.allow_network_env}=1 to enable exchange execution."
            )

    def _get_exchange(self):
        if self._exchange is None:
            self._require_network()
            try:
                import ccxt
            except ImportError as e:
                raise RuntimeError("ccxt is required for exchange execution. Install: pip install ccxt") from e

            try:
                exchange_class = getattr(ccxt, self.exchange_id)
                params = {"enableRateLimit": True}
                if self.api_key:
                    params["apiKey"] = self.api_key
                if self.api_secret:
                    params["secret"] = self.api_secret

                ex = exchange_class(params)

                # Prefer CCXT sandbox mode
                if hasattr(ex, "set_sandbox_mode"):
                    ex.set_sandbox_mode(self.demo)

                # Optional explicit testnet REST base override
                if self.demo and self.base_url_demo:
                    try:
                        ex.urls["api"] = {"public": self.base_url_demo, "private": self.base_url_demo}
                    except Exception:
                        pass

                try:
                    ex.load_markets()
                except Exception:
                    pass

                self._exchange = ex
            except Exception as e:
                raise RuntimeError(f"Failed to initialize CCXT exchange {self.exchange_id}: {e}") from e

        return self._exchange

    def get_account_snapshot(self) -> AccountSnapshot:
        self._require_network()
        ex = self._get_exchange()
        bal = ex.fetch_balance()

        usdt = bal.get("USDT", {}) if isinstance(bal, dict) else {}
        equity = float(usdt.get("total", 0.0) or 0.0)
        available = float(usdt.get("free", 0.0) or 0.0)

        return AccountSnapshot(
            ts_ms=int(ex.milliseconds()),
            equity=equity,
            available_cash=available,
            unrealized_pnl=0.0,
            used_margin=0.0,
            leverage=0.0,
        )

    def get_open_positions(self, *, symbol: Optional[str] = None) -> list[Position]:
        self._require_network()
        ex = self._get_exchange()
        bal = ex.fetch_balance()

        base_asset: Optional[str] = None
        ccxt_sym: Optional[str] = None
        if symbol:
            ccxt_sym = to_ccxt_symbol(symbol)
            base_asset = ccxt_sym.split("/")[0]

        positions: list[Position] = []
        for asset, info in (bal or {}).items():
            if asset in ("info", "free", "used", "total"):
                continue
            if base_asset is not None and asset != base_asset:
                continue
            if not isinstance(info, dict):
                continue

            qty = float(info.get("total", info.get("free", 0.0)) or 0.0)
            if abs(qty) <= 1e-12:
                continue

            positions.append(
                Position(
                    symbol=(ccxt_sym if ccxt_sym else f"{asset}/USDT"),
                    side="buy",
                    qty=float(qty),
                    entry_price=0.0,
                )
            )

        return positions

    def submit_order(self, order: OrderRequest) -> OrderAck:
        self._require_network()
        ex = self._get_exchange()

        symbol_ccxt = to_ccxt_symbol(order.symbol)
        side_str = "buy" if order.side == "buy" else "sell"
        amount = float(order.qty)

        if order.qty_unit == "quote":
            ticker = ex.fetch_ticker(symbol_ccxt)
            last = float(ticker.get("last") or 0.0)
            if last <= 0:
                return OrderAck(order_id="", status="rejected", reason="no_price_for_quote_conversion")
            amount = amount / last

        try:
            result = ex.create_market_order(symbol_ccxt, side_str, amount)
            return OrderAck(order_id=str(result.get("id", "")), status="filled")
        except Exception as e:
            return OrderAck(order_id="", status="rejected", reason=str(e))

    def cancel_order(self, order_id: str) -> bool:
        self._require_network()
        return False

    def iter_fills(self, *, since_ts_ms: Optional[int] = None) -> Iterable[Fill]:
        self._require_network()
        ex = self._get_exchange()

        try:
            trades = ex.fetch_my_trades(since=since_ts_ms) if since_ts_ms else ex.fetch_my_trades()
        except TypeError:
            trades = ex.fetch_my_trades("BTC/USDT", since=since_ts_ms) if since_ts_ms else ex.fetch_my_trades("BTC/USDT")

        for t in trades:
            fee = t.get("fee") or {}
            yield Fill(
                order_id=str(t.get("order", "")),
                ts_ms=int(t.get("timestamp", 0) or 0),
                price=float(t.get("price", 0.0) or 0.0),
                qty=float(t.get("amount", 0.0) or 0.0),
                fee_paid=float(fee.get("cost", 0.0) or 0.0),
                fee_asset=str(fee.get("currency", "USDT")),
            )

    def close(self) -> None:
        if self._exchange is not None:
            try:
                self._exchange.close()
            except Exception:
                pass
