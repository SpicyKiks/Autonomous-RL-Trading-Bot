from __future__ import annotations

import logging
import os
from collections.abc import Iterable

from autonomous_rl_trading_bot.broker.base_broker import BrokerAdapter
from autonomous_rl_trading_bot.common.types import (
    AccountSnapshot,
    Fill,
    OrderAck,
    OrderRequest,
    Position,
)
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
        api_key: str | None = None,
        api_secret: str | None = None,
        demo: bool | None = None,
        base_url_demo: str | None = None,
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
                import logging

                import ccxt
            except ImportError as e:
                raise RuntimeError("ccxt is required for exchange execution. Install: pip install ccxt") from e

            try:
                exchange_class = getattr(ccxt, self.exchange_id)
                params = {"enableRateLimit": True}
                # For spot demo, ensure defaultType is spot (or leave default)
                if self.demo:
                    params["options"] = {"defaultType": "spot"}
                    # Read demo credentials from env vars
                    demo_key = os.getenv("BINANCE_DEMO_API_KEY", "").strip()
                    demo_secret = os.getenv("BINANCE_DEMO_API_SECRET", "").strip()
                    # Use env vars if provided, otherwise fall back to passed-in values
                    api_key_final = demo_key if demo_key else (self.api_key or "")
                    api_secret_final = demo_secret if demo_secret else (self.api_secret or "")
                else:
                    api_key_final = self.api_key or ""
                    api_secret_final = self.api_secret or ""

                if api_key_final:
                    params["apiKey"] = api_key_final
                if api_secret_final:
                    params["secret"] = api_secret_final

                ex = exchange_class(params)
                
                # Explicitly assign credentials (ensure they're attached)
                if api_key_final:
                    ex.apiKey = api_key_final
                if api_secret_final:
                    ex.secret = api_secret_final
                
                # Log credential status (never log full keys/secrets)
                logger = logging.getLogger("arbt")
                api_key_len = len(ex.apiKey or "") if hasattr(ex, "apiKey") else 0
                api_key_last4 = (ex.apiKey[-4:] if (hasattr(ex, "apiKey") and ex.apiKey and len(ex.apiKey) >= 4) else "NONE")
                has_secret = bool(ex.secret if hasattr(ex, "secret") else False)
                logger.info(f"apiKey_len={api_key_len} apiKey_last4={api_key_last4} has_secret={has_secret}")

                # For Binance Demo, manually set URLs (CCXT's sandbox uses testnet, not demo)
                if self.demo and self.base_url_demo:
                    try:
                        # CRITICAL: Disable currency fetching BEFORE any load_markets() call
                        # Demo API doesn't support sapi/v1/capital/config/getall
                        ex.options["fetchCurrencies"] = False
                        if hasattr(ex, "has"):
                            ex.has["fetchCurrencies"] = False

                        # Update URLs dictionary instead of replacing it (preserve CCXT's endpoint structure)
                        if "api" not in ex.urls:
                            ex.urls["api"] = {}
                        
                        # Use base hosts only - CCXT will append paths automatically
                        base_demo = self.base_url_demo.rstrip("/")
                        
                        # Update spot-specific endpoints
                        ex.urls["api"].update({
                            "public": base_demo,
                            "private": base_demo,
                        })
                        
                        # Disable sapi endpoints in demo mode (not supported)
                        # Delete any production sapi endpoints to prevent leakage
                        # Since fetchCurrencies is disabled, these won't be called anyway
                        sapi_keys_to_remove = ["sapi", "sapiV1", "sapiV2", "sapiV3", "sapiV4"]
                        for key in sapi_keys_to_remove:
                            ex.urls["api"].pop(key, None)
                        
                        # Mark as sandbox/demo mode for logging purposes
                        if hasattr(ex, "sandbox"):
                            ex.sandbox = True
                    except Exception:
                        pass
                elif self.demo:
                    # If no base_url_demo provided, use CCXT's sandbox mode (testnet)
                    if hasattr(ex, "set_sandbox_mode"):
                        ex.set_sandbox_mode(True)
                    # Still disable currency fetching for testnet
                    ex.options["fetchCurrencies"] = False
                    if hasattr(ex, "has"):
                        ex.has["fetchCurrencies"] = False

                # Load markets (currency fetching is disabled in demo mode)
                try:
                    ex.load_markets()
                except Exception:
                    pass

                # Log exchange configuration
                logger = logging.getLogger("arbt")
                default_type = ex.options.get("defaultType", "N/A")
                fetch_currencies = ex.options.get("fetchCurrencies", "N/A")
                has_fetch_currencies = ex.has.get("fetchCurrencies", "N/A") if hasattr(ex, "has") else "N/A"
                api_urls = ex.urls.get("api", {})
                # Log subset of relevant URLs
                url_subset = {
                    k: api_urls.get(k, "N/A")
                    for k in ["public", "private", "sapi"]
                    if k in api_urls
                }
                logger.info(
                    f"exchange init: demo={self.demo} defaultType={default_type} "
                    f"fetchCurrencies={fetch_currencies} has_fetchCurrencies={has_fetch_currencies} "
                    f"urls={url_subset}"
                )

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

    def get_open_positions(self, *, symbol: str | None = None) -> list[Position]:
        self._require_network()
        ex = self._get_exchange()
        bal = ex.fetch_balance()

        base_asset: str | None = None
        ccxt_sym: str | None = None
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

        params = {}
        
        # CRITICAL: Never pass 'option' as a param key - CCXT uses ex.options for exchange options
        # Validate params before sending
        if "option" in params:
            logger = logging.getLogger("arbt")
            logger.error(f"Invalid 'option' key found in params: {params}")
            return OrderAck(order_id="", status="rejected", reason="invalid parameter 'option' in order params")
        
        # Log sanitized params for debugging
        logger = logging.getLogger("arbt")
        sanitized_params = {k: v for k, v in params.items() if k != "apiKey" and k != "secret"}
        logger.debug(f"Order params (sanitized): {list(sanitized_params.keys())}")

        try:
            result = ex.create_market_order(symbol_ccxt, side_str, amount, params=params)
            return OrderAck(order_id=str(result.get("id", "")), status="filled")
        except Exception as e:
            return OrderAck(order_id="", status="rejected", reason=str(e))

    def cancel_order(self, order_id: str) -> bool:
        self._require_network()
        return False

    def iter_fills(self, *, since_ts_ms: int | None = None) -> Iterable[Fill]:
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

    def assert_private_access(self) -> None:
        """Verify private API access works (demo mode auth check)."""
        if not self.demo:
            return  # Only check in demo mode
        
        self._require_network()
        ex = self._get_exchange()
        logger = logging.getLogger("arbt")
        
        try:
            # Try fetch_balance for spot (minimal private endpoint)
            ex.fetch_balance()
        except Exception as e:
            error_str = str(e)
            # Check for -2015 auth error
            if "-2015" in error_str or "Invalid API-key" in error_str or "Invalid Api-Key" in error_str:
                api_key_last4 = (ex.apiKey[-4:] if (hasattr(ex, "apiKey") and ex.apiKey and len(ex.apiKey) >= 4) else "NONE")
                raise RuntimeError(
                    f"Demo auth failed (-2015). Check key/secret pair, IP restriction, and that broker attached credentials to CCXT exchange. apiKey_last4={api_key_last4}"
                ) from e
            # Re-raise other errors
            raise

    def close(self) -> None:
        if self._exchange is not None:
            try:
                self._exchange.close()
            except Exception:
                pass
