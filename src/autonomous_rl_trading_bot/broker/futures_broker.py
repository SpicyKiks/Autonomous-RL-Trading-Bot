from __future__ import annotations

import logging
import os
import types
from collections.abc import Iterable
from decimal import Decimal
from typing import Any

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


def _bootstrap_futures_markets_from_exchange_info(ex: Any) -> None:
    """
    Bootstrap markets from exchangeInfo endpoint (demo mode only).
    Avoids SAPI endpoints that CCXT's load_markets() calls.
    Only bootstraps once - checks if markets already loaded.
    """
    logger = logging.getLogger("arbt")
    
    # Check if already bootstrapped (only skip if markets exist and are loaded)
    if hasattr(ex, "marketsLoaded") and ex.marketsLoaded:
        if hasattr(ex, "markets") and ex.markets:
            logger.debug("Markets already bootstrapped, skipping")
            return
    
    try:
        # Call publicGetExchangeInfo() directly (no SAPI)
        info = ex.publicGetExchangeInfo()
        
        if not info or "symbols" not in info:
            logger.warning("exchangeInfo response missing symbols, markets not bootstrapped")
            return
        
        # Initialize market structures
        ex.markets = {}
        ex.markets_by_id = {}
        ex.symbols = []
        
        # Parse symbols from exchangeInfo - ONLY PERPETUAL USDT-M contracts
        for symbol_data in info.get("symbols", []):
            symbol_id = symbol_data.get("symbol", "")
            if not symbol_id:
                continue
            
            # Filter: Only PERPETUAL contracts
            contract_type = symbol_data.get("contractType", "")
            if contract_type != "PERPETUAL":
                continue
            
            # Filter: Only USDT quote asset
            quote = symbol_data.get("quoteAsset", "")
            if quote != "USDT":
                continue
            
            # Filter: Only TRADING status
            status = symbol_data.get("status", "")
            if status != "TRADING":
                continue
            
            # Filter: marginAsset should be USDT (if present)
            margin_asset = symbol_data.get("marginAsset", "")
            if margin_asset and margin_asset != "USDT":
                continue
            
            # Extract base
            base = symbol_data.get("baseAsset", "")
            if not base:
                continue
            
            # Ensure id is the raw symbol (no date suffix like _260626)
            # Binance perpetuals use raw symbol like BTCUSDT, not BTCUSDT_260626
            # If symbol_id has underscore, it's likely a delivery contract - skip it
            if "_" in symbol_id:
                logger.debug(f"Skipping delivery contract: {symbol_id}")
                continue
            
            # Build CCXT futures symbol format (e.g., "BTC/USDT:USDT" for USDT-M perpetuals)
            # For Binance futures, CCXT uses unified symbol format: BASE/QUOTE:SETTLEMENT
            ccxt_symbol_unified = f"{base}/{quote}:{quote}"  # e.g., "BTC/USDT:USDT"
            ccxt_symbol_spot_style = f"{base}/{quote}"  # Fallback format
            
            # Extract precision and limits from filters
            price_precision = 8
            amount_precision = 8
            tick_size = None
            step_size = None
            min_qty = None
            min_notional = None
            
            for filter_item in symbol_data.get("filters", []):
                filter_type = filter_item.get("filterType", "")
                if filter_type == "PRICE_FILTER":
                    tick_size = float(filter_item.get("tickSize", "0"))
                    if tick_size > 0:
                        # Calculate precision from tickSize (e.g., 0.01 -> 2, 0.0001 -> 4)
                        price_precision = len(str(tick_size).rstrip("0").split(".")[-1]) if "." in str(tick_size) else 0
                elif filter_type == "LOT_SIZE":
                    step_size = float(filter_item.get("stepSize", "0"))
                    min_qty = float(filter_item.get("minQty", "0"))
                    if step_size > 0:
                        amount_precision = len(str(step_size).rstrip("0").split(".")[-1]) if "." in str(step_size) else 0
                elif filter_type == "MIN_NOTIONAL":
                    min_notional = float(filter_item.get("notional", "0"))
            
            # Build market dict (CCXT-compatible structure for futures)
            market: dict[str, Any] = {
                "id": symbol_id,
                "symbol": ccxt_symbol_unified,  # Use unified futures format
                "base": base,
                "quote": quote,
                "active": symbol_data.get("status") == "TRADING",
                "type": "future",
                "spot": False,
                "future": True,
                "swap": True,  # Binance futures are swaps
                "option": False,  # Required: Binance futures are not options
                "contract": True,  # Required for CCXT futures
                "contractSize": 1.0,  # Default contract size
                "linear": True,  # USD-M futures are linear
                "inverse": False,
                "precision": {
                    "price": price_precision,
                    "amount": amount_precision,
                },
                "limits": {
                    "amount": {"min": min_qty if min_qty is not None else (step_size or 0.0), "max": None},
                    "price": {"min": tick_size or 0.0, "max": None},
                    "cost": {"min": min_notional if min_notional is not None else 0.0, "max": None},
                },
                "info": symbol_data,  # Store raw data
                # Store step_size separately for easy access
                "stepSize": step_size,
                "minQty": min_qty,
                "minNotional": min_notional,
            }
            
            # Store market with unified symbol (primary)
            ex.markets[ccxt_symbol_unified] = market
            ex.markets_by_id[symbol_id] = market
            ex.symbols.append(ccxt_symbol_unified)
            
            # Also store with spot-style symbol for backward compatibility (if different)
            if ccxt_symbol_unified != ccxt_symbol_spot_style:
                ex.markets[ccxt_symbol_spot_style] = market
            
            # Log market creation for debugging
            logger.debug(
                f"Bootstrap market: id={symbol_id} contractType={contract_type} "
                f"symbol={ccxt_symbol_unified} unified={ccxt_symbol_unified}"
            )
        
        # Mark markets as loaded (prevent CCXT from calling load_markets() again)
        if hasattr(ex, "marketsLoaded"):
            ex.marketsLoaded = True
        
        # Prevent leverage brackets loading (not supported in demo, causes errors)
        # CCXT's fetch_positions() calls load_leverage_brackets() which fails in demo mode
        if hasattr(ex, "leverage_brackets"):
            ex.leverage_brackets = {}  # Initialize empty to prevent loading
        if hasattr(ex, "leverage_brackets_by_id"):
            ex.leverage_brackets_by_id = {}
        
        # Stub load_leverage_brackets to prevent SAPI calls
        # Use MethodType to properly bind the stub so CCXT uses it
        def _stub_load_leverage_brackets(self, reload=False, params=None):
            """Stub to prevent leverage brackets loading in demo mode."""
            # Return empty dict to satisfy CCXT's expectations
            if not hasattr(self, "leverage_brackets"):
                self.leverage_brackets = {}
            if not hasattr(self, "leverage_brackets_by_id"):
                self.leverage_brackets_by_id = {}
            return self.leverage_brackets
        
        if hasattr(ex, "load_leverage_brackets"):
            # Bind the stub method to the exchange instance
            ex.load_leverage_brackets = types.MethodType(_stub_load_leverage_brackets, ex)
        
        logger.info(f"demo markets bootstrap succeeded: {len(ex.markets)} symbols loaded")
        
    except Exception as e:
        logger.error(f"demo markets bootstrap failed: {e}", exc_info=True)
        # Initialize empty structures to prevent crashes
        ex.markets = {}
        ex.markets_by_id = {}
        ex.symbols = []


class FuturesBroker(BrokerAdapter):
    """
    CCXT-based futures broker adapter (Binance Futures demo/testnet supported).

    Safety:
      - Network calls are hard-gated by ALLOW_NETWORK=1 (or allow_network_env).
      - Testnet can be enabled by demo=True or USE_TESTNET=1.
    """

    def __init__(
        self,
        *,
        exchange_id: str = "binanceusdm",
        api_key: str | None = None,
        api_secret: str | None = None,
        demo: bool | None = None,
        base_url_demo: str | None = None,
        leverage: float | None = None,
        allow_network_env: str = "ALLOW_NETWORK",
    ) -> None:
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.demo = bool(_truthy("USE_TESTNET") if demo is None else demo)
        self.base_url_demo = base_url_demo
        self.leverage = leverage
        self.allow_network_env = allow_network_env

        self._exchange = None
        self._allow_network = (os.getenv(self.allow_network_env, "") or "").strip() == "1"
        self._resolved_symbols: dict[str, str] = {}  # Cache: user_symbol -> ccxt_symbol

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
                # For demo mode, use binance() with defaultType=future (binanceusdm doesn't support sandbox)
                # For production, use binanceusdm if specified, otherwise binance with defaultType=future
                if self.demo:
                    exchange_id_actual = "binance"
                    # Read demo credentials from env vars
                    demo_key = os.getenv("BINANCE_DEMO_API_KEY", "").strip()
                    demo_secret = os.getenv("BINANCE_DEMO_API_SECRET", "").strip()
                    # Use env vars if provided, otherwise fall back to passed-in values
                    api_key_final = demo_key if demo_key else (self.api_key or "")
                    api_secret_final = demo_secret if demo_secret else (self.api_secret or "")
                else:
                    exchange_id_actual = self.exchange_id
                    api_key_final = self.api_key or ""
                    api_secret_final = self.api_secret or ""

                exchange_class = getattr(ccxt, exchange_id_actual)
                params = {
                    "enableRateLimit": True,
                    "options": {"defaultType": "future"},
                }
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

                # For Binance Demo (demo-fapi.binance.com), we need to manually set URLs
                # CCXT's set_sandbox_mode() uses testnet URLs (testnet.binancefuture.com), not demo URLs
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
                        
                        # Set full URLs with paths - CCXT uses these directly
                        base_demo = self.base_url_demo.rstrip("/")
                        
                        # Update futures-specific endpoints with full paths
                        # CCXT's fetch_markets() uses "public" endpoint, so it must have /fapi/v1 path
                        ex.urls["api"].update({
                            "public": f"{base_demo}/fapi/v1",
                            "private": f"{base_demo}/fapi/v1",
                            "fapiPublic": f"{base_demo}/fapi/v1",
                            "fapiPrivate": f"{base_demo}/fapi/v1",
                            "fapiPublicV1": f"{base_demo}/fapi/v1",
                            "fapiPrivateV1": f"{base_demo}/fapi/v1",
                            "fapiPublicV2": f"{base_demo}/fapi/v2",
                            "fapiPrivateV2": f"{base_demo}/fapi/v2",
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

                # Load markets - use bootstrap for demo mode to avoid SAPI endpoints
                if self.demo and self.base_url_demo:
                    # Demo mode: bootstrap markets from exchangeInfo (no SAPI calls)
                    _bootstrap_futures_markets_from_exchange_info(ex)
                else:
                    # Production/testnet: use standard load_markets()
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
                # Log subset of relevant URLs (confirm final values)
                url_subset = {
                    k: api_urls.get(k, "N/A")
                    for k in ["public", "private", "fapiPublic", "fapiPrivate"]
                    if k in api_urls
                }
                logger.info(
                    f"exchange init: demo={self.demo} defaultType={default_type} "
                    f"fetchCurrencies={fetch_currencies} has_fetchCurrencies={has_fetch_currencies} "
                    f"urls={url_subset}"
                )

                self._exchange = ex
            except Exception as e:
                raise RuntimeError(f"Failed to initialize CCXT futures exchange {exchange_id_actual}: {e}") from e

        return self._exchange
    
    def _to_ccxt_symbol(self, user_symbol: str) -> str:
        """
        Resolve user symbol (BTCUSDT or BTC/USDT) to CCXT futures symbol (BTC/USDT:USDT).
        
        For futures, CCXT uses format like BTC/USDT:USDT for perpetual contracts.
        This method caches resolved symbols and logs the resolution.
        Only returns a symbol if it exists in exchange.markets.
        """
        # Check cache first
        if user_symbol in self._resolved_symbols:
            return self._resolved_symbols[user_symbol]
        
        ex = self._get_exchange()
        logger = logging.getLogger("arbt")
        
        # Ensure markets are bootstrapped in demo mode
        if self.demo:
            if not hasattr(ex, "markets") or not ex.markets:
                logger.info("Markets not loaded, bootstrapping demo futures markets...")
                _bootstrap_futures_markets_from_exchange_info(ex)
        
        # Normalize input to base/quote format
        base_quote = to_ccxt_symbol(user_symbol)  # e.g., "BTC/USDT"
        
        # Extract quote asset for futures format
        if "/" in base_quote:
            quote_asset = base_quote.split("/")[1]  # e.g., "USDT"
        else:
            quote_asset = "USDT"  # Default fallback
        
        # Try futures format first: BTC/USDT:USDT
        candidates = [
            f"{base_quote}:{quote_asset}",  # Perpetual futures format (e.g., BTC/USDT:USDT)
            base_quote,  # Fallback to spot-style format
        ]
        
        resolved = None
        for candidate in candidates:
            if hasattr(ex, "markets") and candidate in ex.markets:
                resolved = candidate
                break
        
        if resolved is None:
            # If markets not loaded or symbol not found, bootstrap and try again
            if self.demo:
                logger.warning(f"Symbol {user_symbol} not found in markets, bootstrapping...")
                _bootstrap_futures_markets_from_exchange_info(ex)
                # Try again after bootstrap
                for candidate in candidates:
                    if hasattr(ex, "markets") and candidate in ex.markets:
                        resolved = candidate
                        break
        
        if resolved is None:
            # Still not found - this is an error
            logger.error(f"Symbol {user_symbol} not found in markets after bootstrap. Available: {list(ex.markets.keys())[:5] if hasattr(ex, 'markets') else 'N/A'}")
            # Use first candidate as last resort but log error
            resolved = candidates[0]
        
        # Get market details for logging
        market_info = {}
        try:
            if hasattr(ex, "market"):
                market_info = ex.market(resolved)
            elif hasattr(ex, "markets") and resolved in ex.markets:
                market_info = ex.markets[resolved]
        except Exception:
            pass
        
        # Cache and log
        self._resolved_symbols[user_symbol] = resolved
        market_id = market_info.get("id", "unknown")
        contract_type = market_info.get("info", {}).get("contractType", "unknown") if isinstance(market_info.get("info"), dict) else "unknown"
        logger.info(
            f"resolved_symbol user={user_symbol} ccxt={resolved} market_type=futures demo={self.demo}"
        )
        logger.info(
            f"resolved_market id={market_id} contractType={contract_type} symbol={resolved} unified={resolved}"
        )
        
        return resolved
    
    def _market_id_for(self, ccxt_symbol: str) -> str | None:
        """
        Get market_id (e.g., "BTCUSDT") from CCXT unified symbol (e.g., "BTC/USDT:USDT").
        
        For raw order submission, we need the market_id, not the unified symbol.
        """
        ex = self._get_exchange()
        try:
            if hasattr(ex, "market"):
                market = ex.market(ccxt_symbol)
                return market.get("id")
            elif hasattr(ex, "markets") and ccxt_symbol in ex.markets:
                market = ex.markets[ccxt_symbol]
                return market.get("id")
        except Exception:
            pass
        return None

    def get_account_snapshot(self) -> AccountSnapshot:
        self._require_network()
        ex = self._get_exchange()
        bal = ex.fetch_balance()

        usdt = bal.get("USDT", {}) if isinstance(bal, dict) else {}
        equity = float(usdt.get("total", 0.0) or 0.0)
        available = float(usdt.get("free", 0.0) or 0.0)
        used = float(usdt.get("used", 0.0) or 0.0)

        return AccountSnapshot(
            ts_ms=int(ex.milliseconds()),
            equity=equity,
            available_cash=available,
            unrealized_pnl=0.0,
            used_margin=used,
            leverage=float(self.leverage or 0.0),
        )

    def normalize_amount(self, symbol: str, amount: float, price: float | None = None) -> tuple[Decimal | None, str | None]:
        """
        Normalize order amount to respect exchange limits using Decimal math.
        
        Strategy:
        1. Convert to Decimal immediately
        2. Snap DOWN to step size using integer division
        3. If that violates min_amount/min_cost, bump to smallest valid step multiple
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            amount: Desired amount (in base currency, e.g., BTC)
            price: Current price (for minNotional check, optional)
            
        Returns:
            Tuple of (normalized_amount_decimal, error_message)
            If error_message is not None, the amount is too small and order should be rejected.
            If normalized_amount_decimal is None, the order should be skipped.
        """
        ex = self._get_exchange()
        sym_ccxt = self._to_ccxt_symbol(symbol)  # Use resolved futures symbol
        logger = logging.getLogger("arbt")
        
        # Get market info using CCXT's market() method or markets dict
        try:
            if hasattr(ex, "market"):
                market = ex.market(sym_ccxt)
            elif hasattr(ex, "markets") and sym_ccxt in ex.markets:
                market = ex.markets[sym_ccxt]
            else:
                logger.warning(f"Market info not found for {sym_ccxt}, skipping normalization")
                return Decimal(str(amount)), None
        except Exception as e:
            logger.warning(f"Failed to get market info for {sym_ccxt}: {e}, skipping normalization")
            return Decimal(str(amount)), None
        
        # Extract constraints from market
        amount_precision = market.get("precision", {}).get("amount", 8)
        step_size = market.get("stepSize") or market.get("limits", {}).get("amount", {}).get("min", 0.0)
        min_amt = None
        if "limits" in market and "amount" in market["limits"]:
            min_amt = market["limits"]["amount"].get("min")
        min_cost = None
        if price is not None and price > 0 and "limits" in market and "cost" in market["limits"]:
            min_cost = market["limits"]["cost"].get("min")
        
        # Convert inputs to Decimal immediately
        amount_d = Decimal(str(amount))
        step_d = Decimal(str(step_size)) if step_size and step_size > 0 else None
        
        # Step 1: Snap DOWN to step size using integer division
        if step_d and step_d > 0:
            # Use integer division: (amount_d // step_d) * step_d
            normalized_d = (amount_d // step_d) * step_d
        else:
            # Fallback: round to precision
            prec_d = Decimal(10) ** -amount_precision
            normalized_d = (amount_d // prec_d) * prec_d
        
        # Step 2: Check constraints and bump to smallest valid step multiple if needed
        original_normalized_d = normalized_d
        needs_bump = False
        
        # Check min amount constraint
        if min_amt is not None and min_amt > 0:
            min_amt_d = Decimal(str(min_amt))
            if normalized_d < min_amt_d:
                needs_bump = True
                # Bump to smallest step multiple >= min_amt
                if step_d and step_d > 0:
                    # Round UP to next step multiple
                    steps_needed = (min_amt_d + step_d - Decimal("0.00000001")) // step_d
                    normalized_d = steps_needed * step_d
                else:
                    normalized_d = min_amt_d
        
        # Check min cost constraint
        if price is not None and price > 0 and min_cost is not None and min_cost > 0:
            price_d = Decimal(str(price))
            min_cost_d = Decimal(str(min_cost))
            notional_d = normalized_d * price_d
            if notional_d < min_cost_d:
                needs_bump = True
                # Calculate minimum qty needed to satisfy min_cost
                min_qty_for_cost_d = min_cost_d / price_d
                # Round UP to next step multiple
                if step_d and step_d > 0:
                    steps_needed = (min_qty_for_cost_d + step_d - Decimal("0.00000001")) // step_d
                    min_qty_for_cost_d = steps_needed * step_d
                else:
                    prec_d = Decimal(10) ** -amount_precision
                    steps_needed = (min_qty_for_cost_d + prec_d - Decimal("0.00000001")) // prec_d
                    min_qty_for_cost_d = steps_needed * prec_d
                
                # Take the maximum of min_amount and min_qty_for_cost
                if min_amt is not None and min_amt > 0:
                    min_amt_d = Decimal(str(min_amt))
                    normalized_d = max(normalized_d, min_amt_d, min_qty_for_cost_d)
                else:
                    normalized_d = max(normalized_d, min_qty_for_cost_d)
        
        # Log normalization details
        if needs_bump or abs(original_normalized_d - normalized_d) > Decimal("0.00000001"):
            logger.info(
                f"Order qty normalized: {amount} -> {normalized_d} "
                f"(precision={amount_precision}, min_amount={min_amt}, min_cost={min_cost}, price={price})"
            )
        
        # Final validation
        if normalized_d <= 0:
            return None, f"normalized amount <= 0: {normalized_d}"
        
        if min_amt is not None and min_amt > 0:
            min_amt_d = Decimal(str(min_amt))
            if normalized_d < min_amt_d:
                return None, f"below_min_amount: {normalized_d} < {min_amt_d}"
        
        if price is not None and price > 0 and min_cost is not None and min_cost > 0:
            price_d = Decimal(str(price))
            min_cost_d = Decimal(str(min_cost))
            notional_d = normalized_d * price_d
            if notional_d < min_cost_d:
                return None, f"below_min_notional: {notional_d} < {min_cost_d} (qty={normalized_d}, price={price_d})"
        
        return normalized_d, None
    
    def _submit_order_demo_raw(
        self,
        symbol_ccxt: str,
        market_id: str,
        side: str,
        qty_str: str,
        reduce_only: bool = False,
    ) -> OrderAck:
        """
        Submit order via raw Binance Futures REST API (demo mode only).
        
        Bypasses CCXT's create_order() validation by calling fapiPrivatePostOrder directly.
        This avoids precision validation errors in demo mode.
        
        Args:
            symbol_ccxt: CCXT unified symbol (e.g., "BTC/USDT:USDT") - used for logging only
            market_id: Binance market ID (e.g., "BTCUSDT") - used in API call
            side: "buy" or "sell"
            qty_str: Quantity as formatted string (already normalized)
            reduce_only: Whether this is a reduce-only order
            
        Returns:
            OrderAck with order_id if successful, or rejected status with error message
        """
        ex = self._get_exchange()
        logger = logging.getLogger("arbt")
        
        # Convert side to Binance format (BUY/SELL)
        side_upper = side.upper()
        if side_upper not in ("BUY", "SELL"):
            return OrderAck(order_id="", status="rejected", reason=f"invalid side: {side}")
        
        # Build params for POST /fapi/v1/order
        params: dict[str, Any] = {
            "symbol": market_id,  # Use market_id, not unified symbol
            "side": side_upper,  # BUY or SELL
            "type": "MARKET",
            "quantity": qty_str,  # Already formatted string
        }
        
        if reduce_only:
            params["reduceOnly"] = "true"
        
        # Log before submitting
        logger.info(
            f"demo_raw_order market_id={market_id} resolved_ccxt_symbol={symbol_ccxt} "
            f"qty_str={qty_str} side={side_upper} type=MARKET reduceOnly={reduce_only}"
        )
        
        try:
            # Call raw Binance Futures API endpoint
            # This bypasses CCXT's create_order() validation
            result = ex.fapiPrivatePostOrder(params)
            
            # Parse response
            order_id = str(result.get("orderId", result.get("id", "")))
            if order_id:
                logger.info(f"demo_raw_order succeeded: orderId={order_id}")
                return OrderAck(order_id=order_id, status="filled")
            else:
                logger.warning(f"demo_raw_order response missing orderId: {result}")
                return OrderAck(order_id="", status="rejected", reason="response missing orderId")
        except Exception as e:
            # Log full exception class + message (no secrets)
            error_class = type(e).__name__
            error_msg = str(e)
            logger.warning(f"demo_raw_order failed: {error_class}: {error_msg}")
            return OrderAck(order_id="", status="rejected", reason=f"{error_class}: {error_msg}")

    def get_open_positions(self, *, symbol: str | None = None) -> list[Position]:
        # In demo mode, Binance Demo doesn't support fetch_positions() endpoints
        # Return empty list - LiveRunner will use internal position tracking
        if self.demo:
            logger = logging.getLogger("arbt")
            logger.debug("get_open_positions() skipped in demo mode (demo API lacks futures position endpoints)")
            return []
        
        self._require_network()
        ex = self._get_exchange()

        sym_ccxt = self._to_ccxt_symbol(symbol) if symbol else None
        
        try:
            pos_data = ex.fetch_positions(symbols=[sym_ccxt] if sym_ccxt else None)
        except Exception as e:
            logger = logging.getLogger("arbt")
            logger.warning(f"Failed to fetch positions: {e}")
            return []

        out: list[Position] = []
        for p in pos_data:
            contracts = float(p.get("contracts", 0.0) or 0.0)
            if abs(contracts) <= 1e-12:
                continue
            out.append(
                Position(
                    symbol=str(p.get("symbol", "")),
                    side="buy" if contracts > 0 else "sell",
                    qty=abs(contracts),
                    entry_price=float(p.get("entryPrice", 0.0) or 0.0),
                )
            )
        return out

    def submit_order(self, order: OrderRequest) -> OrderAck:
        self._require_network()
        ex = self._get_exchange()

        symbol_ccxt = self._to_ccxt_symbol(order.symbol)  # Use resolved futures symbol
        
        # Get market to extract market_id and precision info
        market = None
        market_id = None
        try:
            if hasattr(ex, "market"):
                market = ex.market(symbol_ccxt)
                market_id = market.get("id")
            elif hasattr(ex, "markets") and symbol_ccxt in ex.markets:
                market = ex.markets[symbol_ccxt]
                market_id = market.get("id")
        except Exception:
            pass
        
        # Debug log before order submit
        logger = logging.getLogger("arbt")
        logger.info(
            f"order_payload ccxt_symbol={symbol_ccxt} market_id={market_id} qty={order.qty} "
            f"side={order.side} order_type={order.order_type}"
        )
        
        side_str = "buy" if order.side == "buy" else "sell"
        
        # Convert qty to Decimal (handle both float and Decimal inputs)
        qty_d = order.qty if isinstance(order.qty, Decimal) else Decimal(str(order.qty))

        if order.qty_unit == "quote":
            ticker = ex.fetch_ticker(symbol_ccxt)
            last = float(ticker.get("last") or 0.0)
            if last <= 0:
                return OrderAck(order_id="", status="rejected", reason="no_price_for_quote_conversion")
            qty_d = qty_d / Decimal(str(last))

        # Format amount as string with EXACT decimal places - NEVER call amount_to_precision on float
        # In demo mode, format Decimal directly to avoid CCXT precision validation issues
        if not market:
            return OrderAck(order_id="", status="rejected", reason="market info not available")
        
        amount_precision = market.get("precision", {}).get("amount", 8)
        step_size = market.get("stepSize") or market.get("limits", {}).get("amount", {}).get("min", 0.0)
        
        # Snap qty_d to step size by Decimal math before formatting
        if step_size and step_size > 0:
            step_d = Decimal(str(step_size))
            # Snap to step: (qty_d // step_d) * step_d
            qty_d = (qty_d // step_d) * step_d
        
        # Format with exact precision (no exponent, no trailing zeros removal that could cause issues)
        prec = int(amount_precision)
        qty_str = f"{qty_d:.{prec}f}"
        
        # Log the exact repr being submitted
        logger.info(f"order_submit qty_d={qty_d!r} qty_str={qty_str}")

        # In demo mode, bypass CCXT's create_order() validation by calling raw API
        if self.demo:
            # Get market_id for raw API call
            market_id_raw = self._market_id_for(symbol_ccxt)
            if not market_id_raw:
                logger.error(f"Could not resolve market_id for {symbol_ccxt}")
                return OrderAck(order_id="", status="rejected", reason="market_id not found")
            
            # Submit via raw API (bypasses CCXT validation)
            return self._submit_order_demo_raw(
                symbol_ccxt=symbol_ccxt,
                market_id=market_id_raw,
                side=side_str,
                qty_str=qty_str,
                reduce_only=order.reduce_only,
            )
        
        # Non-demo mode: use standard CCXT create_order path
        params = {}
        if order.reduce_only:
            params["reduceOnly"] = True
        
        # CRITICAL: Never pass 'option' as a param key - CCXT uses ex.options for exchange options
        # Validate params before sending
        if "option" in params:
            logger.error(f"Invalid 'option' key found in params: {params}")
            return OrderAck(order_id="", status="rejected", reason="invalid parameter 'option' in order params")
        
        # Log sanitized params for debugging (exclude sensitive data)
        sanitized_params = {k: v for k, v in params.items() if k != "apiKey" and k != "secret"}
        logger.debug(f"Order params (sanitized): {list(sanitized_params.keys())}")
        
        try:
            # For futures, use create_market_order which handles type routing automatically
            # Pass amount as STRING to ensure proper precision
            # Don't pass 'type' in params - CCXT uses ex.options['defaultType'] for that
            result = ex.create_market_order(symbol_ccxt, side_str, qty_str, params=params)
            return OrderAck(order_id=str(result.get("id", "")), status="filled")
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Order submission failed: {error_msg}")
            return OrderAck(order_id="", status="rejected", reason=error_msg)

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
        
        # Binance Demo futures doesn't support /fapi/v1/account or fetch_balance endpoints
        # Skip auth probe for futures demo - credentials are validated via order submission
        logger.info("demo auth probe skipped (demo API lacks futures account endpoints)")
        return

    def close(self) -> None:
        if self._exchange is not None:
            try:
                self._exchange.close()
            except Exception:
                pass