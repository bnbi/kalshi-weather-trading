"""
Kalshi API Client
Handles authentication (RSA-PSS) and core API calls.
"""

import time
import base64
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class KalshiClient:
    """Client for interacting with the Kalshi trading API."""

    def __init__(self, api_key_id: str, private_key_path: str = "",
                 base_url: str = "https://api.elections.kalshi.com/trade-api/v2"):
        self.api_key_id = api_key_id
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

        # Load RSA private key (only needed for authenticated endpoints)
        self.private_key = None
        if private_key_path:
            with open(private_key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(), password=None
                )

    def _sign_request(self, method: str, path: str) -> dict:
        """
        Generate authentication headers using RSA-PSS signing.
        Signs: timestamp + method + path (without query params).
        """
        timestamp_ms = str(int(time.time() * 1000))
        # Always sign the path without query parameters
        clean_path = path.split("?")[0]
        message = f"{timestamp_ms}{method.upper()}{clean_path}"

        signature = self.private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, params: dict = None,
                 json_body: dict = None, auth: bool = False,
                 max_retries: int = 3) -> dict:
        """
        Make an API request, optionally with authentication.

        Retries transient failures (401, 5xx, timeouts, connection errors)
        with a fresh signature timestamp on each attempt. Intermittent 401s
        happen when the local clock is momentarily skewed (e.g. right after
        the machine wakes from sleep), so re-signing usually fixes them.

        POST/DELETE requests are NOT retried after a timeout/connection
        error, since the order may have gone through — only auth (401)
        failures are retried for those, which are safe (nothing executed).
        """
        url = f"{self.base_url}{path}"
        is_mutation = method.upper() in ("POST", "DELETE")

        last_error = None
        for attempt in range(max_retries):
            if attempt > 0:
                time.sleep(2 ** attempt)  # 2s, 4s backoff

            # Sign the FULL path (e.g. /trade-api/v2/portfolio/balance)
            # not just the relative path (/portfolio/balance).
            # Re-sign every attempt so the timestamp is fresh.
            if auth:
                full_path = urlparse(url).path
                headers = self._sign_request(method, full_path)
            else:
                headers = {}

            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json_body,
                    timeout=15,
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                last_error = e
                if is_mutation:
                    raise  # order may have been placed — don't double-submit
                continue

            if response.status_code == 401 or response.status_code >= 500:
                last_error = requests.HTTPError(
                    f"{response.status_code} {response.reason} for url: {url}",
                    response=response,
                )
                # 401/5xx before execution — safe to retry even mutations
                continue

            response.raise_for_status()
            return response.json()

        raise last_error

    # ── Public endpoints (no auth required) ─────────────────────────

    def get_markets(self, limit: int = 100, cursor: str = None,
                    status: str = "open", series_ticker: str = None,
                    event_ticker: str = None) -> dict:
        """
        Fetch a list of markets.
        status: 'open', 'closed', 'settled'
        """
        params = {"limit": limit, "status": status}
        if cursor:
            params["cursor"] = cursor
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker

        return self._request("GET", "/markets", params=params)

    def get_market(self, ticker: str) -> dict:
        """Get details for a single market by ticker."""
        return self._request("GET", f"/markets/{ticker}")

    def get_orderbook(self, ticker: str) -> dict:
        """Get the order book for a market."""
        return self._request("GET", f"/markets/{ticker}/orderbook")

    def get_events(self, limit: int = 100, cursor: str = None,
                   status: str = None, series_ticker: str = None) -> dict:
        """Fetch a list of events."""
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker

        return self._request("GET", "/events", params=params)

    def get_event(self, event_ticker: str) -> dict:
        """Get details for a single event."""
        return self._request("GET", f"/events/{event_ticker}")

    def get_series(self, series_ticker: str) -> dict:
        """Get details for a series."""
        return self._request("GET", f"/series/{series_ticker}")

    # ── Authenticated endpoints ─────────────────────────────────────

    def get_balance(self) -> dict:
        """Get your account balance (requires auth)."""
        return self._request("GET", "/portfolio/balance", auth=True)

    def get_positions(self, limit: int = 100, cursor: str = None,
                      settlement_status: str = None) -> dict:
        """Get your current positions."""
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if settlement_status:
            params["settlement_status"] = settlement_status

        return self._request("GET", "/portfolio/positions", params=params, auth=True)

    def create_order(self, ticker: str, side: str, action: str,
                     count: int, type: str = "limit",
                     yes_price: int = None, no_price: int = None,
                     client_order_id: str = None) -> dict:
        """
        Place a limit order via CreateOrder V2.

        Kalshi removed the legacy POST /portfolio/orders endpoint (410 Gone
        since late June 2026). The V2 endpoint quotes everything from the
        YES side of the single book:
            buy YES at p          -> side='bid', price=p
            buy NO  at p          -> side='ask', price=1-p
              (selling YES you don't hold is how a NO position is opened;
               economically identical to buying NO at 1-price)
        Prices and counts are fixed-point strings.

        The (ticker, side yes/no, action, count, *_price cents) signature is
        kept so existing callers don't change.
        """
        if action != "buy":
            raise ValueError(f"only 'buy' orders supported, got {action!r}")

        if side == "yes":
            if yes_price is None:
                raise ValueError("yes_price required for side='yes'")
            book_side = "bid"
            price_dollars = yes_price / 100
        elif side == "no":
            if no_price is None:
                raise ValueError("no_price required for side='no'")
            book_side = "ask"
            price_dollars = (100 - no_price) / 100
        else:
            raise ValueError(f"side must be 'yes' or 'no', got {side!r}")

        body = {
            "ticker": ticker,
            "side": book_side,
            "count": str(count),
            "price": f"{price_dollars:.2f}",
            "time_in_force": "good_till_canceled",
            "self_trade_prevention_type": "taker_at_cross",
        }
        if client_order_id:
            body["client_order_id"] = client_order_id

        response = self._request("POST", "/portfolio/events/orders",
                                 json_body=body, auth=True)
        # V2 returns order fields at the top level; legacy nested them under
        # "order". Normalize so existing callers keep working.
        if "order" not in response:
            response = {"order": response, **response}
        return response

    def get_order(self, order_id: str) -> dict:
        """Get a single order — fill counts, actual costs and fees."""
        return self._request("GET", f"/portfolio/orders/{order_id}", auth=True)

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an open order (V2 endpoint)."""
        return self._request("DELETE", f"/portfolio/events/orders/{order_id}",
                             auth=True)

    def get_orders(self, ticker: str = None, status: str = None,
                   limit: int = 100) -> dict:
        """Get your orders, optionally filtered."""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status

        return self._request("GET", "/portfolio/orders", params=params, auth=True)


def create_client_from_config() -> KalshiClient:
    """Create a KalshiClient using settings from config.py."""
    try:
        import config
    except ImportError:
        raise RuntimeError(
            "config.py not found. Copy config.example.py to config.py "
            "and fill in your API credentials."
        )

    base_url = config.KALSHI_DEMO_URL if config.USE_DEMO else config.KALSHI_BASE_URL
    return KalshiClient(
        api_key_id=config.KALSHI_API_KEY_ID,
        private_key_path=config.KALSHI_PRIVATE_KEY_PATH,
        base_url=base_url,
    )
