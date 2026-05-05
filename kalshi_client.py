"""
Kalshi API Client
Handles authentication (RSA-PSS) and core API calls.
"""

import time
import base64
import requests
from datetime import datetime
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
                 json_body: dict = None, auth: bool = False) -> dict:
        """Make an API request, optionally with authentication."""
        url = f"{self.base_url}{path}"
        headers = self._sign_request(method, path) if auth else {}

        response = self.session.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json_body,
        )
        response.raise_for_status()
        return response.json()

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
        Place an order.

        ticker: market ticker (e.g. 'KXHIGHNY-25JUN15-T80')
        side: 'yes' or 'no'
        action: 'buy' or 'sell'
        count: number of contracts
        type: 'limit' or 'market'
        yes_price: price in cents (1-99) if buying/selling yes
        no_price: price in cents (1-99) if buying/selling no
        """
        body = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": type,
        }
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price
        if client_order_id:
            body["client_order_id"] = client_order_id

        return self._request("POST", "/portfolio/orders", json_body=body, auth=True)

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an open order."""
        return self._request("DELETE", f"/portfolio/orders/{order_id}", auth=True)

    def get_orders(self, ticker: str = None, status: str = None,
                   limit: int = 100) -> dict:
        """Get your orders, optionally filtered."""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status

        return self._request("GET", "/portfolio/orders", params=params, auth=True)


def create_client_from_config():
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
