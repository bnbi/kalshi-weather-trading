"""
Small HTTP helper: GET with bounded retries and backoff.

The public weather APIs (Open-Meteo, api.weather.gov, NCEI) return 429/502/
503/504 or time out several times a week (see launchd_stdout.log). A single
failed fetch used to drop a forecast source for the day, which pushed the
trained model to the naive average or left a verification row unfilled.
"""

from __future__ import annotations

import time

import requests

RETRY_STATUSES = (429, 500, 502, 503, 504)
DEFAULT_BACKOFF = (2.0, 5.0, 10.0)


def get_with_retry(url: str, params: dict | None = None,
                   headers: dict | None = None, timeout: float = 15,
                   retries: int = 3, backoff: tuple = DEFAULT_BACKOFF) -> requests.Response:
    """
    requests.get with retries on timeouts, connection errors and transient
    HTTP statuses. Returns the response (already checked with
    raise_for_status) or raises the last error.
    """
    last_err: Exception | None = None
    attempts = max(1, retries)
    for attempt in range(attempts):
        if attempt > 0:
            wait = backoff[min(attempt - 1, len(backoff) - 1)]
            # Rate limits deserve a longer pause than a flaky gateway.
            if isinstance(last_err, requests.HTTPError) and \
                    getattr(last_err.response, "status_code", None) == 429:
                wait = max(wait, 10.0)
            time.sleep(wait)
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            continue
        if resp.status_code in RETRY_STATUSES:
            last_err = requests.HTTPError(
                f"{resp.status_code} {resp.reason} for url: {resp.url}",
                response=resp)
            continue
        resp.raise_for_status()
        return resp
    assert last_err is not None
    raise last_err
