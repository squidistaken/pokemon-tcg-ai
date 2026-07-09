"""Shared HTTP session: polite User-Agent, timeouts, retries, rate limiting."""

from __future__ import annotations

import time

import requests
from requests.adapters import HTTPAdapter, Retry

USER_AGENT = (
    "pokemon-tcg-ai-deck-scraper/0.1 "
    "(+https://www.kaggle.com/competitions/pokemon-tcg-ai-battle; research use)"
)
DEFAULT_TIMEOUT = 20


class HttpClient:
    """A thin requests wrapper that rate-limits and retries transient errors."""

    def __init__(self, min_interval: float = 1.0, timeout: int = DEFAULT_TIMEOUT):
        self.min_interval = min_interval
        self.timeout = timeout
        self._last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        retries = Retry(
            total=3,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.monotonic()

    def get(self, url: str, **kwargs) -> requests.Response:
        self._throttle()
        kwargs.setdefault("timeout", self.timeout)
        resp = self.session.get(url, **kwargs)
        resp.raise_for_status()
        return resp

    def get_json(self, url: str, **kwargs):
        return self.get(url, **kwargs).json()

    def get_text(self, url: str, **kwargs) -> str:
        return self.get(url, **kwargs).text
