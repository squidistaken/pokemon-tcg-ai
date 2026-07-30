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
    """Thin requests wrapper that rate-limits and retries transient errors."""

    def __init__(self, min_interval: float = 1.0, timeout: int = DEFAULT_TIMEOUT):
        """
        :param min_interval: Minimum seconds between requests (rate limiting).
        :param timeout: Default per-request timeout in seconds.
        """
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
        """
        Sleep as needed so consecutive requests honour ``min_interval``.
        """
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.monotonic()

    def _get(self, url: str, **kwargs) -> requests.Response:
        """
        Rate-limited GET that raises on non-2xx responses.

        :param url: URL to fetch.
        :param kwargs: Extra arguments forwarded to ``requests.get``.
        :return: The successful response.
        :raises requests.HTTPError: If the final response status is an error.
        """
        self._throttle()
        kwargs.setdefault("timeout", self.timeout)
        resp = self.session.get(url, **kwargs)
        resp.raise_for_status()
        return resp

    def get_json(self, url: str, **kwargs):
        """
        GET a URL and parse the response body as JSON.

        :param url: URL to fetch.
        :param kwargs: Extra arguments forwarded to :meth:`get`.
        :return: The decoded JSON payload.
        """
        return self._get(url, **kwargs).json()
