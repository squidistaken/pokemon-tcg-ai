from __future__ import annotations

import threading
import time
from email.utils import parsedate_to_datetime

import requests
from requests.adapters import HTTPAdapter

USER_AGENT = (
    "pokemon-tcg-ai-deck-scraper/0.1 "
    "(+https://www.kaggle.com/competitions/pokemon-tcg-ai-battle; research use)"
)
DEFAULT_TIMEOUT = 20
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 1.5
RETRYABLE_STATUS_CODES = frozenset((429, 500, 502, 503, 504))


class HttpClient:
    """Thin requests wrapper that rate-limits and retries transient errors."""

    def __init__(
        self,
        min_interval: float = 1.0,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    ):
        """
        :param min_interval: Minimum seconds between requests (rate limiting).
        :param timeout: Default per-request timeout in seconds.
        :param max_retries: Retries after the initial request for transient failures.
        :param backoff_factor: Base seconds for exponential retry backoff.
        """
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._last_request: float | None = None
        self._request_lock = threading.Lock()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        # Retries are deliberately handled above the adapter so every network
        # attempt passes through the same rate limiter.
        adapter = HTTPAdapter(max_retries=0)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _throttle(self) -> None:
        """
        Sleep as needed so consecutive requests honour ``min_interval``.
        """
        now = time.monotonic()
        if self._last_request is not None:
            elapsed = now - self._last_request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        self._last_request = time.monotonic()

    def _retry_delay(
        self,
        retry_number: int,
        response: requests.Response | None,
    ) -> float:
        """Return server-requested delay or exponential backoff in seconds."""
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    try:
                        retry_at = parsedate_to_datetime(retry_after).timestamp()
                    except (TypeError, ValueError, OverflowError):
                        pass
                    else:
                        return max(0.0, retry_at - time.time())
        return self.backoff_factor * (2 ** (retry_number - 1))

    def _get(self, url: str, **kwargs) -> requests.Response:
        """
        Rate-limited GET that raises on non-2xx responses.

        :param url: URL to fetch.
        :param kwargs: Extra arguments forwarded to ``requests.get``.
        :return: The successful response.
        :raises requests.HTTPError: If the final response status is an error.
        """
        with self._request_lock:
            kwargs.setdefault("timeout", self.timeout)
            for attempt in range(self.max_retries + 1):
                self._throttle()
                try:
                    response = self.session.get(url, **kwargs)
                except requests.RequestException:
                    if attempt == self.max_retries:
                        raise
                    time.sleep(self._retry_delay(attempt + 1, None))
                    continue

                if (
                    response.status_code in RETRYABLE_STATUS_CODES
                    and attempt < self.max_retries
                ):
                    delay = self._retry_delay(attempt + 1, response)
                    response.close()
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                return response

        raise RuntimeError("HTTP retry loop exited unexpectedly")

    def get_json(self, url: str, **kwargs):
        """
        GET a URL and parse the response body as JSON.

        :param url: URL to fetch.
        :param kwargs: Extra arguments forwarded to :meth:`get`.
        :return: The decoded JSON payload.
        """
        return self._get(url, **kwargs).json()

    def get_text(self, url: str, **kwargs) -> str:
        """GET a URL and return its decoded response body."""
        return self._get(url, **kwargs).text
