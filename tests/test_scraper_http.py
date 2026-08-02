"""Concurrency guarantees for the shared per-source HTTP limiter."""

from __future__ import annotations

import threading
import time
from typing import override

import pytest
import requests

from scraper.http import HttpClient


class FakeResponse:
    text = "ok"

    def __init__(self, status_code: int = 200, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self.headers = headers or {}
        self.closed = False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def close(self) -> None:
        self.closed = True


class TrackingSession:
    """Record whether two callers ever enter one session simultaneously."""

    def __init__(self):
        self.headers: dict[str, str] = {}
        self.active = 0
        self.overlapped = False

    @staticmethod
    def mount(_prefix: str, _adapter) -> None:
        return None

    def get(self, _url: str, **_kwargs) -> FakeResponse:
        self.active += 1
        self.overlapped |= self.active > 1
        time.sleep(0.01)
        self.active -= 1
        return FakeResponse()


def test_one_client_serializes_requests_from_source_and_profile_workers():
    client = HttpClient(min_interval=0)
    session = TrackingSession()
    client.session = session  # type: ignore[assignment]
    barrier = threading.Barrier(3)

    def request() -> None:
        barrier.wait()
        client.get_text("https://example.invalid")

    workers = [threading.Thread(target=request) for _ in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join()

    assert session.overlapped is False


def test_one_client_spaces_request_starts_by_its_configured_interval(monkeypatch):
    class Clock:
        now = 10.0

        @classmethod
        def monotonic(cls) -> float:
            return cls.now

        @classmethod
        def sleep(cls, seconds: float) -> None:
            cls.now += seconds

    starts: list[float] = []

    class TimedSession(TrackingSession):
        @override
        def get(self, _url: str, **_kwargs) -> FakeResponse:
            starts.append(Clock.now)
            return FakeResponse()

    monkeypatch.setattr("scraper.http.time.monotonic", Clock.monotonic)
    monkeypatch.setattr("scraper.http.time.sleep", Clock.sleep)
    client = HttpClient(min_interval=0.5)
    client.session = TimedSession()  # type: ignore[assignment]

    client.get_text("https://example.invalid/one")
    client.get_text("https://example.invalid/two")

    assert starts == [10.0, 10.5]


def test_retry_attempts_also_obey_configured_interval(monkeypatch):
    class Clock:
        now = 10.0

        @classmethod
        def monotonic(cls) -> float:
            return cls.now

        @classmethod
        def sleep(cls, seconds: float) -> None:
            cls.now += seconds

    responses = [FakeResponse(503), FakeResponse()]
    starts: list[float] = []

    class RetryingSession(TrackingSession):
        @override
        def get(self, _url: str, **_kwargs) -> FakeResponse:
            starts.append(Clock.now)
            return responses.pop(0)

    monkeypatch.setattr("scraper.http.time.monotonic", Clock.monotonic)
    monkeypatch.setattr("scraper.http.time.sleep", Clock.sleep)
    client = HttpClient(min_interval=0.5, backoff_factor=0)
    first_response = responses[0]
    client.session = RetryingSession()  # type: ignore[assignment]

    assert client.get_text("https://example.invalid") == "ok"
    assert starts == [10.0, 10.5]
    assert first_response.closed is True


def test_retry_after_takes_precedence_over_backoff(monkeypatch):
    sleeps: list[float] = []

    class RetryingSession(TrackingSession):
        def __init__(self):
            super().__init__()
            self.responses = [FakeResponse(429, {"Retry-After": "2"}), FakeResponse()]

        @override
        def get(self, _url: str, **_kwargs) -> FakeResponse:
            return self.responses.pop(0)

    monkeypatch.setattr("scraper.http.time.sleep", sleeps.append)
    client = HttpClient(min_interval=0, backoff_factor=10)
    client.session = RetryingSession()  # type: ignore[assignment]

    assert client.get_text("https://example.invalid") == "ok"
    assert sleeps == [2.0]


def test_transient_request_errors_retry_only_up_to_limit(monkeypatch):
    class FailingSession(TrackingSession):
        calls = 0

        @override
        def get(self, _url: str, **_kwargs) -> FakeResponse:
            self.calls += 1
            raise requests.ConnectionError("offline")

    monkeypatch.setattr("scraper.http.time.sleep", lambda _seconds: None)
    client = HttpClient(min_interval=0, max_retries=2, backoff_factor=0)
    session = FailingSession()
    client.session = session  # type: ignore[assignment]

    with pytest.raises(requests.ConnectionError, match="offline"):
        client.get_text("https://example.invalid")

    assert session.calls == 3


def test_separate_clients_can_make_requests_concurrently():
    barrier = threading.Barrier(2)
    errors: list[threading.BrokenBarrierError] = []

    class BarrierSession(TrackingSession):
        @override
        def get(self, _url: str, **_kwargs) -> FakeResponse:
            barrier.wait(timeout=1)
            return FakeResponse()

    clients = [HttpClient(min_interval=0), HttpClient(min_interval=0)]
    for client in clients:
        client.session = BarrierSession()  # type: ignore[assignment]

    def request(client: HttpClient) -> None:
        try:
            client.get_text("https://example.invalid")
        except threading.BrokenBarrierError as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    workers = [threading.Thread(target=request, args=(client,)) for client in clients]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2)

    assert all(not worker.is_alive() for worker in workers)
    assert errors == []
