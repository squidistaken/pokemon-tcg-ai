"""Tests for concurrent, failure-isolated source orchestration."""

from __future__ import annotations

import threading

from scraper.models import RawDeck
from scraper.source_runner import iter_source_events


class CoordinatedSource:
    """Wait for another worker before yielding, proving both were started."""

    def __init__(self, name: str, mine: threading.Event, other: threading.Event):
        self.name = name
        self.mine = mine
        self.other = other

    def iter_decks(self, **_kwargs):
        self.mine.set()
        assert self.other.wait(timeout=5)
        yield RawDeck(self.name, self.name)


def test_sources_start_concurrently_and_decks_return_to_one_consumer():
    left_started = threading.Event()
    right_started = threading.Event()
    sources = {
        "left": CoordinatedSource("left", left_started, right_started),
        "right": CoordinatedSource("right", right_started, left_started),
    }

    events = list(iter_source_events(sources, {}))

    assert {event.deck.source for event in events if event.deck} == {"left", "right"}
    assert sum(event.done for event in events) == 2


def test_source_failure_does_not_cancel_successful_worker():
    class BrokenSource:
        @staticmethod
        def iter_decks(**_kwargs):
            raise RuntimeError("remote failed")
            yield  # pragma: no cover

    class GoodSource:
        @staticmethod
        def iter_decks(**_kwargs):
            yield RawDeck("good", "deck")

    events = list(
        iter_source_events({"broken": BrokenSource(), "good": GoodSource()}, {})
    )

    assert any(event.deck and event.deck.source == "good" for event in events)
    errors = [event for event in events if event.error]
    assert len(errors) == 1
    assert errors[0].source == "broken"
    assert str(errors[0].error) == "remote failed"
