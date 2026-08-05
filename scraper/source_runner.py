"""Concurrent orchestration for independent network deck sources."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from .models import RawDeck
from .sources.base import DeckSource


@dataclass(frozen=True)
class SourceEvent:
    """One deck, terminal error, or completion emitted by a source worker."""

    source: str
    deck: RawDeck | None = None
    error: Exception | None = None
    done: bool = False


def iter_source_events(
    sources: Mapping[str, DeckSource], source_kwargs: Mapping[str, object]
) -> Iterator[SourceEvent]:
    """Fetch independent sources concurrently and yield their events centrally.

    Each source owns its client and rate limiter. The caller remains the only
    consumer, so inventory aggregation and deck writers never run concurrently.
    A source failure is represented as an event instead of cancelling the other
    workers.

    :param sources: Instantiated sources keyed by their CLI names.
    :param source_kwargs: Shared keyword arguments accepted by source iterators.
    :return: Deck, error, and completion events as workers produce them.
    """
    events: queue.SimpleQueue[SourceEvent] = queue.SimpleQueue()

    def fetch(name: str, source: DeckSource) -> None:
        try:
            for deck in source.iter_decks(**source_kwargs):
                events.put(SourceEvent(source=name, deck=deck))
        except Exception as exc:  # noqa: BLE001 - isolate one remote source
            events.put(SourceEvent(source=name, error=exc))
        finally:
            events.put(SourceEvent(source=name, done=True))

    workers = [
        threading.Thread(
            target=fetch,
            args=(name, source),
            name=f"scraper-{name}",
            daemon=True,
        )
        for name, source in sources.items()
    ]
    for worker in workers:
        worker.start()

    completed = 0
    while completed < len(workers):
        event = events.get()
        if event.done:
            completed += 1
        yield event

    for worker in workers:
        worker.join()
