"""Tests for the LimitlessTCG source's paginated walk over the tournament index.

The index is paginated and newest-first. These tests drive the walk against a
stubbed API so the stop conditions — page exhausted, short page, ``since`` passed,
``max_decks`` reached — are pinned down without hitting the network, and so it's
visible that the walk does not keep requesting pages once it is provably done.
"""

from __future__ import annotations

import pytest

from scraper.sources.base import SourceFetchError
from scraper.sources.limitless import LimitlessSource


class FakeClient:
    """Stands in for :class:`~scraper.http.HttpClient` with a canned index.

    Records every request so tests can assert which pages were actually fetched —
    the point of the stop conditions is that later pages are never requested.
    """

    def __init__(self, pages: dict[int, list[dict]], standings: dict[str, list[dict]]):
        """
        :param pages: Page number -> the tournament objects that page returns.
        :param standings: Tournament ID -> its standings entries.
        """
        self.pages = pages
        self.standings = standings
        self.page_requests: list[int] = []
        self.standings_requests: list[str] = []

    def get_json(self, url: str, params: dict | None = None, headers=None):  # noqa: ARG002
        """
        Serve a canned index page or standings list.

        :param url: Requested URL.
        :param params: Query parameters (``page`` selects the index page).
        :param headers: Ignored.
        :return: The canned payload; an empty list past the last page.
        """
        if url.endswith("/tournaments"):
            page = (params or {}).get("page", 1)
            self.page_requests.append(page)
            return self.pages.get(page, [])
        tid = url.rsplit("/tournaments/", 1)[1].removesuffix("/standings")
        self.standings_requests.append(tid)
        return self.standings.get(tid, [])


def tournament(tid: str, date: str, name: str | None = None) -> dict:
    """
    :param tid: Tournament ID.
    :param date: ISO date (only the day part is used).
    :param name: Optional event name.
    :return: A tournament object shaped like the API's.
    """
    return {
        "id": tid,
        "date": f"{date}T00:00:00.000Z",
        "name": name or f"Event {tid}",
        "format": "STANDARD",
        "players": 32,
    }


def standing(placing: int, player: str) -> dict:
    """
    :param placing: Finishing position.
    :param player: Player handle.
    :return: A standings entry carrying a minimal one-card decklist.
    """
    return {
        "placing": placing,
        "player": player,
        "record": {"wins": 5, "losses": 1, "ties": 0},
        "deck": {"id": "some-deck", "name": "Some Deck"},
        "decklist": {"pokemon": [{"count": 4, "set": "MEG", "number": "1", "name": "Pikachu"}]},
    }


def source_over(pages, standings) -> tuple[LimitlessSource, FakeClient]:
    """
    :param pages: Page number -> tournament objects.
    :param standings: Tournament ID -> standings entries.
    :return: A source wired to a :class:`FakeClient`, and that client.
    """
    client = FakeClient(pages, standings)
    return LimitlessSource(client), client  # type: ignore[arg-type]


def test_walks_multiple_pages():
    """max_pages > 1 advances through the index instead of reading page 1 only."""
    pages = {n: [tournament(f"t{n}", f"2026-07-{30 - n:02d}")] for n in (1, 2, 3)}
    standings = {f"t{n}": [standing(1, f"p{n}")] for n in (1, 2, 3)}
    src, client = source_over(pages, standings)

    decks = list(src.iter_decks(limit=1, max_pages=3, per_tournament=8))

    assert client.page_requests == [1, 2, 3]
    assert [d.event for d in decks] == ["Event t1", "Event t2", "Event t3"]


def test_defaults_to_a_single_page():
    """The default stays one page, so existing invocations don't silently balloon."""
    pages = {n: [tournament(f"t{n}", "2026-07-01")] for n in (1, 2, 3)}
    standings = {f"t{n}": [standing(1, f"p{n}")] for n in (1, 2, 3)}
    src, client = source_over(pages, standings)

    list(src.iter_decks(limit=1))

    assert client.page_requests == [1]


def test_stops_when_the_index_runs_out():
    """An empty page ends the walk; max_pages is an upper bound, not a demand."""
    pages = {1: [tournament("t1", "2026-07-02")], 2: []}
    src, client = source_over(pages, {"t1": [standing(1, "p1")]})

    decks = list(src.iter_decks(limit=1, max_pages=50))

    assert client.page_requests == [1, 2]  # asked once more, then stopped
    assert len(decks) == 1


def test_stops_on_a_short_page():
    """A page shorter than the page size is the last one."""
    pages = {1: [tournament("t1", "2026-07-02"), tournament("t2", "2026-07-01")]}
    standings = {"t1": [standing(1, "a")], "t2": [standing(1, "b")]}
    src, client = source_over(pages, standings)

    list(src.iter_decks(limit=5, max_pages=50))  # asked for 5, got 2

    assert client.page_requests == [1]


def test_walks_until_exhausted_when_max_pages_is_zero():
    """max_pages=0 means "keep going", bounded only by the index itself."""
    pages = {n: [tournament(f"t{n}", f"2026-07-{20 - n:02d}")] for n in (1, 2, 3, 4)}
    standings = {f"t{n}": [standing(1, f"p{n}")] for n in (1, 2, 3, 4)}
    src, client = source_over(pages, standings)

    decks = list(src.iter_decks(limit=1, max_pages=0))

    assert client.page_requests == [1, 2, 3, 4, 5]  # 5 comes back empty
    assert len(decks) == 4


def test_max_decks_stops_mid_tournament():
    """max_decks is a hard stop, even partway through a tournament's standings."""
    pages = {1: [tournament("t1", "2026-07-02")]}
    standings = {"t1": [standing(i, f"p{i}") for i in range(1, 21)]}
    src, client = source_over(pages, standings)

    decks = list(src.iter_decks(limit=1, max_pages=0, per_tournament=0, max_decks=5))

    assert len(decks) == 5
    assert client.page_requests == [1]  # never asked for page 2


def test_since_ends_the_walk_rather_than_just_the_page():
    """Newest-first means one out-of-window date proves the rest are too.

    The walk must not keep paging backwards through years of older tournaments.
    """
    pages = {
        1: [tournament("new", "2026-07-20"), tournament("old", "2026-01-01")],
        2: [tournament("older", "2025-06-01")],
    }
    standings = {k: [standing(1, k)] for k in ("new", "old", "older")}
    src, client = source_over(pages, standings)

    decks = list(src.iter_decks(limit=2, max_pages=10, since="2026-07-01"))

    assert [d.event for d in decks] == ["Event new"]
    assert client.page_requests == [1]  # stopped without requesting page 2
    assert "older" not in client.standings_requests


def test_until_skips_newer_events_without_ending_the_walk():
    """`until` filters, but newer events don't imply the rest are out of range."""
    pages = {1: [tournament("newer", "2026-07-25"), tournament("wanted", "2026-06-01")]}
    standings = {"newer": [standing(1, "a")], "wanted": [standing(1, "b")]}
    src, _ = source_over(pages, standings)

    decks = list(src.iter_decks(limit=2, max_pages=1, until="2026-07-01"))

    assert [d.event for d in decks] == ["Event wanted"]


def test_per_tournament_caps_decks_best_finish_first():
    """The cap takes the top of the standings, which is the order the API returns."""
    pages = {1: [tournament("t1", "2026-07-02")]}
    standings = {"t1": [standing(i, f"p{i}") for i in range(1, 11)]}
    src, _ = source_over(pages, standings)

    decks = list(src.iter_decks(limit=1, per_tournament=3))

    assert [d.placing for d in decks] == [1, 2, 3]


def test_per_tournament_zero_takes_every_published_list():
    """0 means uncapped — the whole reason the cap was costing us most of the data."""
    pages = {1: [tournament("t1", "2026-07-02")]}
    standings = {"t1": [standing(i, f"p{i}") for i in range(1, 11)]}
    src, _ = source_over(pages, standings)

    decks = list(src.iter_decks(limit=1, per_tournament=0))

    assert len(decks) == 10


def test_standings_without_decklists_are_skipped():
    """Players who didn't publish a list aren't counted against the cap."""
    pages = {1: [tournament("t1", "2026-07-02")]}
    standings = {
        "t1": [
            {"placing": 1, "player": "a", "record": None, "decklist": None},
            standing(2, "b"),
        ]
    }
    src, _ = source_over(pages, standings)

    decks = list(src.iter_decks(limit=1, per_tournament=1))

    assert [d.placing for d in decks] == [2]


def test_an_unreachable_tournament_does_not_end_the_run():
    """One failing standings request is skipped; the walk carries on."""

    class Flaky(FakeClient):
        """Fails on one tournament's standings."""

        def get_json(self, url: str, params=None, headers=None):
            """Raise for the doomed tournament, otherwise behave normally."""
            if "/tournaments/boom/" in url:
                raise RuntimeError("502 upstream")
            return super().get_json(url, params=params, headers=headers)

    pages = {1: [tournament("boom", "2026-07-03"), tournament("fine", "2026-07-02")]}
    client = Flaky(pages, {"fine": [standing(1, "a")]})
    src = LimitlessSource(client)  # type: ignore[arg-type]

    decks = iter(src.iter_decks(limit=2, max_pages=1))

    assert next(decks).event == "Event fine"
    with pytest.raises(SourceFetchError, match="boom"):
        next(decks)
