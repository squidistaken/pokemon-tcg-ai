from __future__ import annotations

import os
from collections.abc import Iterable

from ..http import HttpClient
from ..models import RawCard, RawDeck
from .base import DeckSource, SourceFetchError

API_BASE = "https://play.limitlesstcg.com/api"


def _fmt_record(record) -> str | None:
    """
    Format a standings ``record`` object into a ``"W-L-T"`` string.

    :param record: The API's record value (a dict of wins/losses/ties, or other).
    :return: ``"W-L-T"``, or None when the record is missing or malformed.
    """
    if isinstance(record, dict):
        w, l, t = record.get("wins"), record.get("losses"), record.get("ties")
        if w is not None:
            return f"{w}-{l}-{t}"
    return None


class LimitlessSource(DeckSource):
    """Deck source backed by the LimitlessTCG public tournament JSON API."""

    name = "limitless"

    def __init__(self, client: HttpClient | None = None):
        """
        :param client: HTTP client to use; a 1s-rate-limited one by default.
            An optional ``LIMITLESS_API_KEY`` env var is sent as an access key.
        """
        self.client = client or HttpClient(min_interval=1.0)
        self.api_key = os.environ.get("LIMITLESS_API_KEY")

    def _headers(self) -> dict:
        """
        :return: Request headers carrying the access key, or empty if unset.
        """
        return {"X-Access-Key": self.api_key} if self.api_key else {}

    @staticmethod
    def _cards_from_decklist(decklist: dict) -> list[RawCard]:
        """
        Flatten an API decklist's pokemon/trainer/energy groups into RawCards.

        :param decklist: A standing's ``decklist`` object.
        :return: One RawCard per listed card (name/set/number/count preserved).
        """
        cards: list[RawCard] = []
        for category in ("pokemon", "trainer", "energy"):
            for entry in decklist.get(category) or []:
                cards.append(
                    RawCard(
                        count=int(entry.get("count", 0)),
                        name=entry.get("name", "").strip(),
                        set_code=(entry.get("set") or None),
                        number=(str(entry["number"]) if entry.get("number") else None),
                        category=category,
                    )
                )
        return cards

    @staticmethod
    def _external_ids(tid: str, player: dict) -> dict[str, str]:
        """
        Identify one standing so re-scraping the same tournament doesn't double-count it.

        ``placing`` is unique within a tournament, which makes
        ``(tournament_id, placing)`` a genuine key; the player name is carried too
        so undated/unplaced standings still distinguish themselves.

        :param tid: Tournament ID.
        :param player: The standings entry.
        :return: Stable source-native identifiers for this occurrence.
        """
        ids = {"tournament_id": str(tid)}
        if player.get("placing") is not None:
            ids["placing"] = str(player["placing"])
        if player.get("player"):
            ids["player"] = str(player["player"])
        return ids

    @staticmethod
    def _in_window(date: str | None, since: str | None, until: str | None) -> bool:
        """
        Whether a tournament date falls within an optional ``[since, until]`` window.

        :param date: Tournament ``date`` field (ISO timestamp), or None.
        :param since: Inclusive lower bound ``YYYY-MM-DD``, or None.
        :param until: Inclusive upper bound ``YYYY-MM-DD``, or None.
        :return: True if the date should be included.
        """
        day = (date or "")[:10]
        if not day:
            return since is None and until is None
        if since and day < since:
            return False
        return not (until and day > until)

    def _fetch_page(self, *, limit: int, page: int, fmt: str) -> list[dict]:
        """
        Fetch one page of the tournament index, newest first.

        :param limit: Tournaments per page. The API does not cap this.
        :param page: 1-based page number.
        :param fmt: Game format filter; skipped when empty.
        :return: The page's tournament objects (empty once paging runs out).
        """
        params: dict = {"game": "PTCG", "limit": limit, "page": page}
        if fmt:
            params["format"] = fmt.upper()
        return self.client.get_json(
            f"{API_BASE}/tournaments", params=params, headers=self._headers()
        )

    def _decks_from_tournament(
        self, t: dict, *, fmt: str, per_tournament: int
    ) -> Iterable[RawDeck]:
        """
        Yield the decklists published for one tournament, best finish first.

        :param t: A tournament object from the index.
        :param fmt: Fallback format label when the tournament omits one.
        :param per_tournament: Cap on decks taken; ``0`` means every published list.
        :return: An iterable of :class:`~scraper.models.RawDeck`.
        """
        tid = t.get("id")
        day = (t.get("date") or "")[:10]
        standings = self.client.get_json(
            f"{API_BASE}/tournaments/{tid}/standings", headers=self._headers()
        )
        taken = 0
        for player in standings:
            decklist = player.get("decklist")
            if not decklist:
                continue
            cards = self._cards_from_decklist(decklist)
            if not cards:
                continue
            deck_meta = player.get("deck") or {}
            archetype = (
                deck_meta.get("name") or player.get("player") or "limitless-deck"
            )
            yield RawDeck(
                source=self.name,
                archetype=archetype,
                cards=cards,
                url=f"https://play.limitlesstcg.com/tournament/{tid}/standings",
                fmt=(t.get("format") or fmt or "").lower() or None,
                record=_fmt_record(player.get("record")),
                event=t.get("name"),
                placing=player.get("placing"),
                event_date=day or None,
                external_ids=self._external_ids(tid, player),
            )
            taken += 1
            if per_tournament and taken >= per_tournament:
                break

    def iter_decks(
        self,
        *,
        limit: int = 20,
        fmt: str = "standard",
        per_tournament: int = 8,
        page: int = 1,
        max_pages: int = 1,
        max_decks: int | None = None,
        since: str | None = None,
        until: str | None = None,
        verbose: bool = False,
        **_kwargs,
    ) -> Iterable[RawDeck]:
        """Yield decklists from recent tournaments, walking the index page by page.

        The tournament index is paginated and ordered newest-first, so the walk
        moves backwards through time and stops as soon as it is provably done:
        when a page comes back empty or short (the index is exhausted), when every
        tournament on a page predates ``since``, or when ``max_decks`` is reached.

        Two knobs bound how much this yields, and they multiply — ``max_pages ×
        limit`` tournaments, each contributing up to ``per_tournament`` decks:

        :param limit: Tournaments per page, newest first. The API does not cap it.
        :param fmt: Game format filter (e.g. ``"standard"``).
        :param per_tournament: Cap on decks taken per tournament, best finish
            first. ``0`` takes every published list — note that a large event
            publishes one per player, so the tail is 1-4 / 0-6 finishes whose
            lists are far weaker training signal than the top tables.
        :param page: 1-based page to start from.
        :param max_pages: How many pages to walk from ``page``. ``0`` walks until
            the index runs out, which reaches back years — bound the run with
            ``since`` or ``max_decks`` rather than relying on it stopping soon.
        :param max_decks: Stop after yielding this many decks, if given.
        :param since: Keep only tournaments on/after this ``YYYY-MM-DD`` date;
            also ends the walk, since the index only gets older.
        :param until: Keep only tournaments on/before this ``YYYY-MM-DD`` date.
        :param verbose: Print a line per page as the walk progresses.
        :param _kwargs: Ignored extra source kwargs (shared CLI interface).
        :return: An iterable of :class:`~scraper.models.RawDeck`.
        """
        yielded = 0
        failures: list[str] = []
        current = max(1, page)
        pages_walked = 0

        while max_pages == 0 or pages_walked < max_pages:
            tournaments = self._fetch_page(limit=limit, page=current, fmt=fmt)
            pages_walked += 1
            if not tournaments:
                break  # index exhausted

            if verbose:
                span = f"{tournaments[0].get('date', '')[:10]}..{tournaments[-1].get('date', '')[:10]}"
                print(f"  page {current}: {len(tournaments)} tournaments ({span})")

            exhausted_window = False
            for t in tournaments:
                if not t.get("id"):
                    continue
                day = (t.get("date") or "")[:10]
                if since and day and day < since:
                    # Newest-first, so nothing later in the index can qualify.
                    exhausted_window = True
                    break
                if not self._in_window(t.get("date"), since, until):
                    continue  # newer than `until` (or undated with a window set)
                try:
                    decks = self._decks_from_tournament(
                        t, fmt=fmt, per_tournament=per_tournament
                    )
                    for deck in decks:
                        yield deck
                        yielded += 1
                        if max_decks is not None and yielded >= max_decks:
                            if failures:
                                raise _source_fetch_error(failures)
                            return
                except SourceFetchError:
                    raise
                except Exception as exc:  # noqa: BLE001 - retain other events
                    failures.append(
                        f"tournament {t.get('id')}: {type(exc).__name__}: {exc}"
                    )

            if exhausted_window or len(tournaments) < limit:
                break  # past the window, or that was the last page
            current += 1
        if failures:
            raise _source_fetch_error(failures)


def _source_fetch_error(failures: list[str]) -> SourceFetchError:
    sample = "; ".join(failures[:3])
    return SourceFetchError(
        f"{len(failures)} Limitless standings request(s) failed: {sample}"
    )
