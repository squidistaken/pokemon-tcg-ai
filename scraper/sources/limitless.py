from __future__ import annotations

import os
from collections.abc import Iterable

from ..http import HttpClient
from ..models import RawCard, RawDeck
from .base import DeckSource

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

    def iter_decks(
        self,
        *,
        limit: int = 20,
        fmt: str = "standard",
        per_tournament: int = 8,
        page: int = 1,
        since: str | None = None,
        until: str | None = None,
        **_kwargs,
    ) -> Iterable[RawDeck]:
        """Yield decklists from recent tournaments, optionally within a date window.

        :param limit: Maximum tournaments to fetch per page (most recent first).
        :param fmt: Game format filter (e.g. ``"standard"``).
        :param per_tournament: Maximum decks taken per tournament.
        :param page: 1-based page of results (each page is ``limit`` tournaments).
        :param since: Keep only tournaments on/after this ``YYYY-MM-DD`` date.
        :param until: Keep only tournaments on/before this ``YYYY-MM-DD`` date.
        :param _kwargs: Ignored extra source kwargs (shared CLI interface).
        :return: An iterable of :class:`~scraper.models.RawDeck`.
        """
        params = {"game": "PTCG", "limit": limit, "page": page}
        if fmt:
            params["format"] = fmt.upper()
        tournaments = self.client.get_json(
            f"{API_BASE}/tournaments", params=params, headers=self._headers()
        )
        for t in tournaments:
            tid = t.get("id")
            if not tid:
                continue
            day = (t.get("date") or "")[:10]
            if since and day and day < since:
                break  # newest-first: everything after is older than the window
            if not self._in_window(t.get("date"), since, until):
                continue  # newer than `until` (or undated with a window set)
            try:
                standings = self.client.get_json(
                    f"{API_BASE}/tournaments/{tid}/standings", headers=self._headers()
                )
            except Exception:  # noqa: BLE001, S112
                continue
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
                url = f"https://play.limitlesstcg.com/tournament/{tid}/standings"
                yield RawDeck(
                    source=self.name,
                    archetype=archetype,
                    cards=cards,
                    url=url,
                    fmt=(t.get("format") or fmt or "").lower() or None,
                    record=_fmt_record(player.get("record")),
                    event=t.get("name"),
                    placing=player.get("placing"),
                )
                taken += 1
                if taken >= per_tournament:
                    break
