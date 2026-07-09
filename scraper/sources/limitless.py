"""LimitlessTCG deck source (play.limitlesstcg.com).

Primary path is the public JSON API (no key required for public tournaments):
``/api/tournaments`` then ``/api/tournaments/{id}/standings``. Each standing's
``decklist`` field groups ``pokemon``/``trainer``/``energy`` entries shaped as
``{count, name, set, number}`` -- a direct map to :class:`RawCard`.

If ``LIMITLESS_API_KEY`` is set it is sent as ``X-Access-Key`` (needed for some
private/organizer data); public reads work without it.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from ..http import HttpClient
from ..models import RawCard, RawDeck
from .base import DeckSource

API_BASE = "https://play.limitlesstcg.com/api"


def _fmt_record(record) -> str | None:
    if isinstance(record, dict):
        w, l, t = record.get("wins"), record.get("losses"), record.get("ties")
        if w is not None:
            return f"{w}-{l}-{t}"
    return None


class LimitlessSource(DeckSource):
    name = "limitless"

    def __init__(self, client: HttpClient | None = None):
        self.client = client or HttpClient(min_interval=1.0)
        self.api_key = os.environ.get("LIMITLESS_API_KEY")

    def _headers(self) -> dict:
        return {"X-Access-Key": self.api_key} if self.api_key else {}

    def _cards_from_decklist(self, decklist: dict) -> list[RawCard]:
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

    def iter_decks(
        self,
        *,
        limit: int = 20,
        fmt: str = "standard",
        per_tournament: int = 8,
        **_kwargs,
    ) -> Iterable[RawDeck]:
        """Yield decklists from the most recent tournaments.

        ``limit`` caps the number of tournaments scanned; ``per_tournament``
        caps decks taken from each (top placings first).
        """
        params = {"game": "PTCG", "limit": limit}
        if fmt:
            params["format"] = fmt.upper()
        tournaments = self.client.get_json(
            f"{API_BASE}/tournaments", params=params, headers=self._headers()
        )
        for t in tournaments:
            tid = t.get("id")
            if not tid:
                continue
            try:
                standings = self.client.get_json(
                    f"{API_BASE}/tournaments/{tid}/standings", headers=self._headers()
                )
            except Exception:
                continue  # skip tournaments whose standings fail to load
            taken = 0
            for player in standings:
                decklist = player.get("decklist")
                if not decklist:
                    continue
                cards = self._cards_from_decklist(decklist)
                if not cards:
                    continue
                deck_meta = player.get("deck") or {}
                archetype = deck_meta.get("name") or player.get("player") or "limitless-deck"
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
