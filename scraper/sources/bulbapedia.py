"""Bulbapedia deck source (bulbapedia.bulbagarden.net) via the MediaWiki API.

Best-effort wiki scraper. Bulbapedia decklist pages render a table with a
``Quantity | Card | ...`` header and rows like ``3× | Vulpix``. We parse those
into RawCards by name (no set/number, so resolution leans on the reprint
fallback). Decks that reference cards outside our card database are dropped
downstream -- expected, since many wiki decks are older sets.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from bs4 import BeautifulSoup

from ..http import HttpClient
from ..models import RawCard, RawDeck
from .base import DeckSource

API = "https://bulbapedia.bulbagarden.net/w/api.php"
_QTY_RE = re.compile(r"(\d+)")


class BulbapediaSource(DeckSource):
    name = "bulbapedia"

    def __init__(self, client: HttpClient | None = None):
        # Bulbapedia asks for a descriptive UA and gentle rates.
        self.client = client or HttpClient(min_interval=2.0)

    def _fetch_html(self, page: str) -> str | None:
        data = self.client.get_json(
            API,
            params={"action": "parse", "page": page, "format": "json", "prop": "text"},
            headers={"Accept": "application/json"},
        )
        if "error" in data:
            return None
        return data.get("parse", {}).get("text", {}).get("*")

    def _category_pages(self, category: str, limit: int) -> list[str]:
        data = self.client.get_json(
            API,
            params={
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmlimit": limit,
                "format": "json",
            },
        )
        members = data.get("query", {}).get("categorymembers", [])
        return [m["title"] for m in members]

    @staticmethod
    def _parse_decklist_tables(html: str) -> list[list[RawCard]]:
        soup = BeautifulSoup(html, "lxml")
        decks: list[list[RawCard]] = []
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if not rows:
                continue
            header = [c.get_text(" ", strip=True).lower() for c in rows[0].find_all(["th", "td"])]
            if not ("quantity" in header and "card" in header):
                continue
            qty_i = header.index("quantity")
            card_i = header.index("card")
            cards: list[RawCard] = []
            for tr in rows[1:]:
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
                if len(cells) <= max(qty_i, card_i):
                    continue
                m = _QTY_RE.search(cells[qty_i])
                name = cells[card_i].strip()
                if not m or not name:
                    continue
                cards.append(RawCard(count=int(m.group(1)), name=name))
            if cards:
                decks.append(cards)
        return decks

    def iter_decks(
        self,
        *,
        pages: list[str] | None = None,
        category: str | None = None,
        limit: int = 20,
        **_kwargs,
    ) -> Iterable[RawDeck]:
        titles = list(pages or [])
        if category:
            titles.extend(self._category_pages(category, limit))
        for page in titles:
            try:
                html = self._fetch_html(page)
            except Exception:
                continue
            if not html:
                continue
            tables = self._parse_decklist_tables(html)
            for i, cards in enumerate(tables):
                suffix = "" if len(tables) == 1 else f" ({i + 1})"
                yield RawDeck(
                    source=self.name,
                    archetype=f"{page}{suffix}",
                    cards=cards,
                    url=f"https://bulbapedia.bulbagarden.net/wiki/{page.replace(' ', '_')}",
                    fmt=None,
                )
