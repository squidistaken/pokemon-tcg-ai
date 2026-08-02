from __future__ import annotations

import re
from collections.abc import Iterable

from bs4 import BeautifulSoup

from ..http import HttpClient
from ..models import RawCard, RawDeck
from .base import DeckSource, SourceFetchError

API = "https://bulbapedia.bulbagarden.net/w/api.php"
_QTY_RE = re.compile(r"(\d+)")
_PRINTING_RE = re.compile(r"^.+ \((?P<set>.+) (?P<number>[A-Za-z]*\d+[A-Za-z]*)\)$")


class BulbapediaSource(DeckSource):
    """Deck source that parses Bulbapedia decklist tables via the MediaWiki API."""

    name = "bulbapedia"

    def __init__(self, client: HttpClient | None = None):
        """
        :param client: HTTP client to use; a gently rate-limited one by default,
            since Bulbapedia asks for a descriptive UA and low request rates.
        """
        self.client = client or HttpClient(min_interval=2.0)

    def _fetch_html(self, page: str) -> str | None:
        """
        Fetch a wiki page's rendered HTML via the MediaWiki parse API.

        :param page: Wiki page title.
        :return: The page's HTML, or None if the API reported an error.
        """
        data = self.client.get_json(
            API,
            params={"action": "parse", "page": page, "format": "json", "prop": "text"},
            headers={"Accept": "application/json"},
        )
        if "error" in data:
            return None
        return data.get("parse", {}).get("text", {}).get("*")

    def _category_pages(self, category: str, max_pages: int) -> list[str]:
        """
        List page titles belonging to a wiki category.

        :param category: Category name.
        :param max_pages: Maximum number of member pages to return. ``0`` walks
            the category until MediaWiki stops returning a continuation token.
        :return: The member page titles.
        """
        if max_pages < 0:
            raise ValueError("max_pages must be non-negative")

        titles: list[str] = []
        seen_titles: set[str] = set()
        seen_continuations: set[str] = set()
        continuation: str | None = None
        while max_pages == 0 or len(titles) < max_pages:
            remaining = max_pages - len(titles) if max_pages else 500
            params: dict[str, str | int] = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmlimit": min(remaining, 500),
                "format": "json",
            }
            if continuation is not None:
                params["cmcontinue"] = continuation
            data = self.client.get_json(API, params=params)
            members = data.get("query", {}).get("categorymembers", [])
            for member in members:
                title = member.get("title")
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                titles.append(title)
                if max_pages and len(titles) == max_pages:
                    return titles

            next_continuation = data.get("continue", {}).get("cmcontinue")
            if not next_continuation or next_continuation in seen_continuations:
                break
            seen_continuations.add(next_continuation)
            continuation = next_continuation
        return titles

    @staticmethod
    def _parse_decklist_tables(html: str) -> list[list[RawCard]]:
        """
        Extract decklists from a page's ``Quantity | Card`` tables.

        :param html: Rendered wiki page HTML.
        :return: One RawCard list per decklist table found.
        """
        soup = BeautifulSoup(html, "lxml")
        decks: list[list[RawCard]] = []
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if not rows:
                continue
            header = [
                c.get_text(" ", strip=True).lower()
                for c in rows[0].find_all(["th", "td"])
            ]
            if not ("quantity" in header and "card" in header):
                continue
            qty_i = header.index("quantity")
            card_i = header.index("card")
            cards: list[RawCard] = []
            for tr in rows[1:]:
                cell_nodes = tr.find_all(["td", "th"])
                cells = [cell.get_text(" ", strip=True) for cell in cell_nodes]
                if len(cells) <= max(qty_i, card_i):
                    continue
                m = _QTY_RE.search(cells[qty_i])
                name = cells[card_i].strip()
                if not m or not name:
                    continue
                set_name: str | None = None
                number: str | None = None
                link = cell_nodes[card_i].find("a", title=True)
                printing = _PRINTING_RE.match(str(link.get("title"))) if link else None
                if printing:
                    set_name = printing.group("set")
                    number = printing.group("number")
                cards.append(
                    RawCard(
                        count=int(m.group(1)),
                        name=name,
                        set_code=set_name,
                        number=number,
                    )
                )
            if cards:
                decks.append(cards)
        return decks

    def iter_decks(
        self,
        *,
        pages: list[str] | None = None,
        category: str | None = None,
        limit: int = 20,
        bulbapedia_max_pages: int | None = None,
        **_kwargs,
    ) -> Iterable[RawDeck]:
        """
        Yield decks parsed from the given wiki pages and/or a category.

        :param pages: Explicit wiki page titles to scrape.
        :param category: Optional category whose member pages are added.
        :param limit: Legacy maximum category pages used when
            ``bulbapedia_max_pages`` is not provided.
        :param bulbapedia_max_pages: Maximum category pages to enumerate. ``0``
            traverses the category until exhausted.
        :param _kwargs: Ignored extra source kwargs (shared CLI interface).
        :return: An iterable of :class:`~scraper.models.RawDeck`.
        """
        titles = list(pages or [])
        if category:
            max_pages = limit if bulbapedia_max_pages is None else bulbapedia_max_pages
            titles.extend(self._category_pages(category, max_pages))
        titles = list(dict.fromkeys(titles))
        failures: list[str] = []
        deck_count = 0
        for page in titles:
            try:
                html = self._fetch_html(page)
            except Exception as exc:  # noqa: BLE001 - keep partial source results
                failures.append(f"{page}: {type(exc).__name__}: {exc}")
                continue
            if not html:
                failures.append(f"{page}: API returned no page HTML")
                continue
            tables = self._parse_decklist_tables(html)
            for i, cards in enumerate(tables):
                deck_count += 1
                suffix = "" if len(tables) == 1 else f" ({i + 1})"
                yield RawDeck(
                    source=self.name,
                    archetype=f"{page}{suffix}",
                    cards=cards,
                    url=f"https://bulbapedia.bulbagarden.net/wiki/{page.replace(' ', '_')}",
                    fmt=None,
                    # The wiki gives no event, record, placing or date, so page +
                    # table index is the only thing distinguishing two decklists on
                    # one page — without it they'd look like the same occurrence.
                    external_ids={"page": page, "table_index": str(i)},
                )
        if failures:
            sample = "; ".join(failures[:3])
            raise SourceFetchError(
                f"{len(failures)} Bulbapedia page request(s) failed: {sample}"
            )
        if titles and deck_count == 0:
            raise SourceFetchError(
                f"no recognized decklist tables across {len(titles)} "
                "Bulbapedia page(s)"
            )
