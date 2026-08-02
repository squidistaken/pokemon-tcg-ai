"""Tests for Bulbapedia decklist identity parsing and category pagination."""

from __future__ import annotations

import pytest

from scraper.sources.base import SourceFetchError
from scraper.sources.bulbapedia import BulbapediaSource


class FakeClient:
    """Serve canned MediaWiki category and parse responses."""

    def __init__(self, category_responses: dict[str | None, dict]):
        self.category_responses = category_responses
        self.category_requests: list[dict] = []
        self.parse_requests: list[str] = []

    def get_json(self, _url: str, params: dict | None = None, headers=None):  # noqa: ARG002
        params = params or {}
        if params.get("action") == "query":
            self.category_requests.append(params)
            return self.category_responses[params.get("cmcontinue")]
        page = params["page"]
        self.parse_requests.append(page)
        return {"parse": {"text": {"*": _deck_html(page)}}}


def _deck_html(name: str) -> str:
    return f"""
    <table>
      <tr><th>Quantity</th><th>Card</th></tr>
      <tr><td>1</td><td>{name}</td></tr>
    </table>
    """


def _members(*titles: str, continuation: str | None = None) -> dict:
    response: dict = {
        "query": {"categorymembers": [{"title": title} for title in titles]}
    }
    if continuation is not None:
        response["continue"] = {"continue": "-||", "cmcontinue": continuation}
    return response


def test_card_links_preserve_expansion_and_collection_number():
    html = """
    <table>
      <tr><th>Quantity</th><th>Card</th><th>Type</th></tr>
      <tr>
        <td>2×</td>
        <td><a href="/wiki/Charmeleon_(Obsidian_Flames_27)"
               title="Charmeleon (Obsidian Flames 27)">Charmeleon</a></td>
        <td>Pokémon</td>
      </tr>
      <tr>
        <td>4×</td>
        <td><a href="/wiki/Professor_Oak_(TCG)"
               title="Professor Oak (TCG)">Professor Oak</a></td>
        <td>Trainer</td>
      </tr>
    </table>
    """

    decks = BulbapediaSource._parse_decklist_tables(html)  # noqa: SLF001

    assert len(decks) == 1
    assert (decks[0][0].name, decks[0][0].set_code, decks[0][0].number) == (
        "Charmeleon",
        "Obsidian Flames",
        "27",
    )
    assert (decks[0][1].set_code, decks[0][1].number) == (None, None)


def test_category_pagination_stops_at_configured_page_limit():
    client = FakeClient(
        {
            None: _members("Alpha", "Beta", continuation="next"),
            "next": _members("Gamma", "Delta", continuation="unused"),
        }
    )
    source = BulbapediaSource(client)  # type: ignore[arg-type]

    pages = source._category_pages("Decks", max_pages=3)  # noqa: SLF001

    assert pages == ["Alpha", "Beta", "Gamma"]
    assert [request["cmlimit"] for request in client.category_requests] == [3, 1]
    assert [request.get("cmcontinue") for request in client.category_requests] == [
        None,
        "next",
    ]


def test_category_pagination_zero_walks_until_continuation_is_exhausted():
    client = FakeClient(
        {
            None: _members("Alpha", "Beta", continuation="second"),
            "second": _members("Beta", "Gamma", continuation="third"),
            "third": _members("Delta"),
        }
    )
    source = BulbapediaSource(client)  # type: ignore[arg-type]

    pages = source._category_pages("Decks", max_pages=0)  # noqa: SLF001

    assert pages == ["Alpha", "Beta", "Gamma", "Delta"]
    assert [request.get("cmcontinue") for request in client.category_requests] == [
        None,
        "second",
        "third",
    ]


def test_explicit_pages_are_first_and_deduplicated_against_category_pages():
    client = FakeClient({None: _members("Category Page", "Explicit Page")})
    source = BulbapediaSource(client)  # type: ignore[arg-type]

    decks = list(
        source.iter_decks(
            pages=["Explicit Page", "Explicit Page"],
            category="Decks",
            bulbapedia_max_pages=0,
        )
    )

    assert [deck.archetype for deck in decks] == ["Explicit Page", "Category Page"]
    assert client.parse_requests == ["Explicit Page", "Category Page"]


def test_page_request_failures_are_reported_after_partial_results():
    class PartlyBrokenClient(FakeClient):
        def get_json(self, url: str, params=None, headers=None):
            if (params or {}).get("page") == "Broken":
                raise RuntimeError("429")
            return super().get_json(url, params=params, headers=headers)

    client = PartlyBrokenClient({})
    source = BulbapediaSource(client)  # type: ignore[arg-type]
    decks = iter(source.iter_decks(pages=["Working", "Broken"]))

    assert next(decks).archetype == "Working"
    with pytest.raises(SourceFetchError, match="Broken.*429"):
        next(decks)


def test_all_valid_pages_without_deck_tables_are_reported():
    class NoDeckClient(FakeClient):
        def get_json(self, url: str, params=None, headers=None):
            if (params or {}).get("action") == "parse":
                return {"parse": {"text": {"*": "<p>Valid article</p>"}}}
            return super().get_json(url, params=params, headers=headers)

    source = BulbapediaSource(NoDeckClient({}))  # type: ignore[arg-type]

    with pytest.raises(SourceFetchError, match="no recognized decklist tables.*2"):
        list(source.iter_decks(pages=["Article A", "Article B"]))


def test_page_without_deck_table_is_allowed_when_another_page_has_one():
    class MixedClient(FakeClient):
        def get_json(self, url: str, params=None, headers=None):
            if (params or {}).get("page") == "Article":
                return {"parse": {"text": {"*": "<p>Valid article</p>"}}}
            return super().get_json(url, params=params, headers=headers)

    source = BulbapediaSource(MixedClient({}))  # type: ignore[arg-type]

    decks = list(source.iter_decks(pages=["Article", "Deck Page"]))

    assert [deck.archetype for deck in decks] == ["Deck Page"]
