"""Data models shared across the scraper module."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RawCard:
    """A single card line as scraped from a source, before ID resolution.
    """

    count: int
    name: str
    set_code: str | None = None
    number: str | None = None
    category: str | None = (
        None  # "pokemon" | "trainer" | "energy" if the source groups them
    )


@dataclass
class RawDeck:
    """A decklist as scraped from a source, before ID resolution."""

    source: str
    archetype: str  # human name, becomes the deck file slug
    cards: list[RawCard] = field(default_factory=list)
    url: str | None = None
    fmt: str | None = None  # e.g. "standard"
    record: str | None = None  # e.g. "9-0-1", if available
    event: str | None = None  # tournament / page name, if available
    placing: int | None = None  # finishing position, if available

    def total_cards(self) -> int:
        """
        :return: Total number of cards, summing each entry's copy ``count``.
        """
        return sum(c.count for c in self.cards)


@dataclass
class ResolvedDeck:
    """The result of resolving a RawDeck against the card index."""

    raw: RawDeck
    ids: list[int]  # expanded list of Card IDs (one entry per copy)
    unresolved: list[RawCard] = field(default_factory=list)
    # Fuzzy substitutions applied, for auditing: (scraped name -> matched name, score).
    fuzzy: list[tuple[str, str, float]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """
        :return: True if every scraped card resolved to a Card ID.
        """
        return not self.unresolved
