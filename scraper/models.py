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
    event_date: str | None = None  # YYYY-MM-DD the event was held, if available
    # Source-native IDs pinning this occurrence down across re-scrapes, e.g.
    # {"tournament_id": "...", "placing": "3"}. Sources should populate something
    # stable here; without it, repeat occurrences are identified by their
    # descriptive fields alone (see scraper.manifest.Observation.key).
    external_ids: dict[str, str] = field(default_factory=dict)

    def total_cards(self) -> int:
        """
        :return: Total number of cards, summing each entry's copy ``count``.
        """
        return sum(c.count for c in self.cards)


@dataclass(frozen=True)
class CardSwap:
    """One auditable source-printing to competition-card substitution."""

    source_name: str
    source_set: str | None
    source_number: str | None
    count: int
    target_id: int
    target_name: str
    kind: str  # "variant"
    confidence: float
    rationale: str

    def to_json(self) -> dict[str, str | int | float | None]:
        """Return the stable representation stored on a manifest observation."""
        return {
            "source_name": self.source_name,
            "source_set": self.source_set,
            "source_number": self.source_number,
            "count": self.count,
            "target_id": self.target_id,
            "target_name": self.target_name,
            "kind": self.kind,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


@dataclass
class ResolvedDeck:
    """The result of resolving a RawDeck against the card index."""

    source_deck: RawDeck
    ids: list[int]  # expanded list of Card IDs (one entry per copy)
    unresolved_cards: list[RawCard] = field(default_factory=list)
    # Fuzzy substitutions applied, for auditing: (scraped name -> matched name, score).
    fuzzy_matches: list[tuple[str, str, float]] = field(default_factory=list)
    # Structured swaps applied, retained per source observation in the manifest.
    swaps: list[CardSwap] = field(default_factory=list)
    # Candidates rejected by post-resolution copy-count or ACE SPEC guards.
    swap_failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """
        :return: True if every scraped card resolved to a Card ID.
        """
        return not self.unresolved_cards
