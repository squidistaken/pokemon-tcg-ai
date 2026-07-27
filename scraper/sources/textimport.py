from __future__ import annotations

import os
import re
from collections.abc import Iterable

from ..models import RawCard, RawDeck
from .base import DeckSource

_SECTION_RE = re.compile(
    r"^\s*(pok[eé]mon|trainer|energy|total|cards?)\b.*:", re.IGNORECASE
)
_BULLET_RE = re.compile(r"^\s*[-*•]\s*")
_COUNT_RE = re.compile(r"^\s*(\d+)\s*x?\s+(.+?)\s*$")
_PAREN_SET_RE = re.compile(r"\s*\(([A-Za-z]{2,5})[-\s]?([0-9A-Za-z]+)?\)\s*$")
_BARE_SET_RE = re.compile(r"\s+([A-Z]{2,5})(?:[-\s]([0-9]+[A-Za-z]*))?\s*$")


def parse_card_line(line: str) -> RawCard | None:
    """Parse a single decklist line into a RawCard, or None if it isn't one.

    :param line: One line of decklist text.
    :return: The parsed :class:`~scraper.models.RawCard`, or None for section
        headers, blanks, and other non-card lines.
    """
    line = _BULLET_RE.sub("", line).strip()
    if not line or _SECTION_RE.match(line):
        return None
    m = _COUNT_RE.match(line)
    if not m:
        return None
    count = int(m.group(1))
    rest = m.group(2).strip()

    set_code: str | None = None
    number: str | None = None

    paren = _PAREN_SET_RE.search(rest)
    if paren:
        set_code, number = paren.group(1).upper(), paren.group(2)
        rest = rest[: paren.start()].strip()
    else:
        bare = _BARE_SET_RE.search(rest)
        if bare:
            set_code, number = bare.group(1).upper(), bare.group(2)
            rest = rest[: bare.start()].strip()

    name = rest.strip()
    if not name:
        return None
    return RawCard(count=count, name=name, set_code=set_code, number=number)


def parse_decklist_text(text: str) -> list[RawCard]:
    """Parse a whole decklist blob into RawCards.

    :param text: The full decklist text (multiple lines).
    :return: The RawCards for every card line found.
    """
    cards: list[RawCard] = []
    for line in text.splitlines():
        card = parse_card_line(line)
        if card is not None:
            cards.append(card)
    return cards


class TextSource(DeckSource):
    """Read a decklist from a file (or raw text) in the universal format."""

    name = "text"

    def iter_decks(
        self,
        *,
        input: str | None = None,
        text: str | None = None,
        archetype: str | None = None,
        fmt: str | None = None,
        **_kwargs,
    ) -> Iterable[RawDeck]:
        """
        Yield the single deck parsed from a file or a raw text blob.

        :param input: Path to a decklist text file (used when ``text`` is None).
        :param text: Raw decklist text; takes precedence over ``input``.
        :param archetype: Deck name; defaults to the input file's stem.
        :param fmt: Optional format label recorded on the deck.
        :param _kwargs: Ignored extra source kwargs (shared CLI interface).
        :return: An iterable yielding one :class:`~scraper.models.RawDeck`.
        :raises ValueError: If neither ``input`` nor ``text`` is given.
        """
        if text is None:
            if not input:
                raise ValueError("TextSource needs --input <file> or text=")
            with open(input, encoding="utf-8") as f:
                text = f.read()
            if archetype is None:
                archetype = os.path.splitext(os.path.basename(input))[0]
        cards = parse_decklist_text(text)
        yield RawDeck(
            source=self.name,
            archetype=archetype or "imported-deck",
            cards=cards,
            url=None,
            fmt=fmt,
        )
