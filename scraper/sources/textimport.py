"""Parse decklists from the universal "count Name SET Number" export text.

This is the offline path (no network) and the shared line parser reused by the
LimitlessTCG HTML fallback. Handles the common forms produced by the Limitless
Deck Builder, PTCG Live, and wiki bullet lists::

    Pokémon: 6
    4 Dragapult ex TWM 130
    2 Kyogre MEG 45
    Trainer: 30
    * 4 Iono PAL 185
    Energy: 15
    15 Basic Water Energy SVE 3
    4 Water Energy
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable

from ..models import RawCard, RawDeck
from .base import DeckSource

# Section headers like "Pokémon: 6" / "Trainer:" / "Energy: 15" / "Total: 60".
_SECTION_RE = re.compile(
    r"^\s*(pok[eé]mon|trainer|energy|total|cards?)\b.*:", re.IGNORECASE
)
# Leading bullet / list markers to strip.
_BULLET_RE = re.compile(r"^\s*[-*•]\s*")
# A card line: "<count> <name...> [SET [NUMBER]]" with optional (SET-NUMBER).
_COUNT_RE = re.compile(r"^\s*(\d+)\s*x?\s+(.+?)\s*$")
_PAREN_SET_RE = re.compile(r"\s*\(([A-Za-z]{2,5})[-\s]?([0-9A-Za-z]+)?\)\s*$")
# A trailing SET code (2-5 uppercase letters), optionally followed by a number.
# The number is optional so hand-typed/wiki lines like "Mega Signal MEG" work;
# a lone lowercase/mixed-case trailing word (e.g. "Iono") is left in the name.
_BARE_SET_RE = re.compile(r"\s+([A-Z]{2,5})(?:[-\s]([0-9]+[A-Za-z]*))?\s*$")


def parse_card_line(line: str) -> RawCard | None:
    """Parse a single decklist line into a RawCard, or None if it isn't one."""
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
    """Parse a whole decklist blob into RawCards."""
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
        input: str | None = None,  # noqa: A002 - matches CLI flag name
        text: str | None = None,
        archetype: str | None = None,
        fmt: str | None = None,
        **_kwargs,
    ) -> Iterable[RawDeck]:
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
