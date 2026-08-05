"""Deck source registry."""

from __future__ import annotations

from .base import DeckSource
from .bulbapedia import BulbapediaSource
from .limitless import LimitlessSource
from .textimport import TextSource, parse_decklist_text

SOURCES: dict[str, type[DeckSource]] = {
    LimitlessSource.name: LimitlessSource,
    BulbapediaSource.name: BulbapediaSource,
    TextSource.name: TextSource,
}
NETWORK_SOURCES: dict[str, type[DeckSource]] = {
    LimitlessSource.name: LimitlessSource,
    BulbapediaSource.name: BulbapediaSource,
}

__all__ = [
    "NETWORK_SOURCES",
    "SOURCES",
    "BulbapediaSource",
    "DeckSource",
    "LimitlessSource",
    "TextSource",
    "parse_decklist_text",
]
