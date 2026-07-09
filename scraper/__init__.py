"""Independent deck-scraping module for the Pokémon TCG AI Battle Challenge.

Pulls real decklists from the internet (LimitlessTCG, Bulbapedia) and from
pasted text, maps each card to our internal ``Card ID`` (from ``EN_Card_Data.csv``),
validates the deck against the engine's legality rules, and saves each as a
60-line deck CSV in ``decks/`` with source metadata in ``decks/manifest.json``.

This package is self-contained: it does not import from ``cg`` or ``main`` and
has no effect on the competition agent at runtime.
"""

from .models import RawCard, RawDeck, ResolvedDeck
from .card_index import CardIndex
from .resolver import resolve_deck
from .validator import validate

__all__ = [
    "RawCard",
    "RawDeck",
    "ResolvedDeck",
    "CardIndex",
    "resolve_deck",
    "validate",
]
