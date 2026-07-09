"""Resolve a scraped :class:`RawDeck` into a list of Card IDs."""

from __future__ import annotations

from .card_index import CardIndex
from .models import RawCard, RawDeck, ResolvedDeck


def resolve_deck(raw: RawDeck, index: CardIndex) -> ResolvedDeck:
    """Map every card in ``raw`` to a Card ID, expanded one entry per copy.

    Cards that cannot be matched to any ID in the card database are collected
    into ``ResolvedDeck.unresolved`` (leaving the deck ``ok == False``); the
    caller decides whether to drop such decks.
    """
    ids: list[int] = []
    unresolved: list[RawCard] = []
    fuzzy: list[tuple[str, str, float]] = []
    for card in raw.cards:
        result = index.match(card.name, card.set_code, card.number)
        if result.card_id is None:
            unresolved.append(card)
            continue
        if result.method == "fuzzy":
            matched = index.by_id[result.card_id].name
            fuzzy.append((card.name, matched, result.score or 0.0))
        ids.extend([result.card_id] * max(0, card.count))
    return ResolvedDeck(raw=raw, ids=ids, unresolved=unresolved, fuzzy=fuzzy)
