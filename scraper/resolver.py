from __future__ import annotations

from .card_index import CardIndex, normalize_name
from .card_swapper import CardSwapper
from .models import RawCard, RawDeck, ResolvedDeck


def resolve_deck(
    raw: RawDeck, index: CardIndex, swapper: CardSwapper | None = None
) -> ResolvedDeck:
    """Map every card in ``raw`` to a Card ID, expanded one entry per copy.

    Cards that cannot be matched to any ID in the card database are collected
    into ``ResolvedDeck.unresolved_cards``; the caller decides whether to drop
    such decks. If ``swapper`` is given, a card that fails normal resolution
    gets one more chance against its hardcoded substitute before being given up
    on.

    :param raw: The scraped decklist to resolve.
    :param index: Card index used to match names/sets to Card IDs.
    :param swapper: Optional hardcoded-substitute table for otherwise-missing
        cards (e.g. staples absent from the card pool).
    :return: A :class:`ResolvedDeck` with the expanded Card IDs, any unresolved
        cards, the fuzzy substitutions applied, and the swaps applied.
    """
    matches = [index.match(card.name, card.set_code, card.number) for card in raw.cards]

    # A card already in the deck under its real name pins any evolution that
    # depends on it: a missing Pokémon must never be swapped for an unrelated
    # substitute while another card in the same deck evolves from it specifically.
    evolution_targets = {
        normalize_name(index.by_id[m.card_id].previous_stage)
        for m in matches
        if m.card_id is not None and index.by_id[m.card_id].previous_stage
    }

    ids: list[int] = []
    unresolved: list[RawCard] = []
    fuzzy: list[tuple[str, str, float]] = []
    swaps: list[tuple[str, str]] = []

    for card, result in zip(raw.cards, matches):
        if (
            result.card_id is None
            and swapper is not None
            and normalize_name(card.name) not in evolution_targets
        ):
            swap = swapper.lookup(card.name)
            if swap is not None:
                sub_name, sub_id = swap
                swaps.append((card.name, sub_name))
                result = result._replace(card_id=sub_id, method="swap", matched_name=sub_name)

        if result.card_id is None:
            unresolved.append(card)
            continue
        if result.method == "fuzzy":
            matched = index.by_id[result.card_id].name
            fuzzy.append((card.name, matched, result.score or 0.0))
        ids.extend([result.card_id] * max(0, card.count))

    return ResolvedDeck(
        source_deck=raw, ids=ids, unresolved_cards=unresolved, fuzzy_matches=fuzzy, swaps=swaps
    )
