from __future__ import annotations

from collections import Counter

from .card_index import CardIndex

DECK_SIZE = 60
DECK_SAME_CARD_MAX = 4


def validate(ids: list[int], index: CardIndex) -> list[str]:
    """Return a list of rule violations; an empty list means the deck is legal.

    Rules (in engine order):
      1. exactly 60 cards;
      2. every ID exists in the card database;
      3. <= 4 copies of the same card name (Basic Energy exempt);
      4. at least one Basic Pokémon;
      5. at most one ACE SPEC card.

    :param ids: The deck's Card IDs (one entry per copy).
    :param index: Card index used to look up per-card legality flags.
    :return: Human-readable rule violations; empty if the deck is legal.
    """
    errors: list[str] = []

    if len(ids) != DECK_SIZE:
        errors.append(f"deck has {len(ids)} cards, must be exactly {DECK_SIZE}")

    unknown = [cid for cid in ids if cid not in index.by_id]
    if unknown:
        errors.append(f"unknown card IDs not in card database: {sorted(set(unknown))}")

    # Remaining checks only make sense for IDs we actually know.
    known = [cid for cid in ids if cid in index.by_id]

    name_counts: Counter[str] = Counter()
    ace_spec = 0
    has_basic = False
    for cid in known:
        info = index.by_id[cid]
        name_counts[info.name] += 1
        if info.is_ace_spec:
            ace_spec += 1
        if info.is_basic_pokemon:
            has_basic = True

    for name, count in name_counts.items():
        if count > DECK_SAME_CARD_MAX:
            # Look up whether this name is a Basic Energy (exempt from the cap).
            sample = next(cid for cid in known if index.by_id[cid].name == name)
            if not index.by_id[sample].is_basic_energy:
                errors.append(f"{count} copies of {name!r} (max {DECK_SAME_CARD_MAX})")

    if not has_basic:
        errors.append("deck has no Basic Pokémon")

    if ace_spec > 1:
        errors.append(f"{ace_spec} ACE SPEC cards (max 1)")

    return errors
