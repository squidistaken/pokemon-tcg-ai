from __future__ import annotations

from collections import Counter

from .card_index import CardIndex, normalize_name, normalize_number
from .card_swapper import CardSwapper
from .models import CardSwap, RawCard, RawDeck, ResolvedDeck


def _variant_swap(card: RawCard, card_id: int, index: CardIndex) -> CardSwap:
    return CardSwap(
        source_name=card.name,
        source_set=card.set_code,
        source_number=normalize_number(card.number),
        count=max(0, card.count),
        target_id=card_id,
        target_name=index.by_id[card_id].name,
        kind="variant",
        confidence=1.0,
        rationale="unique competition printing with the same card name",
    )


def resolve_deck(
    raw: RawDeck, index: CardIndex, swapper: CardSwapper | None = None
) -> ResolvedDeck:
    """Map every card in raw to a Card ID, expanded one entry per copy.

    Cards that cannot be matched to any ID in the card database are collected
    into ResolvedDeck.unresolved_cards; the caller decides whether to drop such
    decks. If swapper is given, ambiguous cards get one more chance against
    gameplay profiles for competition printings with the same name.

    Candidate lines are assigned together so the final choice respects the
    four-copy and ACE SPEC limits. A source card line is never split.

    :param raw: The scraped decklist to resolve.
    :param index: Card index used to match names/sets to Card IDs.
    :param swapper: Optional same-name gameplay-profile matcher.
    :return: A ResolvedDeck with Card IDs and resolution provenance.
    """
    matches = [index.match(card.name, card.set_code, card.number) for card in raw.cards]

    line_ids: dict[int, int] = {}
    name_counts: Counter[str] = Counter()
    ace_count = 0
    fuzzy: list[tuple[str, str, float]] = []
    swaps: list[CardSwap] = []
    pending: list[tuple[int, RawCard, tuple[CardSwap, ...]]] = []

    for position, (card, result) in enumerate(zip(raw.cards, matches, strict=True)):
        if result.card_id is None:
            candidates = swapper.resolve(card) if swapper is not None else ()
            pending.append((position, card, candidates))
            continue

        line_ids[position] = result.card_id
        info = index.by_id[result.card_id]
        copies = max(0, card.count)
        name_counts[normalize_name(info.name)] += copies
        ace_count += copies if info.is_ace_spec else 0
        if result.method == "fuzzy":
            fuzzy.append((card.name, info.name, result.score or 0.0))
        elif result.method == "variant":
            swaps.append(_variant_swap(card, result.card_id, index))

    assignments: dict[int, CardSwap] = {}
    assignable = [item for item in pending if item[2]]

    def assign(offset: int, current_ace_count: int) -> bool:
        if offset == len(assignable):
            return True
        position, card, candidates = assignable[offset]
        copies = max(0, card.count)
        for candidate in candidates:
            target = index.by_id[candidate.target_id]
            target_name = normalize_name(target.name)
            if not target.is_basic_energy and name_counts[target_name] + copies > 4:
                continue
            if target.is_ace_spec and current_ace_count + copies > 1:
                continue
            assignments[position] = candidate
            name_counts[target_name] += copies
            if assign(
                offset + 1,
                current_ace_count + (copies if target.is_ace_spec else 0),
            ):
                return True
            name_counts[target_name] -= copies
            assignments.pop(position, None)
        return False

    assigned = assign(0, ace_count)
    failures: list[str] = []
    if assignable and not assigned:
        failures.append(
            "no complete swap assignment satisfies copy-count and ACE SPEC guards"
        )

    unresolved_positions = {position for position, _, _ in pending}
    if assigned:
        for position, _, _ in assignable:
            candidate = assignments[position]
            line_ids[position] = candidate.target_id
            swaps.append(candidate)
            unresolved_positions.remove(position)

    ids: list[int] = []
    unresolved: list[RawCard] = []
    for position, card in enumerate(raw.cards):
        if position in unresolved_positions:
            unresolved.append(card)
            continue
        ids.extend([line_ids[position]] * max(0, card.count))

    return ResolvedDeck(
        source_deck=raw,
        ids=ids,
        unresolved_cards=unresolved,
        fuzzy_matches=fuzzy,
        swaps=swaps,
        swap_failures=failures,
    )
