from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .card_index import CardIndex, normalize_name

DECK_SIZE = 60
DECK_SAME_CARD_MAX = 4


@dataclass
class ValidationResult:
    """Outcome of :func:`validate_deck`: hard errors plus non-fatal warnings."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """
        :return: True if the deck has no legality errors (warnings don't count).
        """
        return not self.errors


def validate_deck(
    ids: list[int],
    index: CardIndex,
    *,
    warn_impossible_evolutions: bool = False,
) -> ValidationResult:
    """Check a deck's engine legality and optionally flag dead-weight evolutions.

    Rules (in engine order; violations are hard errors):
      1. exactly 60 cards;
      2. every ID exists in the card database;
      3. <= 4 copies of the same card name (Basic Energy exempt);
      4. at least one Basic Pokémon;
      5. at most one ACE SPEC card.

    :param ids: The deck's Card IDs (one entry per copy).
    :param index: Card index used to look up per-card legality flags.
    :param warn_impossible_evolutions: If True, also warn (non-fatally) about any
        Stage 1/2 Pokémon whose immediate previous stage isn't in the deck at
        all — e.g. a Vaporeon with no Eevee. Such a deck is legal (evolution
        lines aren't enforced by the engine) but that copy can never evolve.
    :return: A :class:`ValidationResult`; ``errors`` empty means the deck is legal.
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
    name_sample: dict[str, int] = {}  # name -> a representative Card ID
    ace_spec = 0
    has_basic = False
    for cid in known:
        info = index.by_id[cid]
        name_counts[info.name] += 1
        name_sample.setdefault(info.name, cid)
        if info.is_ace_spec:
            ace_spec += 1
        if info.is_basic_pokemon:
            has_basic = True

    for name, count in name_counts.items():
        if count > DECK_SAME_CARD_MAX and not index.by_id[name_sample[name]].is_basic_energy:
            # Look up whether this name is a Basic Energy (exempt from the cap).
            errors.append(f"{count} copies of {name!r} (max {DECK_SAME_CARD_MAX})")

    if not has_basic:
        errors.append("deck has no Basic Pokémon")

    if ace_spec > 1:
        errors.append(f"{ace_spec} ACE SPEC cards (max 1)")

    warnings: list[str] = []
    if warn_impossible_evolutions:
        present = {normalize_name(name) for name in name_counts}
        for name in name_counts:
            info = index.by_id[name_sample[name]]
            if info.previous_stage and normalize_name(info.previous_stage) not in present:
                warnings.append(
                    f"{name!r} has no {info.previous_stage!r} in the deck "
                    "(legal, but that copy can never evolve)"
                )

    return ValidationResult(errors=errors, warnings=warnings)
