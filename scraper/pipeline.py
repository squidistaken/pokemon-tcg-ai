from __future__ import annotations

from dataclasses import dataclass, field

from .card_index import CardIndex
from .card_swapper import CardSwapper
from .models import RawDeck
from .resolver import resolve_deck
from .validator import validate_deck
from .writer import DeckWriter


@dataclass
class RunSummary:
    """Tallies of a scrape run: how many decks were fetched, written, or dropped."""

    fetched: int = 0
    written: int = 0
    #: Decks already in the corpus that contributed a *new* occurrence — the same
    #: 60 cards brought by another player or to another event. No file is written;
    #: the existing entry's ``observation_count`` goes up.
    reobserved: int = 0
    #: Occurrences already on record (a re-scrape of the same standing). Nothing
    #: changes, counts included.
    already_recorded: int = 0
    dropped_unresolved: int = 0
    dropped_invalid: int = 0
    written_slugs: list[str] = field(default_factory=list)
    drops: list[str] = field(default_factory=list)  # human-readable reasons
    warnings: list[str] = field(default_factory=list)  # non-fatal, e.g. impossible evolutions
    #: Written/re-observed decks that only resolved because of a hardcoded card
    #: swap (see :mod:`scraper.card_swapper`) — a subset of ``written`` + ``reobserved``.
    swapped_decks: int = 0

    def format(self) -> str:
        """
        :return: A multi-line, human-readable summary of the run's tallies.
        """
        lines = [
            f"  fetched:            {self.fetched}",
            f"  written (new):      {self.written}",
            f"  re-observed:        {self.reobserved}",
            f"  already recorded:   {self.already_recorded}",
            f"  dropped unresolved: {self.dropped_unresolved}",
            f"  dropped invalid:    {self.dropped_invalid}",
            f"  warnings:           {len(self.warnings)}",
            f"  swapped decks:      {self.swapped_decks}",
        ]
        if self.written_slugs:
            lines.append("  decks: " + ", ".join(self.written_slugs))
        return "\n".join(lines)


def process_deck(
    raw: RawDeck,
    index: CardIndex,
    writer: DeckWriter,
    summary: RunSummary,
    *,
    dry_run: bool = False,
    verbose: bool = False,
    date: str | None = None,
    warn_impossible_evolutions: bool = False,
    swapper: CardSwapper | None = None,
) -> None:
    """
    Resolve, validate, and (unless dry-run) record a single scraped deck.

    Updates ``summary`` in place. A deck is dropped if any card fails to resolve to
    a Card ID or if the resolved deck fails engine legality. Otherwise it is
    recorded, which is one of three things: a new deck file, a new occurrence of a
    deck already in the corpus (same 60 cards, different player or event — the
    file is not rewritten, the entry's ``observation_count`` goes up), or an
    occurrence already on record, which changes nothing.

    :param raw: The scraped decklist to process.
    :param index: Card index used to resolve names/sets to Card IDs.
    :param writer: Destination writer (also the dedup authority).
    :param summary: Run tallies, mutated in place.
    :param dry_run: Resolve and validate but do not touch the disk. Outcomes are
        classified against the corpus as it stands, so two identical new decks in
        one dry run both count as new — nothing is recorded in between.
    :param verbose: Print per-deck resolution/drop diagnostics.
    :param date: Scrape date recorded in the manifest entry.
    :param warn_impossible_evolutions: Non-fatally flag Stage 1/2 Pokémon with no
        copy of their previous stage in the deck (legal, but can never evolve).
        Warnings are recorded in ``summary`` and the deck's manifest entry.
    :param swapper: Optional hardcoded-substitute table consulted for cards that
        would otherwise be unresolved (see :mod:`scraper.card_swapper`). Swaps
        applied are recorded on the deck's manifest entry, like validator warnings.
    """
    summary.fetched += 1
    resolved = resolve_deck(raw, index, swapper)

    if verbose and resolved.fuzzy_matches:
        for scraped, matched, score in resolved.fuzzy_matches:
            print(f"  fuzzy: {scraped!r} -> {matched!r} ({score})")

    if verbose and resolved.swaps:
        for scraped, substitute in resolved.swaps:
            print(f"  swap: {scraped!r} -> {substitute!r}")

    if not resolved.ok:
        summary.dropped_unresolved += 1
        avail = {s.upper() for s in index.available_sets}

        def why(card):
            """Explain why a single card could not be resolved.

            :param card: The unresolved :class:`~scraper.models.RawCard`.
            """
            if card.set_code and card.set_code.upper() not in avail:
                return f"{card.name} (set {card.set_code.upper()} not in card pool)"
            return f"{card.name} (no matching card in pool)"

        names = "; ".join(why(c) for c in resolved.unresolved_cards[:6])
        reason = f"[{raw.source}] {raw.archetype!r}: unresolved -> {names}"
        summary.drops.append(reason)
        if verbose:
            print("DROP (unresolved):", reason)
        return

    result = validate_deck(
        resolved.ids, index, warn_impossible_evolutions=warn_impossible_evolutions
    )
    if result.errors:
        summary.dropped_invalid += 1
        reason = f"[{raw.source}] {raw.archetype!r}: {'; '.join(result.errors)}"
        summary.drops.append(reason)
        if verbose:
            print("DROP (invalid):", reason)
        return

    if result.warnings:
        for w in result.warnings:
            reason = f"[{raw.source}] {raw.archetype!r}: {w}"
            summary.warnings.append(reason)
            if verbose:
                print("WARN:", reason)

    deck_warnings = [
        *result.warnings,
        *(f"swapped {orig!r} -> {sub!r}" for orig, sub in resolved.swaps),
    ]

    if dry_run:
        outcome = writer.classify(resolved, date=date)
    else:
        outcome = writer.write(resolved, date=date, warnings=deck_warnings)

    if resolved.swaps and (outcome.new_deck or outcome.new_observation):
        summary.swapped_decks += 1

    prefix = "OK (dry-run)" if dry_run else "WROTE"
    if outcome.new_deck:
        summary.written += 1
        summary.written_slugs.append(outcome.slug)
        if verbose:
            print(f"{prefix}: {outcome.slug}.csv ({raw.archetype!r}, {len(resolved.ids)} cards)")
    elif outcome.new_observation:
        summary.reobserved += 1
        if verbose:
            print(
                f"RE-OBSERVED: {raw.archetype!r} is the same 60 cards as "
                f"{outcome.slug!r}; recorded as another occurrence"
            )
    else:
        summary.already_recorded += 1
        if verbose:
            print(f"ALREADY RECORDED: this exact occurrence of {outcome.slug!r} is on file")
