"""Glue: run a source, resolve + validate + write each deck, report a summary."""

from __future__ import annotations

from dataclasses import dataclass, field

from .card_index import CardIndex
from .models import RawDeck
from .resolver import resolve_deck
from .validator import validate
from .writer import DeckWriter


@dataclass
class RunSummary:
    fetched: int = 0
    written: int = 0
    deduped: int = 0
    dropped_unresolved: int = 0
    dropped_invalid: int = 0
    written_slugs: list[str] = field(default_factory=list)
    drops: list[str] = field(default_factory=list)  # human-readable reasons

    def format(self) -> str:
        lines = [
            f"  fetched:            {self.fetched}",
            f"  written:            {self.written}",
            f"  deduped (skipped):  {self.deduped}",
            f"  dropped unresolved: {self.dropped_unresolved}",
            f"  dropped invalid:    {self.dropped_invalid}",
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
) -> None:
    """Resolve, validate, and (unless dry-run) write a single deck."""
    summary.fetched += 1
    resolved = resolve_deck(raw, index)

    if verbose and resolved.fuzzy:
        for scraped, matched, score in resolved.fuzzy:
            print(f"  fuzzy: {scraped!r} -> {matched!r} ({score})")

    if not resolved.ok:
        summary.dropped_unresolved += 1
        avail = {s.upper() for s in index.available_sets}

        def why(card):
            if card.set_code and card.set_code.upper() not in avail:
                return f"{card.name} (set {card.set_code.upper()} not in card pool)"
            return f"{card.name} (no matching card in pool)"

        names = "; ".join(why(c) for c in resolved.unresolved[:6])
        reason = f"[{raw.source}] {raw.archetype!r}: unresolved -> {names}"
        summary.drops.append(reason)
        if verbose:
            print("DROP (unresolved):", reason)
        return

    errors = validate(resolved.ids, index)
    if errors:
        summary.dropped_invalid += 1
        reason = f"[{raw.source}] {raw.archetype!r}: {'; '.join(errors)}"
        summary.drops.append(reason)
        if verbose:
            print("DROP (invalid):", reason)
        return

    if writer.is_duplicate(resolved.ids):
        summary.deduped += 1
        if verbose:
            print(f"DEDUP: {raw.archetype!r} already saved")
        return

    if dry_run:
        summary.written += 1  # would-be write
        if verbose:
            print(f"OK (dry-run): {raw.archetype!r} -> {len(resolved.ids)} cards")
        return

    slug = writer.write(resolved, date=date)
    if slug is None:  # raced dedup
        summary.deduped += 1
        return
    summary.written += 1
    summary.written_slugs.append(slug)
    if verbose:
        print(f"WROTE: {slug}.csv ({raw.archetype!r})")
