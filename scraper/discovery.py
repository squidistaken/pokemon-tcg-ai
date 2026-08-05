"""Fetch-only CLI that inventories source card printings without writing decks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .card_index import CardIndex, normalize_name, normalize_number
from .card_swapper import LimitlessProfileLoader
from .inventory import CanonicalPrinting, SeenCardInventory, resolve_inventory_path
from .models import RawCard
from .source_runner import iter_source_events
from .sources import NETWORK_SOURCES, LimitlessSource

DEFAULT_OUTPUT = Path("outputs/card_discovery/seen_cards.jsonl.gz")


def build_parser() -> argparse.ArgumentParser:
    """Return the fetch-only discovery argument parser."""
    parser = argparse.ArgumentParser(
        prog="scraper.discovery",
        description="Fetch deck sources and inventory their unique card printings.",
    )
    parser.add_argument(
        "--source",
        default="all",
        choices=(*NETWORK_SOURCES, "all"),
        help="Network source to inspect; all runs them concurrently",
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--format", dest="fmt", default="standard")
    parser.add_argument("--per-tournament", type=int, default=8)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--max-decks", type=int, default=15000)
    parser.add_argument("--since", default="2026-01-01")
    parser.add_argument("--until")
    parser.add_argument("--pages")
    parser.add_argument("--category", default="Deck archetypes")
    parser.add_argument("--bulbapedia-max-pages", type=int, default=600)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Atomically checkpoint after this many new decks; 0 = final only",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def _source_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "limit": args.limit,
        "fmt": args.fmt,
        "per_tournament": args.per_tournament,
        "page": args.page,
        "max_pages": args.max_pages,
        "max_decks": args.max_decks,
        "since": args.since,
        "until": args.until,
        "pages": [item.strip() for item in args.pages.split(",")]
        if args.pages
        else None,
        "category": args.category,
        "bulbapedia_max_pages": args.bulbapedia_max_pages,
        "verbose": args.verbose,
    }


def build_canonicalizer(profile_loader: LimitlessProfileLoader):
    def canonicalize(_source: str, card: RawCard) -> CanonicalPrinting:
        set_code = profile_loader.canonical_set_code(card.set_code)
        if set_code is None and card.set_code:
            set_code = normalize_name(card.set_code) or None
        return CanonicalPrinting(
            name=normalize_name(card.name),
            set_code=set_code,
            number=normalize_number(card.number),
        )

    return canonicalize


def main(argv: list[str] | None = None) -> int:
    """Run card discovery and atomically checkpoint the resulting inventory."""
    args = build_parser().parse_args(argv)
    if args.checkpoint_every < 0:
        print("--checkpoint-every must be non-negative", file=sys.stderr)
        return 2

    source_names = list(NETWORK_SOURCES) if args.source == "all" else [args.source]
    sources = {name: NETWORK_SOURCES[name]() for name in source_names}
    limitless = sources.get("limitless")
    profile_loader = LimitlessProfileLoader(
        client=limitless.client if isinstance(limitless, LimitlessSource) else None
    )
    inventory = SeenCardInventory(
        CardIndex(), build_canonicalizer(profile_loader), profile_loader
    )
    try:
        existing_inventory = resolve_inventory_path(args.out)
    except FileNotFoundError:
        existing_inventory = None
    if existing_inventory is not None:
        inventory.merge_file(existing_inventory)
        inventory.refresh_metadata_errors()

    new_decks = 0
    repeated_decks = 0
    errors: list[tuple[str, Exception]] = []
    for name in source_names:
        print(f"== source: {name} ==")
    for event in iter_source_events(sources, _source_kwargs(args)):
        if event.error is not None:
            errors.append((event.source, event.error))
            print(f"  source {event.source!r} error: {event.error}", file=sys.stderr)
            continue
        if event.deck is None:
            continue
        try:
            observed = inventory.observe(event.deck)
        except Exception as exc:  # noqa: BLE001 - retry this deck on the next run
            errors.append((event.source, exc))
            print(
                f"  inventory error for [{event.source}] "
                f"{event.deck.archetype}: {exc}",
                file=sys.stderr,
            )
            continue
        if observed:
            new_decks += 1
            if args.verbose:
                print(f"  observed [{event.deck.source}] {event.deck.archetype}")
            if args.checkpoint_every and new_decks % args.checkpoint_every == 0:
                inventory.write(args.out)
        else:
            repeated_decks += 1

    inventory.write(args.out)
    print("\nDiscovery summary:")
    print(f"  new deck observations: {new_decks}")
    print(f"  already inventoried:   {repeated_decks}")
    print(f"  unique printings:      {len(inventory.records)}")
    print(f"  source errors:         {len(errors)}")
    print(f"  inventory:             {args.out}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
