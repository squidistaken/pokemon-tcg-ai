"""CLI entry point for the deck scraper.

Examples::

    python -m scraper --source limitless --format standard --limit 20
    python -m scraper --source bulbapedia --pages "Abyss (TCG),Aurora Blast (TCG)"
    python -m scraper --source text --input mylist.txt --name my-deck
    python -m scraper --source all --limit 20 --verbose
"""

from __future__ import annotations

import argparse
import datetime
import sys

from .card_index import CardIndex
from .pipeline import RunSummary, process_deck
from .sources import SOURCES
from .writer import DEFAULT_DECKS_DIR, DeckWriter


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scraper", description="Scrape and save legal PTCG decks.")
    p.add_argument(
        "--source",
        default="limitless",
        help="Source: limitless | bulbapedia | text | all (default: limitless)",
    )
    p.add_argument("--limit", type=int, default=20, help="Max tournaments/pages to scan")
    p.add_argument("--format", dest="fmt", default="standard", help="Game format (limitless)")
    p.add_argument("--per-tournament", type=int, default=8, help="Max decks per tournament")
    p.add_argument("--pages", help="Comma-separated wiki page titles (bulbapedia)")
    p.add_argument("--category", help="Wiki category to enumerate (bulbapedia)")
    p.add_argument("--input", help="Decklist text file (text source)")
    p.add_argument("--name", help="Archetype/deck name (text source)")
    p.add_argument("--out", default=DEFAULT_DECKS_DIR, help="Output decks directory")
    p.add_argument("--dry-run", action="store_true", help="Resolve+validate only; don't write")
    p.add_argument("--verbose", "-v", action="store_true", help="Per-deck logging")
    return p


def _source_kwargs(args) -> dict:
    return {
        "limit": args.limit,
        "fmt": args.fmt,
        "per_tournament": args.per_tournament,
        "pages": [s.strip() for s in args.pages.split(",")] if args.pages else None,
        "category": args.category,
        "input": args.input,
        "archetype": args.name,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    source_names = list(SOURCES) if args.source == "all" else [args.source]
    for s in source_names:
        if s not in SOURCES:
            print(f"Unknown source {s!r}. Choices: {', '.join(SOURCES)} | all", file=sys.stderr)
            return 2

    index = CardIndex()
    writer = DeckWriter(args.out)
    summary = RunSummary()
    today = datetime.date.today().isoformat()
    kwargs = _source_kwargs(args)

    for name in source_names:
        source = SOURCES[name]()
        print(f"== source: {name} ==")
        try:
            for raw in source.iter_decks(**kwargs):
                process_deck(
                    raw, index, writer, summary,
                    dry_run=args.dry_run, verbose=args.verbose, date=today,
                )
        except Exception as e:  # noqa: BLE001 - report and continue with what we have
            print(f"  source {name!r} error: {e}", file=sys.stderr)

    print("\nSummary:")
    print(summary.format())
    if summary.drops and not args.verbose:
        print(f"  ({len(summary.drops)} decks dropped; re-run with --verbose for reasons)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
