from __future__ import annotations

import argparse
import datetime
import sys

from .card_index import CardIndex
from .card_swapper import CardSwapper
from .pipeline import RunSummary, process_deck
from .sources import SOURCES
from .writer import DEFAULT_DECKS_DIR, DeckWriter


def build_parser() -> argparse.ArgumentParser:
    """
    :return: The argument parser for the ``python -m scraper`` CLI.
    """
    p = argparse.ArgumentParser(
        prog="scraper", description="Scrape and save legal PTCG decks."
    )
    p.add_argument(
        "--source",
        default="limitless",
        help="Source: limitless | bulbapedia | text | all (default: limitless)",
    )
    p.add_argument(
        "--limit", type=int, default=20, help="Tournaments per page / wiki pages to scan"
    )
    p.add_argument(
        "--format", dest="fmt", default="standard", help="Game format (limitless)"
    )
    p.add_argument(
        "--per-tournament",
        type=int,
        default=8,
        help="Max decks per tournament, best finish first; 0 = every published list",
    )
    p.add_argument(
        "--page", type=int, default=1, help="Results page to start from (limitless)"
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Pages to walk from --page, going back in time; 0 = until exhausted",
    )
    p.add_argument(
        "--max-decks", type=int, help="Stop after scraping this many decks (limitless)"
    )
    p.add_argument("--since", help="Keep tournaments on/after YYYY-MM-DD (limitless)")
    p.add_argument("--until", help="Keep tournaments on/before YYYY-MM-DD (limitless)")
    p.add_argument("--pages", help="Comma-separated wiki page titles (bulbapedia)")
    p.add_argument("--category", help="Wiki category to enumerate (bulbapedia)")
    p.add_argument("--input", help="Decklist text file (text source)")
    p.add_argument("--name", help="Archetype/deck name (text source)")
    p.add_argument("--out", default=DEFAULT_DECKS_DIR, help="Output decks directory")
    p.add_argument(
        "--dry-run", action="store_true", help="Resolve+validate only; don't write"
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Per-deck logging")
    p.add_argument(
        "--warn-impossible-evolutions",
        action="store_true",
        help="Flag (non-fatally) Stage 1/2 Pokémon with no copy of their "
        "previous stage in the deck, e.g. a Vaporeon with no Eevee",
    )
    p.add_argument(
        "--disable-card-swap",
        action="store_true",
        help="Don't substitute hardcoded staples missing from the card pool "
        "(see scraper.card_swapper); swapping is on by default",
    )
    return p


def _source_kwargs(args) -> dict:
    """
    Collect the parsed CLI args into the keyword dict every source accepts.

    :param args: Parsed argparse namespace.
    :return: Keyword arguments forwarded to ``DeckSource.iter_decks``.
    """
    return {
        "limit": args.limit,
        "fmt": args.fmt,
        "per_tournament": args.per_tournament,
        "page": args.page,
        "max_pages": args.max_pages,
        "max_decks": args.max_decks,
        "since": args.since,
        "until": args.until,
        "pages": [s.strip() for s in args.pages.split(",")] if args.pages else None,
        "category": args.category,
        "input": args.input,
        "archetype": args.name,
        "verbose": args.verbose,
    }


def main(argv: list[str] | None = None) -> int:
    """
    Run the scraper CLI: scrape the selected source(s) and write legal decks.

    :param argv: Argument vector; defaults to ``sys.argv`` when None.
    :return: Process exit code (0 on success, 2 for an unknown source).
    """
    args = build_parser().parse_args(argv)

    source_names = list(SOURCES) if args.source == "all" else [args.source]
    for s in source_names:
        if s not in SOURCES:
            print(
                f"Unknown source {s!r}. Choices: {', '.join(SOURCES)} | all",
                file=sys.stderr,
            )
            return 2

    index = CardIndex()
    swapper = None if args.disable_card_swap else CardSwapper(index)
    writer = DeckWriter(args.out)
    summary = RunSummary()
    today = datetime.date.today().isoformat()  # noqa: DTZ011 - local run date is fine for a scrape label
    kwargs = _source_kwargs(args)

    for name in source_names:
        source = SOURCES[name]()
        print(f"== source: {name} ==")
        try:
            for raw in source.iter_decks(**kwargs):
                process_deck(
                    raw,
                    index,
                    writer,
                    summary,
                    dry_run=args.dry_run,
                    verbose=args.verbose,
                    date=today,
                    warn_impossible_evolutions=args.warn_impossible_evolutions,
                    swapper=swapper,
                )
        except Exception as e:  # noqa: BLE001 - report and continue with what we have
            print(f"  source {name!r} error: {e}", file=sys.stderr)

    print("\nSummary:")
    print(summary.format())
    if summary.drops and not args.verbose:
        print(
            f"  ({len(summary.drops)} decks dropped; re-run with --verbose for reasons)"
        )
    if summary.warnings and not args.verbose:
        print(
            f"  ({len(summary.warnings)} warnings; re-run with --verbose for details)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
