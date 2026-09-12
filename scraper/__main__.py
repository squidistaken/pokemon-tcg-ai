from __future__ import annotations

import argparse
import datetime
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from .card_index import CardIndex
from .card_swapper import (
    DEFAULT_MAPPING_RULE_DIR,
    CardSwapper,
    HeuristicCardSwapper,
    LimitlessProfileLoader,
    MappingCardSwapper,
)
from .discovery import build_canonicalizer
from .inventory import SeenCardInventory
from .pipeline import RunSummary, process_deck
from .source_runner import iter_source_events
from .sources import NETWORK_SOURCES, SOURCES, LimitlessSource
from .writer import DEFAULT_DECKS_DIR, DeckWriter

STRATEGY_DIRS = {
    "mapping": "mapping-resolved",
    "heuristic": "heuristic-resolved",
}


def _fresh_gap_path(out: str | Path) -> Path:
    """Return a run-specific gap path; production reports never resume older data."""
    run_id = os.environ.get("SLURM_JOB_ID") or datetime.datetime.now(
        datetime.UTC
    ).strftime("%Y%m%dT%H%M%S%fZ")
    return Path(out) / f"mapping-gaps-{run_id}.jsonl.gz"


@dataclass
class ResolutionRun:
    """One independent resolution strategy and its isolated output corpus."""

    name: str
    swapper: CardSwapper | None
    writer: DeckWriter
    summary: RunSummary


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
        "--limit",
        type=int,
        default=20,
        help="Tournaments per page / wiki pages to scan",
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
    p.add_argument(
        "--bulbapedia-max-pages",
        type=int,
        default=20,
        help="Maximum Bulbapedia category pages; 0 = until exhausted",
    )
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
        help="Disable the selected fallback card resolver(s)",
    )
    p.add_argument(
        "--card-swap-strategy",
        choices=("heuristic", "mapping", "all"),
        default="heuristic",
        help="Fallback resolver(s); 'all' writes isolated strategy subfolders",
    )
    p.add_argument(
        "--card-swap-map",
        type=Path,
        default=DEFAULT_MAPPING_RULE_DIR,
        help="Directory containing versioned reviewed mapping-rule JSON fragments",
    )
    p.add_argument(
        "--use-rejected-mappings",
        action="store_true",
        help="Also use rules rejected by the independent agent reviewer",
    )
    p.add_argument(
        "--minimum-mapping-confidence",
        type=int,
        choices=range(1, 6),
        default=1,
        metavar="1-5",
        help="Lowest mapping-confidence tier eligible for substitution (default: 1)",
    )
    p.add_argument(
        "--mapping-gap-out",
        type=Path,
        help="Fresh mapping gap report path (default: run-specific file under <out>)",
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
        "bulbapedia_max_pages": args.bulbapedia_max_pages,
        "input": args.input,
        "archetype": args.name,
        "verbose": args.verbose,
    }


def _resolution_runs(
    args, index: CardIndex, profile_loader: LimitlessProfileLoader | None = None
) -> list[ResolutionRun]:
    names = (
        ("mapping", "heuristic")
        if args.card_swap_strategy == "all"
        else (args.card_swap_strategy,)
    )
    runs: list[ResolutionRun] = []
    for name in names:
        swapper: CardSwapper | None = None
        if not args.disable_card_swap:
            if name == "mapping":
                swapper = MappingCardSwapper(
                    index,
                    args.card_swap_map,
                    profile_loader.canonical_set_code if profile_loader else None,
                    use_rejected_mappings=args.use_rejected_mappings,
                    minimum_mapping_confidence=args.minimum_mapping_confidence,
                )
            else:
                swapper = HeuristicCardSwapper(index, profile_loader)
        runs.append(
            ResolutionRun(
                name=name,
                swapper=swapper,
                writer=DeckWriter(
                    str(Path(args.out) / STRATEGY_DIRS[name])
                    if len(names) > 1
                    else args.out
                ),
                summary=RunSummary(),
            )
        )
    return runs


def _print_summary(run: ResolutionRun, *, verbose: bool, labelled: bool) -> None:
    heading = f"Summary ({STRATEGY_DIRS[run.name]}):" if labelled else "Summary:"
    print(f"\n{heading}")
    print(run.summary.format())
    if run.summary.drops and not verbose:
        print(
            f"  ({len(run.summary.drops)} decks dropped; "
            "re-run with --verbose for reasons)"
        )
    if run.summary.warnings and not verbose:
        print(
            f"  ({len(run.summary.warnings)} warnings; "
            "re-run with --verbose for details)"
        )


def main(argv: list[str] | None = None) -> int:
    """
    Run the scraper CLI: scrape the selected source(s) and write legal decks.

    :param argv: Argument vector; defaults to ``sys.argv`` when None.
    :return: Process exit code (0 on success, 2 for an unknown source).
    """
    args = build_parser().parse_args(argv)

    source_names = list(NETWORK_SOURCES) if args.source == "all" else [args.source]
    for s in source_names:
        if s not in SOURCES:
            print(
                f"Unknown source {s!r}. Choices: {', '.join(SOURCES)} | all",
                file=sys.stderr,
            )
            return 2

    sources = {name: SOURCES[name]() for name in source_names}
    limitless = sources.get("limitless")
    if isinstance(limitless, LimitlessSource):
        # The source API and card-profile requests share one thread-safe limiter.
        profile_loader = LimitlessProfileLoader(client=limitless.client)
    else:
        profile_loader = LimitlessProfileLoader()

    index = CardIndex()
    runs = _resolution_runs(args, index, profile_loader)
    mapping_run = next((run for run in runs if run.name == "mapping"), None)
    mapping_swapper = (
        mapping_run.swapper
        if mapping_run is not None
        and isinstance(mapping_run.swapper, MappingCardSwapper)
        else None
    )
    gap_inventory: SeenCardInventory | None = None
    gap_path = args.mapping_gap_out or _fresh_gap_path(args.out)
    if mapping_run is not None and not args.disable_card_swap and not args.dry_run:
        assert mapping_swapper is not None
        if gap_path.exists():
            print(
                f"mapping gap report already exists; refusing to append: {gap_path}",
                file=sys.stderr,
            )
            return 2
        gap_inventory = SeenCardInventory(
            index,
            build_canonicalizer(profile_loader),
            profile_loader,
            checkpoint_label=f"mapping-rules:{mapping_swapper.rules.fingerprint}",
        )
        gap_inventory.write(gap_path)
    gap_decks = 0
    had_source_errors = False
    today = datetime.date.today().isoformat()  # noqa: DTZ011 - local run date is fine for a scrape label
    for name in source_names:
        print(f"== source: {name} ==")
    for event in iter_source_events(sources, _source_kwargs(args)):
        if event.error is not None:
            had_source_errors = True
            print(f"  source {event.source!r} error: {event.error}", file=sys.stderr)
            continue
        if event.deck is None:
            continue
        if gap_inventory is not None and mapping_swapper is not None:
            gaps = [
                card
                for card in event.deck.cards
                if index.match(card.name, card.set_code, card.number).card_id is None
                and not mapping_swapper.resolve(card)
            ]
            try:
                gap_observed = gaps and gap_inventory.observe(
                    replace(event.deck, cards=gaps)
                )
            except Exception as exc:  # noqa: BLE001 - preserve decks; retry gaps later
                had_source_errors = True
                print(
                    f"  mapping gap inventory error for [{event.source}] "
                    f"{event.deck.archetype}: {exc}",
                    file=sys.stderr,
                )
            else:
                if gap_observed:
                    gap_decks += 1
                    if gap_decks % 100 == 0:
                        gap_inventory.write(gap_path)
        for run in runs:
            process_deck(
                event.deck,
                index,
                run.writer,
                run.summary,
                dry_run=args.dry_run,
                verbose=args.verbose,
                date=today,
                warn_impossible_evolutions=args.warn_impossible_evolutions,
                swapper=run.swapper,
            )

    if not args.dry_run:
        for run in runs:
            run.writer.ensure_manifest()
        if gap_inventory is not None:
            gap_inventory.write(gap_path)
    for run in runs:
        _print_summary(run, verbose=args.verbose, labelled=len(runs) > 1)
    return 1 if had_source_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
