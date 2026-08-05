"""``python -m scraper.analysis`` — analyze the scraped deck corpus.

Loads every deck CSV under ``--dir``, computes similarity / diversity / metagame
metrics against the manifest's archetype labels, writes a plot suite and a
captured report. With ``--prune`` it instead collapses near-duplicate clusters.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from rich.panel import Panel

from .console import caveat, console
from .loading import (
    REPO_ROOT,
    count_unique_decks,
    load_all_decks,
    load_archetypes,
    load_card_database,
    load_card_index,
    load_manifest,
)
from .matrices import build_count_matrix, build_presence_matrix
from .plots import (
    plot_archetype_distribution,
    plot_card_inclusion,
    plot_set_usage,
    plot_similarity_histogram,
    plot_structure_distributions,
    save_heatmap,
)
from .prune import prune_near_duplicates
from .reporting import (
    corpus_structure_summary,
    report,
    report_clustering,
    report_diversity,
    report_meta,
    report_near_duplicates,
    report_sets,
    save_report,
)
from .similarity import (
    build_semantic_descriptors,
    jaccard_matrix,
    semantic_similarity_matrix,
    weighted_jaccard_matrix,
)
from .structure import deck_structure_stats

OUTPUT_ROOT = REPO_ROOT / "outputs" / "deck_analysis"


def discover_corpora(deck_dir: Path) -> list[Path]:
    """Return the manifest-backed corpora represented by ``deck_dir``.

    A directory with its own manifest is one corpus.  Otherwise, each immediate
    child with a manifest is a separate corpus; this is the standard
    ``decks/{mapping,heuristic}-resolved`` layout.  A legacy flat directory with
    no manifest-backed children remains a single corpus.
    """
    if (deck_dir / "manifest.json").is_file():
        return [deck_dir]
    try:
        children = sorted(
            child
            for child in deck_dir.iterdir()
            if child.is_dir() and (child / "manifest.json").is_file()
        )
    except OSError:
        children = []
    return children or [deck_dir]


def analysis_output_dir(deck_dir: Path) -> Path:
    """Return the strategy-specific output directory for one corpus."""
    return OUTPUT_ROOT / deck_dir.name


def build_parser() -> argparse.ArgumentParser:
    """
    :return: The argument parser for the ``python -m scraper.analysis`` CLI.
    """
    parser = argparse.ArgumentParser(
        prog="scraper.analysis",
        description="Analyze the scraped deck corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dir", default="decks", help="Corpus directory (searched recursively)."
    )
    parser.add_argument(
        "--top", type=int, default=10, help="How many most-similar pairs to list."
    )
    parser.add_argument(
        "--dupe-threshold",
        type=float,
        default=0.9,
        help="Similarity at/above which a deck pair counts as a near-duplicate.",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Collapse near-duplicate clusters (>= --dupe-threshold) to one "
        "representative each (deletes the redundant deck files), then exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """
    Run the corpus analysis (or, with ``--prune``, the near-duplicate collapse).

    :param argv: Argument vector; defaults to ``sys.argv`` when None.
    """
    args = build_parser().parse_args(argv)

    deck_dir = (REPO_ROOT / args.dir).resolve()
    corpora = discover_corpora(deck_dir)

    if args.prune:
        if len(corpora) != 1:
            names = ", ".join(path.name for path in corpora)
            sys.exit(
                f"--prune must target one corpus explicitly with --dir; found: {names}"
            )
        prune_near_duplicates(corpora[0], args.dupe_threshold)
        return

    for corpus_dir in corpora:
        _analyze_corpus(corpus_dir, args)


def _analyze_corpus(deck_dir: Path, args: argparse.Namespace) -> None:
    """Analyze one corpus and write its artifacts to its own subdirectory."""
    # ``save_report`` clears Rich's recording before printing its confirmation.
    # Clear that confirmation before starting the next corpus's report.
    console.export_text(clear=True)

    console.print(f"[dim]Loading decks from[/] {deck_dir} ...")
    names, decks = load_all_decks(deck_dir)
    if not decks:
        sys.exit(f"No valid decks found in {deck_dir}")

    presence = build_presence_matrix(decks)
    counts = build_count_matrix(decks)
    set_sim = jaccard_matrix(presence)
    count_sim = weighted_jaccard_matrix(counts)
    archetypes = load_archetypes(deck_dir, names)

    n = len(names)
    console.print(
        Panel(
            f"[bold]decks analyzed[/]  {n}\n"
            f"[bold]unique decks[/]    {count_unique_decks(decks)}\n"
            f"[bold]unique pairs[/]    {n * (n - 1) // 2}",
            title="Scraped deck corpus",
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
            expand=False,
        )
    )

    db = load_card_database()
    index = load_card_index()

    # Similarity metrics
    report(names, set_sim, archetypes, args.top, "Set Jaccard (unique card IDs)")
    report(names, count_sim, archetypes, args.top, "Weighted Jaccard (card counts)")
    if archetypes is not None:
        report_clustering(set_sim, archetypes, "Set Jaccard (unique card IDs)")
        report_clustering(count_sim, archetypes, "Weighted Jaccard (card counts)")

    if db is not None:
        descriptors, _ = build_semantic_descriptors(decks, db)
        sem_sim = semantic_similarity_matrix(descriptors)
        report(
            names, sem_sim, archetypes, args.top,
            "Card-semantic (cosine of z-scored descriptors)",
        )
        caveat(
            "Range is [bold][-1, 1][/] (z-scored cosine), [bold]NOT[/] comparable "
            "in scale to the Jaccard metrics above; ~0 = unrelated, negative = "
            "opposite strategic profiles."
        )
        if archetypes is not None:
            report_clustering(sem_sim, archetypes, "Card-semantic")

    # Corpus diversity/variance
    report_diversity(presence, counts, count_sim, archetypes)

    # Near-duplicate decks
    report_near_duplicates(names, count_sim, archetypes, args.dupe_threshold)

    # Set/expansion usage
    if index is not None:
        report_sets(decks, index)

    # Deck-internal structure
    stats = columns = None  # reused by the structure plot below
    if db is not None:
        stats, columns = deck_structure_stats(decks, db)
        corpus_structure_summary(stats, columns, archetypes)

    # Metagame metrics
    if archetypes is not None:
        report_meta(decks, names, archetypes, load_manifest(deck_dir), db=db)

    # Plots
    out_dir = analysis_output_dir(deck_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    console.print()
    console.rule("[bold]Plots[/]")
    save_heatmap(
        count_sim, archetypes, out_dir / "similarity_heatmap.png",
        "Weighted-Jaccard similarity",
    )
    plot_similarity_histogram(count_sim, args.dupe_threshold, out_dir / "similarity_histogram.png")
    plot_card_inclusion(presence, out_dir / "card_inclusion.png")
    if archetypes is not None:
        plot_archetype_distribution(archetypes, out_dir / "archetype_distribution.png")
    if index is not None:
        plot_set_usage(decks, index, out_dir / "set_usage.png")
    if stats is not None and columns is not None:
        plot_structure_distributions(stats, columns, out_dir / "structure_distributions.png")

    # Report
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005
    save_report(out_dir / f"report_{stamp}.txt")


if __name__ == "__main__":
    main()
