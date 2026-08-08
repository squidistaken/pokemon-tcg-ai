"""``python -m scraper.analysis`` — analyze the scraped deck corpus.

Loads every deck CSV under ``--dir``, computes similarity / diversity / metagame
metrics against the manifest's archetype labels, writes a plot suite and a
captured report. With ``--prune`` it instead collapses near-duplicate clusters.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

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
DEFAULT_CONFIDENCE_LEVEL = 0.99
DEFAULT_MARGIN_OF_ERROR = 0.02


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


def sample_decks(
    names: list[str],
    decks: list[list[int]],
    max_decks: int,
    seed: int,
) -> tuple[list[str], list[list[int]]]:
    """Take a deterministic uniform sample before quadratic analysis."""
    if max_decks == 0 or len(decks) <= max_decks:
        return names, decks
    indices = sorted(random.Random(seed).sample(range(len(decks)), max_decks))
    return [names[index] for index in indices], [decks[index] for index in indices]


def required_sample_size(
    population_size: int,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    margin_of_error: float = DEFAULT_MARGIN_OF_ERROR,
) -> int:
    """Return a conservative finite-population sample size for proportions."""
    if population_size < 0:
        raise ValueError("population_size must be zero or greater")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if not 0.0 < margin_of_error < 1.0:
        raise ValueError("margin_of_error must be between zero and one")
    if population_size == 0:
        return 0
    z_score = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
    initial = z_score**2 * 0.25 / margin_of_error**2
    corrected = initial / (1.0 + (initial - 1.0) / population_size)
    return min(population_size, math.ceil(corrected))


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
        "--max-decks",
        type=int,
        default=None,
        help="Override the calculated sample size; 0 performs the expensive full analysis.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=DEFAULT_CONFIDENCE_LEVEL,
        help="Confidence level used to calculate the sample size.",
    )
    parser.add_argument(
        "--margin-of-error",
        type=float,
        default=DEFAULT_MARGIN_OF_ERROR,
        help="Margin of error used to calculate the sample size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic deck sampling.",
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_decks is not None and args.max_decks < 0:
        parser.error("--max-decks must be zero or greater")
    if not 0.0 < args.confidence_level < 1.0:
        parser.error("--confidence-level must be between zero and one")
    if not 0.0 < args.margin_of_error < 1.0:
        parser.error("--margin-of-error must be between zero and one")

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
    all_names, all_decks = load_all_decks(deck_dir)
    if not all_decks:
        sys.exit(f"No valid decks found in {deck_dir}")
    corpus_size = len(all_decks)
    calculated_sample = args.max_decks is None
    sample_size = args.max_decks
    if sample_size is None:
        sample_size = required_sample_size(
            corpus_size, args.confidence_level, args.margin_of_error
        )
    names, decks = sample_decks(all_names, all_decks, sample_size, args.seed)

    full_presence = build_presence_matrix(all_decks)
    full_counts = build_count_matrix(all_decks)
    presence = build_presence_matrix(decks)
    counts = build_count_matrix(decks)
    set_sim = jaccard_matrix(presence)
    count_sim = weighted_jaccard_matrix(counts)
    archetypes = load_archetypes(deck_dir, names)
    all_archetypes = load_archetypes(deck_dir, all_names)

    n = len(names)
    if n < corpus_size:
        target = (
            f"{args.confidence_level:.0%} confidence, ±{args.margin_of_error:.1%}"
            if calculated_sample
            else f"explicit cap of {sample_size} decks"
        )
        sample_line = (
            f"\n[bold]sample seed[/]     {args.seed}\n[bold]sample target[/]   {target}"
        )
    else:
        sample_line = ""
    console.print(
        Panel(
            f"[bold]corpus decks[/]    {corpus_size}\n"
            f"[bold]pairwise sample[/] {n}{sample_line}\n"
            f"[bold]unique decks[/]    {count_unique_decks(all_decks)}\n"
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
            names,
            sem_sim,
            archetypes,
            args.top,
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
    report_diversity(full_presence, full_counts, count_sim, archetypes)

    # Near-duplicate decks
    report_near_duplicates(names, count_sim, archetypes, args.dupe_threshold)

    # Set/expansion usage
    if index is not None:
        report_sets(all_decks, index)

    # Deck-internal structure
    stats = columns = None  # reused by the structure plot below
    if db is not None:
        stats, columns = deck_structure_stats(all_decks, db)
        corpus_structure_summary(stats, columns, all_archetypes)

    # Metagame metrics
    if all_archetypes is not None:
        report_meta(
            all_decks,
            all_names,
            all_archetypes,
            load_manifest(deck_dir),
            db=db,
        )

    # Plots
    out_dir = analysis_output_dir(deck_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    console.print()
    console.rule("[bold]Plots[/]")
    save_heatmap(
        count_sim,
        archetypes,
        out_dir / "similarity_heatmap.png",
        "Weighted-Jaccard similarity",
    )
    plot_similarity_histogram(
        count_sim, args.dupe_threshold, out_dir / "similarity_histogram.png"
    )
    plot_card_inclusion(full_presence, out_dir / "card_inclusion.png")
    if all_archetypes is not None:
        plot_archetype_distribution(
            all_archetypes, out_dir / "archetype_distribution.png"
        )
    if index is not None:
        plot_set_usage(all_decks, index, out_dir / "set_usage.png")
    if stats is not None and columns is not None:
        plot_structure_distributions(
            stats, columns, out_dir / "structure_distributions.png"
        )

    # Report
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005
    save_report(out_dir / f"report_{stamp}.txt")


if __name__ == "__main__":
    main()
