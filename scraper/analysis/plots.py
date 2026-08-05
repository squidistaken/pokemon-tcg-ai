"""Matplotlib/seaborn plots of the corpus (heatmap, histograms, bar charts)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .console import console

if TYPE_CHECKING:
    from scraper.card_index import CardIndex


def _pyplot():
    """
    Lazily import matplotlib with the non-interactive ``Agg`` backend.

    :return: The ``matplotlib.pyplot`` module.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_heatmap(
    sim: np.ndarray, archetypes: list[str] | None, out_path: Path, title: str
) -> None:
    """
    Save a heatmap of a similarity matrix, ordered by archetype if available.

    :param sim: Pairwise similarity matrix.
    :param archetypes: Optional per-deck archetype labels used for ordering.
    :param out_path: Where to write the PNG.
    :param title: Plot title.
    """
    plt = _pyplot()
    import seaborn as sns

    if archetypes is not None:
        order = np.argsort(archetypes, kind="stable")
        sim = sim[np.ix_(order, order)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(sim, cmap="YlGnBu", xticklabels=False, yticklabels=False, square=True)
    suffix = "ordered by archetype; " if archetypes is not None else ""
    plt.title(f"{title} ({suffix}{sim.shape[0]} decks considered)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"\n[green]✓[/] Heatmap written to [bold]{out_path}[/]")


def plot_archetype_distribution(archetypes: list[str], out_path: Path) -> None:
    """
    Save a horizontal bar chart of deck count per archetype.

    :param archetypes: Per-deck archetype labels.
    :param out_path: PNG path to write.
    :return: None.
    """
    plt = _pyplot()
    labels, values = zip(*Counter(archetypes).most_common(), strict=False)
    plt.figure(figsize=(9, max(4.0, len(labels) * 0.22)))
    plt.barh(range(len(labels)), values, color="#4c72b0")
    plt.yticks(range(len(labels)), labels, fontsize=6)
    plt.gca().invert_yaxis()
    plt.xlabel("decks")
    plt.title(f"Archetype distribution ({len(archetypes)} decks, {len(labels)} archetypes)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_similarity_histogram(count_sim: np.ndarray, threshold: float, out_path: Path) -> None:
    """
    Save a histogram of pairwise weighted-Jaccard similarity.

    The near-duplicate threshold is drawn as a vertical line so the redundant
    tail is visible.

    :param count_sim: Weighted-Jaccard similarity matrix.
    :param threshold: Near-duplicate threshold to mark.
    :param out_path: PNG path to write.
    :return: None.
    """
    plt = _pyplot()
    off = count_sim[np.triu_indices(count_sim.shape[0], k=1)]
    plt.figure(figsize=(8, 5))
    plt.hist(off, bins=60, color="#55a868", edgecolor="white", linewidth=0.3)
    plt.axvline(threshold, color="#c44e52", linestyle="--", label=f"near-dup >= {threshold:.2f}")
    plt.xlabel("weighted-Jaccard similarity")
    plt.ylabel("deck pairs")
    plt.title(f"Pairwise deck similarity ({count_sim.shape[0]} decks considered)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_card_inclusion(presence: np.ndarray, out_path: Path) -> None:
    """
    Save a histogram of per-card inclusion rate across the corpus.

    :param presence: Boolean deck x card presence matrix.
    :param out_path: PNG path to write.
    :return: None.
    """
    plt = _pyplot()
    from matplotlib.ticker import PercentFormatter

    incl = presence.astype(np.float64).mean(axis=0)
    incl = incl[incl > 0]  # cards used by at least one deck
    plt.figure(figsize=(8, 5))
    bins: int | np.ndarray = 1
    if incl.size > 1 and incl.min() < incl.max():
        bins = np.geomspace(incl.min(), incl.max(), 51)
    plt.hist(incl, bins=bins, color="#8172b3", edgecolor="white", linewidth=0.3)
    plt.xscale("log")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    plt.xlabel("decks running the card (log scale)")
    plt.ylabel("cards")
    plt.title(f"Card inclusion rate ({incl.size} cards used)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_set_usage(decks: list[list[int]], index: CardIndex, out_path: Path) -> None:
    """
    Save a bar chart of each card set's share of all card copies played.

    :param decks: List of decks, each a list of card IDs.
    :param index: Scraper ``CardIndex`` (card ID -> set code).
    :param out_path: PNG path to write.
    :return: None.
    """
    plt = _pyplot()
    copies: Counter = Counter()
    for deck in decks:
        for cid in deck:
            info = index.by_id.get(cid)
            if info is not None:
                copies[info.set_code or "(none)"] += 1
    labels, values = zip(*copies.most_common(), strict=False)
    plt.figure(figsize=(9, 5))
    plt.bar(range(len(labels)), values, color="#4c72b0")
    plt.xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
    plt.ylabel("card copies played")
    plt.title("Card-set usage across the corpus")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_structure_distributions(stats: np.ndarray, columns: list[str], out_path: Path) -> None:
    """
    Save small-multiple histograms of headline deck-structure statistics.

    :param stats: Per-deck statistics matrix from ``deck_structure_stats``.
    :param columns: Column names matching ``stats``.
    :param out_path: PNG path to write.
    :return: None.
    """
    plt = _pyplot()
    wanted = ["pokemon_count", "trainer_count", "energy_ratio", "ex_count", "mean_pokemon_hp", "distinct_cards"]
    col_idx = {name: i for i, name in enumerate(columns)}
    keys = [k for k in wanted if k in col_idx]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, key in zip(axes.ravel(), keys, strict=False):
        ax.hist(stats[:, col_idx[key]], bins=30, color="#dd8452", edgecolor="white", linewidth=0.3)
        ax.set_title(key, fontsize=9)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.suptitle("Deck-structure distributions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    console.print(f"[green]✓[/] {out_path}")
