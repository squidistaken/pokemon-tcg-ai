"""Collapse near-duplicate deck clusters down to a single representative each."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from rich.box import ROUNDED
from rich.table import Table

from .. import manifest as manifest_mod
from ..manifest import Manifest
from .console import console
from .loading import load_deck, load_manifest, ordered_deck_paths
from .matrices import build_count_matrix
from .similarity import weighted_jaccard_matrix


def near_duplicate_clusters(
    sim: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """Group decks only when every pair in a cluster meets ``threshold``."""
    clusters: list[list[int]] = []
    for idx in range(sim.shape[0]):
        for cluster in clusters:
            if all(sim[idx, member] >= threshold for member in cluster):
                cluster.append(idx)
                break
        else:
            clusters.append([idx])
    return clusters


def prune_near_duplicates(deck_dir: Path, threshold: float) -> None:
    """
    Collapse each near-duplicate deck cluster down to a single representative.

    Decks are clustered only when every pair has weighted-Jaccard ``>= threshold``
    (similarity is not treated as transitive). Each cluster is reduced to its
    medoid — the list most similar to the rest of its cluster — and the other files
    are **deleted**: the redundant CSVs are removed, their manifest entries dropped
    (their observations first folded into the medoid, so no provenance is lost),
    and any archetype folder left empty is removed. This strips the ±1-2 tech-card
    variants that otherwise oversample popular archetypes under a uniform sampler
    and leak near-identical lists across the train/holdout split (which quietly
    inflates the unseen-deck eval). Archetypes and genuine variation are kept.

    :param deck_dir: Corpus directory (searched recursively).
    :param threshold: Weighted-Jaccard at/above which two decks are duplicates.
    :return: None. Results are printed; redundant files/manifest entries deleted.
    """
    paths = ordered_deck_paths(deck_dir)
    decks: list[list[int]] = []
    kept_paths: list[Path] = []
    for p in paths:
        try:
            decks.append(load_deck(str(p)))
            kept_paths.append(p)
        except ValueError:
            continue
    n = len(decks)

    console.rule("[bold red]Prune near-duplicates[/]", style="red")
    if n < 2:
        console.print("Nothing to prune.")
        return

    counts = build_count_matrix(decks)
    sim = weighted_jaccard_matrix(counts)

    clusters = near_duplicate_clusters(sim, threshold)

    # keep index -> the indices collapsed into it, so each deleted deck's
    # observations can be folded into the survivor rather than thrown away.
    absorbed: dict[int, list[int]] = {}
    delete_idx: list[int] = []
    for members in clusters:
        if len(members) < 2:
            continue
        sub = sim[np.ix_(members, members)]
        keep_local = int(np.argmax(sub.sum(axis=1)))  # medoid of the cluster
        dropped = [idx for j, idx in enumerate(members) if j != keep_local]
        absorbed[members[keep_local]] = dropped
        delete_idx.extend(dropped)

    kept = n - len(delete_idx)
    console.print(
        f"[bold]{n}[/] decks -> [bold green]{kept}[/] kept, "
        f"[bold red]{len(delete_idx)}[/] deleting "
        f"[dim](weighted-Jaccard >= {threshold:.2f})[/]"
    )
    if not delete_idx:
        return

    # Per-archetype before/after (archetype = parent folder name).
    before: Counter[str] = Counter(p.parent.name for p in kept_paths)
    removed: Counter[str] = Counter(kept_paths[i].parent.name for i in delete_idx)
    table = Table(
        box=ROUNDED,
        title="Deleting per archetype",
        title_style="bold",
        title_justify="left",
        header_style="bold magenta",
    )
    table.add_column("archetype")
    table.add_column("before", justify="right")
    table.add_column("removed", justify="right", style="red")
    table.add_column("after", justify="right", style="green")
    for arch, tot in sorted(before.items(), key=lambda kv: -removed.get(kv[0], 0)):
        rem = removed.get(arch, 0)
        if rem:
            table.add_row(arch, str(tot), str(rem), str(tot - rem))
    console.print(table)

    manifest = load_manifest(deck_dir)
    merged_obs = _merge_observations(manifest, kept_paths, absorbed)

    touched_dirs: set[Path] = set()
    for idx in delete_idx:
        p = kept_paths[idx]
        p.unlink(missing_ok=True)
        manifest.decks.pop(p.stem, None)
        touched_dirs.add(p.parent)

    if (deck_dir / manifest_mod.MANIFEST_NAME).exists():
        manifest_mod.save(manifest, deck_dir)

    for d in touched_dirs:
        if d != deck_dir and not any(d.iterdir()):
            d.rmdir()

    console.print(
        f"[green]✓[/] Deleted [bold]{len(delete_idx)}[/] decks; "
        f"manifest now has [bold]{len(manifest.decks)}[/] entries "
        f"([bold]{merged_obs}[/] observation(s) folded into the survivors)."
    )


def _merge_observations(
    manifest: Manifest,
    paths: list[Path],
    absorbed: dict[int, list[int]],
) -> int:
    """
    Fold each pruned deck's occurrences into the cluster representative that replaces it.

    Deleting a near-duplicate's manifest entry outright would discard the very
    popularity signal ``observation_count`` exists to record — the players who
    brought that list would vanish from the corpus. The occurrences move to the
    surviving medoid instead, tagged with ``merged_from`` so it stays visible that
    they were observed with a slightly different list.

    :param manifest: The manifest, mutated in place.
    :param paths: Deck paths, aligned with the matrix indices.
    :param absorbed: Survivor index -> the indices being collapsed into it.
    :return: How many observations were moved.
    """
    moved = 0
    for keep_idx, dropped in absorbed.items():
        survivor = manifest.decks.get(paths[keep_idx].stem)
        if survivor is None:
            continue
        for idx in dropped:
            entry = manifest.decks.get(paths[idx].stem)
            if entry is None:
                continue
            for obs in entry.observations:
                obs.merged_from = obs.merged_from or paths[idx].stem
                moved += survivor.add_observation(obs)
    return moved
