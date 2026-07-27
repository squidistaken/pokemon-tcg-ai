"""Collapse near-duplicate deck clusters down to a single representative each."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rich.box import ROUNDED
from rich.table import Table

from .console import console
from .loading import load_deck, load_manifest, ordered_deck_paths
from .matrices import build_count_matrix
from .similarity import weighted_jaccard_matrix


def prune_near_duplicates(deck_dir: Path, threshold: float) -> None:
    """
    Collapse each near-duplicate deck cluster down to a single representative.

    Decks whose pairwise weighted-Jaccard is ``>= threshold`` are unioned into
    clusters (the same transitive clustering the near-duplicate report counts).
    Each cluster is reduced to its medoid — the list most similar to the rest of
    its cluster — and the other files are **deleted**: the redundant CSVs are
    removed, their manifest entries dropped, and any archetype folder left empty
    is removed. This strips the ±1-2 tech-card variants that otherwise oversample
    popular archetypes under a uniform sampler and leak near-identical lists
    across the train/holdout split (which quietly inflates the unseen-deck eval).
    Archetypes and genuine variation are kept.

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

    parent = list(range(n))

    def _find(x: int) -> int:
        """Union-find root of ``x`` with path halving."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    iu = np.triu_indices(n, k=1)
    for k in np.where(sim[iu] >= threshold)[0]:
        ra, rb = _find(int(iu[0][k])), _find(int(iu[1][k]))
        if ra != rb:
            parent[ra] = rb

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[_find(i)].append(i)

    delete_idx: list[int] = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        sub = sim[np.ix_(members, members)]
        keep_local = int(np.argmax(sub.sum(axis=1)))  # medoid of the cluster
        delete_idx.extend(idx for j, idx in enumerate(members) if j != keep_local)

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
    touched_dirs: set[Path] = set()
    for idx in delete_idx:
        p = kept_paths[idx]
        p.unlink(missing_ok=True)
        manifest.pop(p.stem, None)
        touched_dirs.add(p.parent)

    manifest_path = deck_dir / "manifest.json"
    if manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    for d in touched_dirs:
        if d != deck_dir and not any(d.iterdir()):
            d.rmdir()

    console.print(
        f"[green]✓[/] Deleted [bold]{len(delete_idx)}[/] decks; "
        f"manifest now has [bold]{len(manifest)}[/] entries."
    )
