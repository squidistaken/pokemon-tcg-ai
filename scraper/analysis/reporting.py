"""Rich-console reports rendered from the analysis metrics.

Every function here prints to the shared :data:`~scraper.analysis.console.console`
so the whole run can be exported by :func:`save_report`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rich.box import ROUNDED
from rich.table import Table

from .clustering import clustering_agreement, silhouette_from_similarity
from .console import caveat, console
from .metagame import (
    archetype_core_cards,
    archetype_winrates,
    card_placement_correlation,
    card_pool_coverage,
    card_usage_rates,
    metagame_diversity,
)

if TYPE_CHECKING:
    from scraper.card_index import CardIndex
    from src.env.card_database import CardDatabase


def report_clustering(sim: np.ndarray, labels: list[str], label: str) -> None:
    """
    Print clustering-quality metrics for one similarity metric.

    :param sim: (n, n) pairwise similarity matrix with values in [0, 1].
    :param labels: Ground-truth archetype label for each deck (length n).
    :param label: Human-readable name of the similarity metric (for the heading).
    :return: None.
    """
    sil = silhouette_from_similarity(sim, labels)
    agree = clustering_agreement(sim, labels)

    table = Table(
        box=ROUNDED,
        title=f"Clustering quality vs. archetypes — {label}",
        title_style="bold cyan",
        title_justify="left",
        show_header=False,
    )
    table.add_column("metric", style="dim")
    table.add_column("value", justify="right", style="bold")
    table.add_row("silhouette (1 - sim distance)", f"{sil:+.3f}")
    table.add_row("adjusted Rand index (ARI)", f"[bold green]{agree['ari']:+.3f}[/]")
    table.add_row("normalized mutual info (NMI)", f"{agree['nmi']:.3f}")
    table.add_row(
        "clusters formed",
        f"{agree['n_clusters']} [dim](vs {len(set(labels))} true archetypes)[/]",
    )
    console.print(table)
    caveat(
        "Trust [bold]ARI[/] (chance-corrected, ~0 for random labels). "
        "[bold]NMI[/] has a ~0.5 random floor at this class count, so read it "
        "relatively, not against 0."
    )


def report_meta(
    decks: list[list[int]],
    names: list[str],
    archetypes: list[str],
    manifest: dict,
    db: CardDatabase | None = None,
) -> None:
    """
    Print a readable metagame summary tying the archetype metrics together.

    :param decks: List of decks, each a list of card IDs.
    :param names: Deck file stems, aligned with ``decks``.
    :param archetypes: Archetype label per deck, aligned with ``decks``.
    :param manifest: Full manifest dict keyed by deck stem (see load_manifest).
    :param db: Optional ``CardDatabase`` for readable card names.
    """
    console.print()
    console.rule("[bold magenta]Metagame[/]", style="magenta")

    # diversity
    div = metagame_diversity(archetypes, decks)
    dtable = Table(
        box=ROUNDED,
        title="Metagame diversity",
        title_style="bold",
        title_justify="left",
        show_header=False,
        min_width=44,
    )
    dtable.add_column("metric", style="dim")
    dtable.add_column("value", justify="right", style="bold")
    dtable.add_row("decks", f"{len(decks)}")
    dtable.add_row("archetypes", f"{div['n_archetypes']}")
    dtable.add_row("distinct cards", f"{div['distinct_cards']}")
    if db is not None:
        cov = card_pool_coverage(decks, db)
        dtable.add_row(
            "card-pool coverage",
            f"{cov['distinct_used']} / {cov['total_available']} "
            f"[dim]({cov['coverage']:.1%}, {cov['n_unused']} unused)[/]",
        )
    dtable.add_row(
        "most common archetype",
        f"[cyan]{div['top_archetype']}[/] [dim]({div['top_share']:.1%})[/]",
    )
    dtable.add_row("Shannon entropy", f"{div['shannon_entropy']:.3f} [dim]nats[/]")
    dtable.add_row("effective archetypes", f"{div['effective_archetypes']:.1f}")
    console.print(dtable)
    if db is not None:
        caveat(
            "[bold]Card-pool coverage[/] is against the engine's [italic]entire[/] "
            "card set, which is the competition's fixed playable pool (there is no "
            "separate rotation/legality subset). So this is [bold]true[/] coverage "
            'of that pool -- "unused" cards are playable but simply never run '
            "competitively."
        )

    # most-included cards
    utable = Table(
        box=ROUNDED,
        title="Most-included cards (corpus-wide)",
        title_style="bold",
        title_justify="left",
        header_style="bold magenta",
    )
    utable.add_column("incl%", justify="right", style="green")
    utable.add_column("avg#", justify="right", style="dim")
    utable.add_column("card")
    for _cid, name, rate, avg in card_usage_rates(decks, names, db=db, top=20):
        utable.add_row(f"{rate:.1%}", f"{avg:.2f}", name)
    console.print(utable)

    # core / tech breakdown
    ctable = Table(
        box=ROUNDED,
        title="Archetype core / tech breakdown (top 10 by size)",
        title_style="bold",
        title_justify="left",
        header_style="bold magenta",
        min_width=50,
    )
    ctable.add_column("archetype", style="cyan")
    ctable.add_column("decks", justify="right", style="dim")
    ctable.add_column("core", justify="right", style="green")
    ctable.add_column("tech", justify="right", style="yellow")
    core = archetype_core_cards(decks, names, archetypes, db=db)
    for arch in sorted(core, key=lambda a: core[a]["n_decks"], reverse=True)[:10]:
        info = core[arch]
        ctable.add_row(
            arch, str(info["n_decks"]), str(info["n_core"]), str(info["n_tech"])
        )
    console.print(ctable)

    # win-rate by archetype
    wr = archetype_winrates(names, archetypes, manifest, min_decks=3)
    ranked = wr["rows"]
    title = f"Win-rate by archetype (pooled, ≥{wr['min_decks']} decks)"
    if wr["n_filtered"]:
        title += (
            f"  [dim]({wr['n_filtered']} archetype(s) < {wr['min_decks']} "
            f"decks hidden)[/]"
        )
    wtable = Table(
        box=ROUNDED,
        title=title,
        title_style="bold",
        title_justify="left",
        header_style="bold magenta",
    )
    wtable.add_column("archetype", style="cyan")
    wtable.add_column("decks", justify="right", style="dim")
    wtable.add_column("pooled WR", justify="right", style="bold green")
    wtable.add_column("mean WR", justify="right", style="dim")
    wtable.add_column("std", justify="right", style="dim")

    def _add_wr_rows(rows: list) -> None:
        """Append per-archetype win-rate rows to the results table."""
        for arch, n, pooled, mean, std in rows:
            wtable.add_row(arch, str(n), f"{pooled:.1%}", f"{mean:.1%}", f"±{std:.1%}")

    show = 10
    if len(ranked) > 2 * show:  # split into best / worst blocks
        _add_wr_rows(ranked[:show])
        wtable.add_section()
        _add_wr_rows(ranked[-show:])
    else:
        _add_wr_rows(ranked)
    console.print(wtable)
    caveat(
        "[bold red]Sample-dependent, NOT intrinsic quality.[/] Win-rate reflects "
        "meta positioning and pilot skill in [italic]this[/] snapshot, not a deck's "
        "power in a vacuum. [bold]Pooled WR[/] weights by games played; [bold]mean "
        "WR[/] weights each deck equally. Low-[italic]n[/] archetypes stay noisy "
        "even past the ≥3 filter."
    )

    # card <-> win-rate correlation
    corr = card_placement_correlation(decks, names, manifest, db=db, min_decks=10)
    caveat(
        "[bold red]DESCRIPTIVE, NOT CAUSAL.[/] Card inclusion is confounded with "
        "archetype and with player / meta strength. A positive [italic]r[/] means "
        '"winning decks tend to run this staple", [bold]not[/] "this card causes '
        'wins". Do not read these as card power levels.'
    )
    corr_table = Table(
        box=ROUNDED,
        title=(
            f"Card ↔ win-rate correlation  "
            f"[dim](scored {corr['n_decks_scored']} decks, "
            f"tested {corr['n_cards_tested']} cards)[/]"
        ),
        title_style="bold",
        title_justify="left",
        header_style="bold magenta",
    )
    corr_table.add_column("sign", justify="center")
    corr_table.add_column("r", justify="right", style="bold")
    corr_table.add_column("n", justify="right", style="dim")
    corr_table.add_column("card")
    for _cid, name, r, nd in corr["positive"][:10]:
        corr_table.add_row("[green]▲[/]", f"[green]{r:+.3f}[/]", str(nd), name)
    if corr["positive"][:10] and corr["negative"][:10]:
        corr_table.add_section()
    for _cid, name, r, nd in corr["negative"][:10]:
        corr_table.add_row("[red]▼[/]", f"[red]{r:+.3f}[/]", str(nd), name)
    console.print(corr_table)


def corpus_structure_summary(
    stats: np.ndarray,
    columns: list[str],
    archetypes: list[str] | None = None,
) -> None:
    """
    Print corpus-wide mean/std for each structure statistic, plus an optional
    per-archetype breakdown of a couple of headline stats.

    :param stats: Per-deck statistics matrix from ``deck_structure_stats``.
    :param columns: Column names matching ``stats``.
    :param archetypes: Optional per-deck archetype labels (matrix order). When
                       given, a compact per-archetype table of mean energy ratio
                       and mean Pokemon count is printed.
    :return: None. Results are printed.
    """
    means = stats.mean(axis=0)
    stds = stats.std(axis=0)

    console.print()
    console.rule("[bold green]Deck structure[/]", style="green")

    stable = Table(
        box=ROUNDED,
        title=f"Corpus composition ({stats.shape[0]} decks)",
        title_style="bold",
        title_justify="left",
        header_style="bold magenta",
    )
    stable.add_column("stat", style="cyan")
    stable.add_column("mean", justify="right", style="bold")
    stable.add_column("std", justify="right", style="dim")
    for name, mean, std in zip(columns, means, stds, strict=False):
        stable.add_row(name, f"{mean:.3f}", f"{std:.3f}")
    console.print(stable)
    caveat(
        "No consistency (draw/search) stat — [dim]CardDatabase[/] has no such tag, "
        "so it is omitted rather than guessed. HP / retreat are copy-weighted; "
        "stage counts need not sum to [dim]pokemon_count[/]."
    )

    if archetypes is None:
        return

    col_idx = {name: i for i, name in enumerate(columns)}
    arch = np.array(archetypes)

    atable = Table(
        box=ROUNDED,
        title="Per-archetype composition (mean; archetypes with >= 2 decks, most decks first)",
        title_style="bold",
        title_justify="left",
        header_style="bold magenta",
    )
    atable.add_column("archetype", style="cyan")
    atable.add_column("n", justify="right", style="dim")
    for label in ("poke", "trainer", "energy", "E-ratio", "basic", "st1", "st2", "ex", "HP"):
        atable.add_column(label, justify="right")
    show = [(name, col_idx[key]) for name, key in (
        ("poke", "pokemon_count"), ("trainer", "trainer_count"), ("energy", "energy_count"),
        ("E-ratio", "energy_ratio"), ("basic", "basic_pokemon"), ("st1", "stage1_pokemon"),
        ("st2", "stage2_pokemon"), ("ex", "ex_count"), ("HP", "mean_pokemon_hp"),
    )]
    counts_by_arch = Counter(archetypes)
    for a in [a for a, _ in counts_by_arch.most_common() if counts_by_arch[a] >= 2][:15]:
        mask = arch == a
        row = [a, str(int(mask.sum()))]
        for label, idx in show:
            mean = stats[mask, idx].mean()
            row.append(f"{mean:.2f}" if label in ("E-ratio",) else f"{mean:.1f}")
        atable.add_row(*row)
    console.print(atable)


def report(
    names: list[str],
    sim: np.ndarray,
    archetypes: list[str] | None,
    top: int,
    label: str,
) -> None:
    """
    Print similarity statistics and the most-similar deck pairs for one metric.

    :param names: Deck names in matrix order.
    :param sim: Pairwise similarity matrix.
    :param archetypes: Optional per-deck archetype labels.
    :param top: Number of most-similar pairs to list.
    :param label: Human-readable name of the metric (used in headings).
    """
    n = len(names)
    iu = np.triu_indices(n, k=1)  # unique off-diagonal pairs
    off = sim[iu]

    console.print()
    console.rule(f"[bold cyan]{label}[/]", style="cyan")

    # distribution summary
    dist = Table(
        box=ROUNDED,
        title="Pairwise similarity (off-diagonal)",
        title_style="bold",
        title_justify="left",
        show_header=False,
        min_width=40,
    )
    dist.add_column("stat", style="dim")
    dist.add_column("value", justify="right", style="bold")
    dist.add_row("mean", f"{off.mean():.3f}")
    dist.add_row("median", f"{np.median(off):.3f}")
    dist.add_row("min", f"{off.min():.3f}")
    dist.add_row("max", f"{off.max():.3f}")
    console.print(dist)

    # top-similar pairs
    pairs = Table(
        box=ROUNDED,
        title=f"Top {top} most similar deck pairs",
        title_style="bold",
        title_justify="left",
        header_style="bold magenta",
    )
    pairs.add_column("#", justify="right", style="dim")
    pairs.add_column("sim", justify="right", style="bold green")
    pairs.add_column("deck A")
    pairs.add_column("deck B")
    if archetypes is not None:
        pairs.add_column("archetype", justify="center")
    order = np.argsort(off)[::-1][:top]
    for rank, idx in enumerate(order, 1):
        i, j = iu[0][idx], iu[1][idx]
        row = [str(rank), f"{off[idx]:.3f}", names[i], names[j]]
        if archetypes is not None:
            row.append(
                "[green]same[/]"
                if archetypes[i] == archetypes[j]
                else "[yellow]diff[/]"
            )
        pairs.add_row(*row)
    console.print(pairs)

    if archetypes is not None:
        arch = np.array(archetypes)
        same_mask = arch[iu[0]] == arch[iu[1]]
        intra = off[same_mask]
        inter = off[~same_mask]
        val = Table(
            box=ROUNDED,
            title="Archetype label validation (from manifest.json)",
            title_style="bold",
            title_justify="left",
            show_header=False,
        )
        val.add_column("metric", style="dim")
        val.add_column("value", justify="right", style="bold")
        if intra.size:
            val.add_row(
                "mean intra-archetype similarity",
                f"{intra.mean():.3f} [dim]({intra.size} pairs)[/]",
            )
        if inter.size:
            val.add_row(
                "mean inter-archetype similarity",
                f"{inter.mean():.3f} [dim]({inter.size} pairs)[/]",
            )
        if intra.size and inter.size:
            sep = intra.mean() - inter.mean()
            colour = "green" if sep > 0 else "red"
            val.add_row("separation (intra - inter)", f"[{colour}]{sep:+.3f}[/]")
        console.print(val)


def report_sets(decks: list[list[int]], index: CardIndex) -> None:
    """
    Print how the corpus draws across the engine's card sets (expansions).

    :param decks: List of decks, each a list of card IDs.
    :param index: A ``scraper.card_index.CardIndex`` (card ID -> set code).
    :return: None. Results are printed.
    """
    used_by_set: dict[str, set[int]] = defaultdict(set)
    copies_by_set: Counter = Counter()
    total_by_set: Counter = Counter()
    for info in index.by_id.values():
        total_by_set[info.set_code] += 1
    for deck in decks:
        for cid in deck:
            info = index.by_id.get(cid)
            if info is not None:
                used_by_set[info.set_code].add(cid)
                copies_by_set[info.set_code] += 1

    total_copies = sum(copies_by_set.values()) or 1
    console.print()
    console.rule("[bold blue]Set / expansion usage[/]", style="blue")
    table = Table(
        box=ROUNDED,
        title=f"Card sets used across {len(decks)} decks",
        title_style="bold",
        title_justify="left",
        header_style="bold magenta",
    )
    table.add_column("set", style="cyan")
    table.add_column("cards used", justify="right")
    table.add_column("set coverage", justify="right", style="bold")
    table.add_column("copies", justify="right")
    table.add_column("copy share", justify="right", style="bold green")
    for set_code in sorted(total_by_set, key=lambda s: -copies_by_set.get(s, 0)):
        total = total_by_set[set_code]
        used = len(used_by_set.get(set_code, ()))
        copies = copies_by_set.get(set_code, 0)
        table.add_row(
            set_code or "(none)",
            f"{used}/{total}",
            f"{used / total:.0%}" if total else "-",
            str(copies),
            f"{copies / total_copies:.1%}",
        )
    console.print(table)
    unused = sorted(s for s in total_by_set if copies_by_set.get(s, 0) == 0)
    if unused:
        caveat(f"{len(unused)} set(s) unused by the corpus: " + ", ".join(unused))


def report_diversity(
    presence: np.ndarray,
    counts: np.ndarray,
    count_sim: np.ndarray,
    archetypes: list[str] | None = None,
) -> None:
    """
    Print corpus-level diversity: how much the deck vectors vary, and how tight
    each archetype is.

    :param presence: Boolean deck x card presence matrix.
    :param counts: Integer deck x card copy-count matrix.
    :param count_sim: Weighted-Jaccard pairwise similarity matrix.
    :param archetypes: Optional per-deck archetype labels.
    :return: None. Results are printed.
    """
    n = presence.shape[0]
    iu = np.triu_indices(n, k=1)
    mean_dist = float(1.0 - count_sim[iu].mean())
    incl = presence.astype(np.float64).mean(axis=0)
    active = incl > 0
    incl_var = float((incl[active] * (1.0 - incl[active])).mean()) if active.any() else 0.0
    count_var = float(counts.astype(np.float64)[:, active].var(axis=0).mean()) if active.any() else 0.0

    console.print()
    console.rule("[bold magenta]Corpus diversity[/]", style="magenta")
    t = Table(
        box=ROUNDED,
        title="Deck-vector spread",
        title_style="bold",
        title_justify="left",
        show_header=False,
        min_width=48,
    )
    t.add_column("metric", style="dim")
    t.add_column("value", justify="right", style="bold")
    t.add_row("mean pairwise distance (1 - weighted Jaccard)", f"{mean_dist:.3f}")
    t.add_row("mean card-inclusion variance (Bernoulli)", f"{incl_var:.4f}")
    t.add_row("mean card-count variance", f"{count_var:.3f}")
    console.print(t)
    caveat(
        "Higher mean pairwise distance = a more diverse corpus. Inclusion "
        "variance peaks at 0.25 (a card in ~half the decks); near 0 means cards "
        "are either near-universal or near-absent."
    )

    # Card-usage concentration: how many cards the corpus barely exercises
    # (singletons the model sees ~once) versus how much of all play a few staples
    # account for -- the shape of the training signal over the card space.
    decks_per_card = (presence > 0).sum(axis=0)
    copies_per_card = counts.sum(axis=0)
    n_used = int((decks_per_card > 0).sum())
    total_copies = float(copies_per_card.sum()) or 1.0
    top_share = float(np.sort(copies_per_card)[::-1][:10].sum()) / total_copies
    ct = Table(
        box=ROUNDED,
        title="Card-usage concentration",
        title_style="bold",
        title_justify="left",
        show_header=False,
        min_width=48,
    )
    ct.add_column("metric", style="dim")
    ct.add_column("value", justify="right", style="bold")
    ct.add_row("cards used by the corpus", str(n_used))
    ct.add_row("used in exactly 1 deck (singletons)", f"{int((decks_per_card == 1).sum())}")
    ct.add_row("used in <= 2 decks", f"{int(((decks_per_card >= 1) & (decks_per_card <= 2)).sum())}")
    ct.add_row("top-10 cards' share of all copies", f"{top_share:.1%}")
    console.print(ct)

    if archetypes is None:
        return
    arch = np.array(archetypes)
    rows = []
    for a in np.unique(arch):
        idx = np.where(arch == a)[0]
        if idx.size < 2:
            continue
        sub = count_sim[np.ix_(idx, idx)]
        rows.append((a, idx.size, float(sub[np.triu_indices(idx.size, k=1)].mean())))
    if not rows:
        return
    at = Table(
        box=ROUNDED,
        title="Archetype homogeneity (mean intra-similarity; tightest first)",
        title_style="bold",
        title_justify="left",
        header_style="bold magenta",
    )
    at.add_column("archetype", style="cyan")
    at.add_column("n", justify="right", style="dim")
    at.add_column("mean intra-sim", justify="right", style="bold")
    for a, k, m in sorted(rows, key=lambda r: -r[2])[:12]:
        at.add_row(a, str(k), f"{m:.3f}")
    console.print(at)


def report_near_duplicates(
    names: list[str],
    count_sim: np.ndarray,
    archetypes: list[str] | None,
    threshold: float,
    top: int = 15,
) -> None:
    """
    List near-duplicate deck pairs (weighted-Jaccard >= ``threshold``).

    Near-identical lists inflate popular archetypes and get over-sampled by a
    uniform deck sampler, so flagging them shows where the corpus is redundant.

    :param names: Deck names in matrix order.
    :param count_sim: Weighted-Jaccard similarity matrix.
    :param archetypes: Optional per-deck archetype labels.
    :param threshold: Minimum similarity to count as a near-duplicate.
    :param top: Maximum pairs to list.
    :return: None. Results are printed.
    """
    n = len(names)
    iu = np.triu_indices(n, k=1)
    off = count_sim[iu]
    idxs = np.where(off >= threshold)[0]

    console.print()
    console.rule("[bold red]Near-duplicates[/]", style="red")
    console.print(f"[bold]{idxs.size}[/] deck pair(s) with weighted-Jaccard >= {threshold:.2f}")

    parent = list(range(n))

    def _find(x: int) -> int:
        """Union-find root of ``x`` with path halving."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k in idxs:
        ra, rb = _find(int(iu[0][k])), _find(int(iu[1][k]))
        if ra != rb:
            parent[ra] = rb
    effective = len({_find(i) for i in range(n)})
    console.print(
        f"[bold]{effective}[/] effective distinct decks after collapsing near-duplicate "
        f"clusters [dim]({n - effective} redundant, {(n - effective) / n:.0%} of the corpus)[/]"
    )
    if idxs.size == 0:
        return
    involved = {int(iu[0][k]) for k in idxs} | {int(iu[1][k]) for k in idxs}
    console.print(f"[dim]{len(involved)} of {n} decks are in at least one near-duplicate pair[/]")

    table = Table(
        box=ROUNDED,
        title=f"Top {min(top, idxs.size)} most-similar pairs (>= {threshold:.2f})",
        title_style="bold",
        title_justify="left",
        header_style="bold magenta",
    )
    table.add_column("sim", justify="right", style="bold red")
    table.add_column("deck A")
    table.add_column("deck B")
    if archetypes is not None:
        table.add_column("archetype", justify="center")
    for k in idxs[np.argsort(off[idxs])[::-1][:top]]:
        i, j = int(iu[0][k]), int(iu[1][k])
        row = [f"{off[k]:.3f}", names[i], names[j]]
        if archetypes is not None:
            row.append(
                "[green]same[/]" if archetypes[i] == archetypes[j] else "[yellow]diff[/]"
            )
        table.add_row(*row)
    console.print(table)


def save_report(path: Path) -> None:
    """
    Write everything printed so far to ``path`` (format chosen by extension).

    :param path: Output path; ``.html`` saves HTML, ``.svg`` saves SVG, anything
        else saves plain text.
    :return: None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".html":
        path.write_text(console.export_html(), encoding="utf-8")
    elif suffix == ".svg":
        path.write_text(console.export_svg(title="Deck corpus analysis"), encoding="utf-8")
    else:
        path.write_text(console.export_text(), encoding="utf-8")
    console.print(f"[green]✓[/] Report written to [bold]{path}[/]")
