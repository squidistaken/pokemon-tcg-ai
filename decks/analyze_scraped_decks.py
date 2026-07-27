# TODO: Remove the slop.

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.env.deck import load_deck

if TYPE_CHECKING:
    from scraper.card_index import CardIndex
    from src.env.card_database import CardDatabase



console = Console(record=True, width=100)


def caveat(message: str) -> None:
    """
    Print an interpretation caveat.

    :param message: The caveat text.
    """
    console.print(
        Panel(
            message,
            title="Interpretation",
            title_align="left",
            border_style="yellow",
            padding=(0, 1),
        )
    )


ENERGY_TYPE_NAMES = [
    "colorless",
    "grass",
    "fire",
    "water",
    "lightning",
    "psychic",
    "fighting",
    "darkness",
    "metal",
    "dragon",
    "rainbow",
    "team_rocket",
]

CARD_TYPE_NAMES = [
    "pokemon",
    "item",
    "tool",
    "supporter",
    "stadium",
    "basic_energy",
    "special_energy",
]

def load_all_decks(deck_dir: Path) -> tuple[list[str], list[list[int]]]:
    """
    Load every deck CSV in ``deck_dir`` exactly once.

    :param deck_dir: Directory containing the scraped deck CSV files.
    :return: (names, decks) where names[i] is the file stem and decks[i] is the
             list of 60 card IDs for that deck.
    """
    names: list[str] = []
    decks: list[list[int]] = []
    skipped = 0
    all_csvs = sorted(deck_dir.rglob("*.csv"))
    nested = [p for p in all_csvs if p.parent != deck_dir]
    for path in nested or all_csvs:
        try:
            decks.append(load_deck(str(path)))
            names.append(path.stem)
        except ValueError as exc:  # not a valid 60-card deck
            print(f"  skipping {path.name}: {exc}", file=sys.stderr)
            skipped += 1
    if skipped:
        print(f"  ({skipped} file(s) skipped)", file=sys.stderr)
    return names, decks


def load_manifest(deck_dir: Path) -> dict:
    """
    Load the full corpus manifest keyed by deck stem.

    :param deck_dir: Directory that may contain ``manifest.json``.
    :return: Mapping ``stem -> metadata dict``; empty dict if no manifest.
    """
    manifest_path = deck_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text())


def load_archetypes(deck_dir: Path, names: list[str]) -> list[str] | None:
    """
    Look up each deck's archetype label from the corpus manifest, if present.

    :param deck_dir: Directory that may contain ``manifest.json``.
    :param names: Deck file stems, in matrix order.
    :return: Archetype label per deck, or None if no manifest is available.
    """
    manifest = load_manifest(deck_dir)
    if not manifest:
        return None
    return [manifest.get(name, {}).get("archetype", "Unknown") for name in names]


def count_unique_decks(decks: list[list[int]]) -> int:
    """
    Count decks with a distinct set of card IDs.

    :param decks: List of decks, each a list of card IDs.
    :return: Number of distinct card-ID sets in the corpus.
    """
    return len({frozenset(deck) for deck in decks})


def load_card_database() -> "CardDatabase | None":
    """
    Load the engine card database, or return None if it is unavailable.

    :return: A loaded ``CardDatabase``, or None if construction failed.
    """
    console.print("[dim]Loading CardDatabase ...[/]")
    try:
        from src.env.card_database import CardDatabase

        return CardDatabase()
    except Exception as exc:  # noqa: BLE001 - engine may be unbuilt
        console.print(f"[yellow](CardDatabase unavailable: {exc})[/]")
        return None


def build_presence_matrix(decks: list[list[int]]) -> np.ndarray:
    """
    Build a binary deck x card-ID presence matrix (one row per deck).

    :param decks: List of decks, each a list of card IDs.
    :return: Boolean array of shape (n_decks, max_card_id + 1).
    """
    max_id = max((cid for deck in decks for cid in deck), default=0)
    presence = np.zeros((len(decks), max_id + 1), dtype=bool)
    for i, deck in enumerate(decks):
        presence[i, deck] = True
    return presence


def build_count_matrix(decks: list[list[int]]) -> np.ndarray:
    """
    Build a deck x card-ID *count* matrix (copies of each card per deck).

    :param decks: List of decks, each a list of card IDs.
    :return: Integer array of shape (n_decks, max_card_id + 1).
    """
    max_id = max((cid for deck in decks for cid in deck), default=0)
    counts = np.zeros((len(decks), max_id + 1), dtype=np.int32)
    for i, deck in enumerate(decks):
        np.add.at(counts[i], deck, 1)
    return counts


def jaccard_matrix(presence: np.ndarray) -> np.ndarray:
    """
    Vectorized pairwise Jaccard similarity over the presence matrix.

    :param presence: Boolean (n_decks, n_cards) presence matrix.
    :return: (n_decks, n_decks) float matrix of Jaccard similarities in [0, 1].
    """
    p = presence.astype(np.float64)
    intersection = p @ p.T
    sizes = p.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, intersection / union, 0.0)
    return sim


def weighted_jaccard_matrix(counts: np.ndarray) -> np.ndarray:
    """
    Vectorized pairwise weighted (multiset) Jaccard similarity.

    Weighted Jaccard is ``sum_k min(a_k, b_k) / sum_k max(a_k, b_k)`` and
    accounts for how many copies of each card a deck runs, so two lists that
    share the same cards but at different counts score below 1.0.

    :param counts: Integer (n_decks, n_cards) count matrix.
    :return: (n_decks, n_decks) float matrix of weighted Jaccard in [0, 1].
    """
    n = counts.shape[0]
    intersection = np.zeros((n, n), dtype=np.float64)
    for t in range(1, int(counts.max()) + 1):
        at_least_t = (counts >= t).astype(np.float64)
        intersection += at_least_t @ at_least_t.T
    sizes = counts.sum(axis=1).astype(np.float64)
    union = sizes[:, None] + sizes[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, intersection / union, 0.0)
    return sim


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """
    Elementwise divide, returning 0 where the denominator is 0.

    :param num: Numerator array.
    :param den: Denominator array (broadcastable to ``num``).
    :return: ``num / den`` with 0 substituted wherever ``den == 0``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den != 0, num / den, 0.0)
    return out


def build_semantic_descriptors(
    decks: list[list[int]], db: "CardDatabase"
) -> tuple[np.ndarray, list[str]]:
    """
    Build a count-weighted *semantic* descriptor vector per deck.

    Each deck is summarized by the aggregate properties of its 60 cards rather
    than by which card IDs it runs, so two decks that share a strategy score as
    similar even when their card IDs barely overlap.

    :param decks: List of decks, each a list of 60 card IDs (with duplicates
                  for multiple copies).
    :param db: Loaded :class:`CardDatabase` providing per-card feature tables.
    :return: ``(descriptors, feature_names)`` where ``descriptors`` has shape
             ``(n_decks, 25)`` and ``feature_names`` labels its columns.
    """
    counts = build_count_matrix(decks).astype(np.float64)  # (n_decks, C)
    n_cards = counts.shape[1]

    cf = db.card_features.numpy()[
        :n_cards
    ]  # (C, 12): hp, retreat, basic, stage1, stage2, ex, ...
    cc = db.card_cats.numpy()[
        :n_cards
    ]  # (C, 4): card_type, energy_type, weakness, resistance

    card_type = cc[:, 0]  # 1..7 (0 = unused row)
    energy_type = cc[:, 1]  # 1..12 (0 = absent)
    hp = cf[:, 0]
    basic, stage1, stage2 = cf[:, 2], cf[:, 3], cf[:, 4]
    is_ex, is_ace = cf[:, 5], cf[:, 8]

    is_pokemon = card_type == 1  # (C,)
    is_typed = np.isin(card_type, (1, 6, 7))  # pokemon + energies

    # energy-type distribution
    energy_onehot = np.zeros((n_cards, 12), dtype=np.float64)
    typed_with_energy = is_typed & (energy_type >= 1)
    idx = np.where(typed_with_energy)[0]
    energy_onehot[idx, energy_type[idx] - 1] = 1.0
    energy_counts = counts @ energy_onehot  # (n, 12)
    typed_totals = counts @ typed_with_energy.astype(np.float64)  # (n,)
    energy_frac = _safe_div(energy_counts, typed_totals[:, None])

    # card-type composition
    type_onehot = np.zeros((n_cards, 7), dtype=np.float64)
    idx = np.where(card_type >= 1)[0]
    type_onehot[idx, card_type[idx] - 1] = 1.0
    type_counts = counts @ type_onehot  # (n, 7)
    deck_sizes = counts.sum(axis=1)  # (n,), == 60
    type_frac = _safe_div(type_counts, deck_sizes[:, None])

    # stage mix among Pokemon
    pokemon_totals = counts @ is_pokemon.astype(np.float64)  # (n,)
    stage_basic = counts @ (basic * is_pokemon)
    stage_1 = counts @ (stage1 * is_pokemon)
    stage_2 = counts @ (stage2 * is_pokemon)
    stage_frac = _safe_div(
        np.stack([stage_basic, stage_1, stage_2], axis=1),
        pokemon_totals[:, None],
    )

    # mean Pokemon HP
    hp_sum = counts @ (hp * is_pokemon)
    mean_hp = _safe_div(hp_sum, pokemon_totals)  # (n,)

    # ex + ace-spec density
    ex_sum = counts @ (is_ex * is_pokemon)
    ex_density = _safe_div(ex_sum, pokemon_totals)  # ex / #pokemon
    ace_density = _safe_div(counts @ is_ace, deck_sizes)  # ace-spec / 60

    descriptors = np.concatenate(
        [
            energy_frac,
            type_frac,
            stage_frac,
            mean_hp[:, None],
            ex_density[:, None],
            ace_density[:, None],
        ],
        axis=1,
    )

    feature_names = (
        [f"energy_{n}" for n in ENERGY_TYPE_NAMES]
        + [f"cardtype_{n}" for n in CARD_TYPE_NAMES]
        + ["stage_basic", "stage_stage1", "stage_stage2"]
        + ["mean_pokemon_hp", "ex_density", "ace_spec_density"]
    )
    return descriptors, feature_names


def semantic_similarity_matrix(descriptors: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine similarity over z-scored semantic descriptors.

    :param descriptors: ``(n_decks, n_features)`` descriptor matrix from
                        :func:`build_semantic_descriptors`.
    :return: ``(n_decks, n_decks)`` cosine-similarity matrix in ``[-1, 1]``.
    """
    x = descriptors.astype(np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std_safe = np.where(std > 0, std, 1.0)
    z = np.where(std > 0, (x - mean) / std_safe, 0.0)

    norms = np.linalg.norm(z, axis=1)
    norms_safe = np.where(norms > 0, norms, 1.0)
    unit = z / norms_safe[:, None]
    sim = unit @ unit.T
    return np.clip(sim, -1.0, 1.0)


def silhouette_from_similarity(sim: np.ndarray, labels: list[str]) -> float:
    """
    Mean silhouette coefficient of a labelling under a similarity matrix.

    :param sim: (n, n) pairwise similarity matrix with values in [0, 1].
    :param labels: Ground-truth archetype label for each deck (length n).
    :return: Mean silhouette coefficient over all decks, in [-1, 1].
    """
    n = len(labels)
    if n == 0:
        return 0.0
    dist = 1.0 - np.asarray(sim, dtype=np.float64)

    _, codes = np.unique(np.asarray(labels), return_inverse=True)
    k = int(codes.max()) + 1
    if k < 2:
        return 0.0  # a single cluster has no silhouette
    membership = np.zeros((n, k), dtype=np.float64)
    membership[np.arange(n), codes] = 1.0

    # sums[i, c] = total distance from deck i to every deck in cluster c.
    sums = dist @ membership  # (n, k)
    sizes = membership.sum(axis=0)  # (k,)
    own_size = sizes[codes]  # (n,)

    # a(i): mean intra-cluster distance (own cluster; exclude self, dist[i,i]=0).
    own_sum = sums[np.arange(n), codes]
    a = np.where(own_size > 1, own_sum / np.maximum(own_size - 1.0, 1.0), 0.0)

    # b(i): min over *other* clusters of the mean distance to that cluster.
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_to_cluster = sums / sizes[None, :]  # (n, k)
    mean_to_cluster[np.arange(n), codes] = np.inf  # mask own cluster
    b = mean_to_cluster.min(axis=1)

    denom = np.maximum(a, b)
    sil = np.where(denom > 0, (b - a) / denom, 0.0)
    sil[own_size == 1] = 0.0  # singleton clusters contribute 0 by convention
    return float(sil.mean())


def _average_linkage_clusters(dist: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Average-linkage agglomerative clustering.

    :param dist: (n, n) symmetric distance matrix (zero diagonal).
    :param n_clusters: Number of clusters to cut the dendrogram to.
    :return: Integer cluster label per point (length n, values 0..n_clusters-1).
    """
    n = dist.shape[0]
    n_clusters = max(1, min(n_clusters, n))
    d = dist.astype(np.float64).copy()
    np.fill_diagonal(d, np.inf)
    sizes = np.ones(n, dtype=np.float64)
    assign = np.arange(n)  # original point -> current cluster representative
    n_active = n
    while n_active > n_clusters:
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if not np.isfinite(d[i, j]):
            break  # nothing left to merge (all remaining are inf)
        ni, nj = sizes[i], sizes[j]
        merged = (ni * d[i, :] + nj * d[j, :]) / (ni + nj)
        d[i, :] = merged
        d[:, i] = merged
        d[i, i] = np.inf
        d[j, :] = np.inf  # retire cluster j
        d[:, j] = np.inf
        sizes[i] += nj
        assign[assign == j] = i
        n_active -= 1
    _, labels = np.unique(assign, return_inverse=True)
    return labels


def _contingency(true_codes: np.ndarray, pred_codes: np.ndarray) -> np.ndarray:
    """
    Contingency table counting co-occurrences of two integer labellings.

    :param true_codes: Integer label per point (length n).
    :param pred_codes: Integer label per point (length n).
    :return: (n_true, n_pred) integer count matrix.
    """
    n_true = int(true_codes.max()) + 1
    n_pred = int(pred_codes.max()) + 1
    table = np.zeros((n_true, n_pred), dtype=np.float64)
    np.add.at(table, (true_codes, pred_codes), 1.0)
    return table


def _adjusted_rand_index(table: np.ndarray) -> float:
    """
    Adjusted Rand Index from a contingency table (pure numpy).

    :param table: (n_true, n_pred) contingency count matrix.
    :return: ARI in [-1, 1]; 1.0 for identical partitions, ~0 for random ones.
    """

    def comb2(x: np.ndarray) -> np.ndarray:
        """Element-wise "n choose 2", the pair count within each group."""
        return x * (x - 1.0) / 2.0

    n = table.sum()
    sum_cells = comb2(table).sum()
    sum_rows = comb2(table.sum(axis=1)).sum()
    sum_cols = comb2(table.sum(axis=0)).sum()
    total_pairs = comb2(np.array(n))
    expected = sum_rows * sum_cols / total_pairs if total_pairs > 0 else 0.0
    max_index = 0.5 * (sum_rows + sum_cols)
    denom = max_index - expected
    if denom == 0:
        return 1.0  # both partitions trivial -> perfect agreement by convention
    return float((sum_cells - expected) / denom)


def _normalized_mutual_info(table: np.ndarray) -> float:
    """
    Normalized Mutual Information from a contingency table (pure numpy).

    :param table: (n_true, n_pred) contingency count matrix.
    :return: NMI in [0, 1]; 1.0 for identical partitions, ~0 for independent.
    """
    n = table.sum()
    if n == 0:
        return 1.0
    p_ij = table / n
    p_row = p_ij.sum(axis=1)
    p_col = p_ij.sum(axis=0)

    nz = p_ij > 0
    outer = p_row[:, None] * p_col[None, :]
    mi = np.sum(p_ij[nz] * np.log(p_ij[nz] / outer[nz]))

    def entropy(p: np.ndarray) -> float:
        """Shannon entropy (nats) of a probability vector, ignoring zeros."""
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    h_row = entropy(p_row)
    h_col = entropy(p_col)
    normalizer = 0.5 * (h_row + h_col)
    if normalizer == 0:
        return 1.0  # both partitions have a single class -> perfectly agree
    return float(mi / normalizer)


def clustering_agreement(
    sim: np.ndarray, labels: list[str], n_clusters: int | None = None
) -> dict:
    """
    Cluster a similarity matrix unsupervised and score it against true labels.

    :param sim: (n, n) pairwise similarity matrix with values in [0, 1].
    :param labels: Ground-truth archetype label for each deck (length n).
    :param n_clusters: Number of clusters to form; defaults to the number of
                       distinct archetype labels.
    :return: ``{"ari": float, "nmi": float, "n_clusters": int}``.
    """
    true_labels, true_codes = np.unique(np.asarray(labels), return_inverse=True)
    if n_clusters is None:
        n_clusters = len(true_labels)

    dist = 1.0 - np.asarray(sim, dtype=np.float64)
    dist = 0.5 * (dist + dist.T)  # enforce symmetry for the linkage
    np.fill_diagonal(dist, 0.0)
    pred_codes = _average_linkage_clusters(dist, n_clusters)

    table = _contingency(true_codes, pred_codes)
    return {
        "ari": _adjusted_rand_index(table),
        "nmi": _normalized_mutual_info(table),
        "n_clusters": len(np.unique(pred_codes)),
    }


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

def archetype_core_cards(
    decks: list[list[int]],
    names: list[str],  # noqa: ARG001
    archetypes: list[str],
    db: "CardDatabase | None" = None,
    threshold: float = 0.6,
) -> dict:
    """
    Identify the "core" cards and count the "tech"/variable slots per archetype.

    For each archetype, a card is core if it appears (>=1 copy) in more than
    ``threshold`` of that archetype's deck lists; it is a tech/variable slot
    if it appears in at least one list but in ``threshold`` or fewer of them.

    :param decks: List of decks, each a list of card IDs (duplicates = copies).
    :param names: Deck file stems, in the same order as ``decks``.
    :param archetypes: Archetype label per deck, in the same order as ``decks``.
    :param db: Optional ``CardDatabase`` for readable card names in the output.
    :param threshold: Inclusion fraction above which a card counts as core.
    :return: Mapping ``archetype -> {"n_decks", "core_cards", "n_core",
             "n_tech"}`` where ``core_cards`` is a list of
             ``(card_id, name, inclusion_fraction)`` sorted by inclusion desc.
    """
    by_arch: dict[str, list[int]] = defaultdict(list)
    for i, arch in enumerate(archetypes):
        by_arch[arch].append(i)

    result: dict = {}
    for arch, idxs in by_arch.items():
        n = len(idxs)
        presence_counts: Counter = Counter()
        for i in idxs:
            for cid in set(decks[i]):
                presence_counts[cid] += 1

        core = []
        n_tech = 0
        for cid, cnt in presence_counts.items():
            frac = cnt / n
            if frac > threshold:
                name = db.card_name(cid) if db is not None else str(cid)
                core.append((cid, name, frac))
            else:
                n_tech += 1
        core.sort(key=lambda t: t[2], reverse=True)
        result[arch] = {
            "n_decks": n,
            "core_cards": core,
            "n_core": len(core),
            "n_tech": n_tech,
        }
    return result


def metagame_diversity(archetypes: list[str], decks: list[list[int]]) -> dict:
    """
    Summarize how concentrated or diverse the metagame is.

    :param archetypes: Archetype label per deck.
    :param decks: List of decks, each a list of card IDs (for the distinct-card
                  count across the whole corpus).
    :return: Dict with ``n_archetypes``, ``top_archetype``, ``top_share``,
             ``shannon_entropy`` (nats), ``effective_archetypes``
             (``exp(entropy)``) and ``distinct_cards``.
    """
    counts = Counter(archetypes)
    total = sum(counts.values())
    freqs = np.array([c / total for c in counts.values()], dtype=np.float64)
    entropy = float(-np.sum(freqs * np.log(freqs)))  # nats; 0*log0 := 0
    top_arch, top_count = counts.most_common(1)[0]
    distinct_cards = len({cid for deck in decks for cid in deck})
    return {
        "n_archetypes": len(counts),
        "top_archetype": top_arch,
        "top_share": top_count / total,
        "shannon_entropy": entropy,
        "effective_archetypes": float(np.exp(entropy)),
        "distinct_cards": distinct_cards,
    }


def card_pool_coverage(decks: list[list[int]], db: "CardDatabase") -> dict:
    """
    Measure how much of the engine's card pool the corpus actually uses.

    :param decks: List of decks, each a list of card IDs.
    :param db: Loaded :class:`CardDatabase` providing the exists flag.
    :return: Dict with ``distinct_used``, ``total_available``, ``n_unused`` and
             ``coverage`` (used / available, in ``[0, 1]``).
    """
    distinct_used = len({cid for deck in decks for cid in deck})
    total_available = int(db.card_features.numpy()[:, 11].sum())
    n_unused = total_available - distinct_used
    coverage = distinct_used / total_available if total_available else 0.0
    return {
        "distinct_used": distinct_used,
        "total_available": total_available,
        "n_unused": n_unused,
        "coverage": coverage,
    }


def archetype_winrates(
    names: list[str],
    archetypes: list[str],
    manifest: dict,
    min_decks: int = 3,
) -> dict:
    """
    Aggregate deck win-rates by archetype to compare archetype quality.

    :param names: Deck file stems, aligned with ``archetypes``.
    :param archetypes: Archetype label per deck.
    :param manifest: Full manifest dict keyed by deck stem (see load_manifest).
    :param min_decks: Minimum decks with a valid record for an archetype to be
                      ranked; archetypes below this are dropped (and counted).
    :return: Dict with ``rows`` (list of ``(archetype, n_decks, pooled_winrate,
             mean_winrate, std_winrate)`` sorted by pooled win-rate desc),
             ``n_filtered`` (archetypes dropped by ``min_decks``) and
             ``min_decks``.
    """
    wins: dict[str, int] = defaultdict(int)
    losses: dict[str, int] = defaultdict(int)
    per_deck: dict[str, list[float]] = defaultdict(list)
    for name, arch in zip(names, archetypes, strict=False):
        rec = manifest.get(name, {}).get("record")
        wr = parse_winrate(rec)
        if wr is None:
            continue
        w, l = (int(p) for p in rec.split("-")[:2])  # validated by parse_winrate
        wins[arch] += w
        losses[arch] += l
        per_deck[arch].append(wr)

    rows = []
    n_filtered = 0
    for arch, wrs in per_deck.items():
        if len(wrs) < min_decks:
            n_filtered += 1
            continue
        total_games = wins[arch] + losses[arch]
        pooled = wins[arch] / total_games if total_games else 0.0
        arr = np.array(wrs, dtype=np.float64)
        rows.append((arch, len(wrs), pooled, float(arr.mean()), float(arr.std())))
    rows.sort(key=lambda t: t[2], reverse=True)
    return {"rows": rows, "n_filtered": n_filtered, "min_decks": min_decks}


def card_usage_rates(
    decks: list[list[int]],
    names: list[str],  # noqa: ARG001 - kept for call-site parity with the other metric fns
    db: "CardDatabase | None" = None,
    top: int = 20,
) -> list:
    """
    Rank the most-included cards across the whole corpus.

    For each card: the fraction of decks running >=1 copy ("inclusion rate")
    and, among only those decks that run it, the average number of copies.

    :param decks: List of decks, each a list of card IDs (duplicates = copies).
    :param names: Deck file stems (only used for the corpus size / API parity).
    :param db: Optional ``CardDatabase`` for readable card names.
    :param top: Number of top cards to return, ranked by inclusion rate.
    :return: List of ``(card_id, name, inclusion_rate, avg_copies_when_run)``
             sorted by inclusion rate desc, length <= ``top``.
    """
    n_decks = len(decks)
    deck_count: Counter = Counter()  # decks running the card at all
    copy_total: Counter = Counter()  # total copies across those decks
    for deck in decks:
        card_copies = Counter(deck)
        for cid, copies in card_copies.items():
            deck_count[cid] += 1
            copy_total[cid] += copies

    rows = []
    for cid, dc in deck_count.items():
        name = db.card_name(cid) if db is not None else str(cid)
        rows.append((cid, name, dc / n_decks, copy_total[cid] / dc))
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows[:top]


def parse_winrate(record: str | None) -> float | None:
    """
    Parse a manifest ``record`` string ("W-L-T") into a win-rate.

    :param record: Record string such as ``"4-1-0"``; ties are ignored.
    :return: ``W / (W + L)``, or None if the record is missing, malformed, or
             has no decisive games (``W + L == 0``).
    """
    if not record:
        return None
    parts = record.split("-")
    try:
        w, l = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None
    if w + l == 0:  # no decisive games
        return None
    return w / (w + l)


def card_placement_correlation(
    decks: list[list[int]],
    names: list[str],
    manifest: dict,
    db: "CardDatabase | None" = None,
    min_decks: int = 10,
    top: int = 15,
) -> dict:
    """
    Point-biserial correlation between running a card and deck win-rate.

    :param decks: List of decks, each a list of card IDs.
    :param names: Deck file stems, aligned with ``decks``; used to look up the
                  record in ``manifest``.
    :param manifest: Full manifest dict keyed by deck stem (see load_manifest).
    :param db: Optional ``CardDatabase`` for readable card names.
    :param min_decks: Minimum number of decks a card must appear in to qualify.
    :param top: How many top positive and top negative cards to return.
    :return: Dict with ``n_decks_scored``, ``n_cards_tested`` and
             ``positive`` / ``negative`` lists of
             ``(card_id, name, r, n_decks_with_card)``.
    """
    winrates = []
    kept_decks = []
    for name, deck in zip(names, decks, strict=False):
        wr = parse_winrate(manifest.get(name, {}).get("record"))
        if wr is None:
            continue
        winrates.append(wr)
        kept_decks.append(deck)

    n = len(kept_decks)
    y = np.array(winrates, dtype=np.float64)
    y_centered = y - y.mean()
    y_ss = float(np.sum(y_centered**2))

    card_deckcount: Counter = Counter()
    for deck in kept_decks:
        for cid in set(deck):
            card_deckcount[cid] += 1

    rows = []
    for cid, cnt in card_deckcount.items():
        if cnt < min_decks or cnt == n:  # need variation in x too
            continue
        x = np.array([1.0 if cid in deck else 0.0 for deck in kept_decks])
        x_centered = x - x.mean()
        denom = np.sqrt(float(np.sum(x_centered**2)) * y_ss)
        if denom == 0:
            continue
        r = float(np.sum(x_centered * y_centered) / denom)
        name = db.card_name(cid) if db is not None else str(cid)
        rows.append((cid, name, r, cnt))

    rows.sort(key=lambda t: t[2], reverse=True)
    return {
        "n_decks_scored": n,
        "n_cards_tested": len(rows),
        "positive": rows[:top],
        "negative": rows[-top:][::-1],
    }


def report_meta(
    decks: list[list[int]],
    names: list[str],
    archetypes: list[str],
    manifest: dict,
    db: "CardDatabase | None" = None,
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


def deck_structure_stats(
    decks: list[list[int]], db: "CardDatabase"
) -> tuple[np.ndarray, list[str]]:
    """
    Compute per-deck compositional statistics for the whole corpus at once.

    :param decks: List of decks, each a list of 60 int card IDs (duplicates =
                  copies).
    :param db: Loaded card database providing the ID-indexed lookup tables.
    :return: ``(stats, columns)`` where ``stats`` is a float64 array of shape
             ``(n_decks, len(columns))`` and ``columns`` names each column.
    """
    cf = db.card_features.numpy()  # (max_id+1, 12)
    cc = db.card_cats.numpy()  # (max_id+1, 4)

    card_type = cc[:, 0]  # enum+1: 1=Pokemon ... 6=basic E, 7=special E
    hp = cf[:, 0]
    retreat = cf[:, 1]
    is_basic = cf[:, 2] > 0
    is_stage1 = cf[:, 3] > 0
    is_stage2 = cf[:, 4] > 0
    is_ex = cf[:, 5] > 0
    is_ace = cf[:, 8] > 0

    is_pokemon = card_type == 1
    is_trainer = np.isin(card_type, (2, 3, 4, 5))
    is_energy = np.isin(card_type, (6, 7))

    columns = [
        "pokemon_count",
        "trainer_count",
        "energy_count",
        "energy_ratio",
        "distinct_cards",
        "mean_pokemon_hp",
        "median_pokemon_hp",
        "basic_pokemon",
        "stage1_pokemon",
        "stage2_pokemon",
        "ex_count",
        "ace_spec_count",
        "avg_retreat_cost",
    ]
    stats = np.zeros((len(decks), len(columns)), dtype=np.float64)

    for i, deck in enumerate(decks):
        ids = np.asarray(deck, dtype=np.int64)
        deck_len = ids.size or 1  # guard against empty deck (ratio denom)

        poke_mask = is_pokemon[ids]
        poke_ids = ids[poke_mask]
        poke_hp = hp[poke_ids]

        stats[i] = [
            poke_mask.sum(),
            is_trainer[ids].sum(),
            is_energy[ids].sum(),
            is_energy[ids].sum() / deck_len,
            np.unique(ids).size,
            poke_hp.mean() if poke_hp.size else 0.0,
            np.median(poke_hp) if poke_hp.size else 0.0,
            (poke_mask & is_basic[ids]).sum(),
            (poke_mask & is_stage1[ids]).sum(),
            (poke_mask & is_stage2[ids]).sum(),
            is_ex[ids].sum(),
            is_ace[ids].sum(),
            retreat[poke_ids].mean() if poke_ids.size else 0.0,
        ]

    return stats, columns


def corpus_structure_summary(
    stats: np.ndarray,
    columns: list[str],
    archetypes: list[str] | None = None,
) -> None:
    """
    Print corpus-wide mean/std for each structure statistic, plus an optional
    per-archetype breakdown of a couple of headline stats.

    :param stats: Per-deck statistics matrix from :func:`deck_structure_stats`.
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if archetypes is not None:
        order = np.argsort(archetypes, kind="stable")
        sim = sim[np.ix_(order, order)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(sim, cmap="YlGnBu", xticklabels=False, yticklabels=False, square=True)
    plt.title(title + (" (ordered by archetype)" if archetypes is not None else ""))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"\n[green]✓[/] Heatmap written to [bold]{out_path}[/]")


def load_card_index() -> "CardIndex | None":
    """
    Load the scraper's card index (card ID -> set code / name), or None.

    :return: A ``scraper.card_index.CardIndex``, or None if it could not load.
    """
    try:
        from scraper.card_index import CardIndex

        return CardIndex()
    except Exception as exc:  # noqa: BLE001 - card CSV may be absent
        console.print(f"[yellow](CardIndex unavailable: {exc})[/]")
        return None


def report_sets(decks: list[list[int]], index: "CardIndex") -> None:
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


# --------------------------------------------------------------------------- #
# Near-duplicate detection
# --------------------------------------------------------------------------- #
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


def _pyplot():
    """
    Lazily import matplotlib with the non-interactive ``Agg`` backend.

    :return: The ``matplotlib.pyplot`` module.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


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
    plt.title("Pairwise deck similarity")
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
    incl = presence.astype(np.float64).mean(axis=0)
    incl = incl[incl > 0]  # cards used by at least one deck
    plt.figure(figsize=(8, 5))
    plt.hist(incl, bins=50, color="#8172b3", edgecolor="white", linewidth=0.3)
    plt.xlabel("fraction of decks running the card")
    plt.ylabel("cards")
    plt.title(f"Card inclusion rate ({incl.size} cards used)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_set_usage(decks: list[list[int]], index: "CardIndex", out_path: Path) -> None:
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

    :param stats: Per-deck statistics matrix from :func:`deck_structure_stats`.
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


# region Prune


def _ordered_deck_paths(deck_dir: Path) -> list[Path]:
    """
    Deck CSV paths in the same order :func:`load_all_decks` loads them.

    :param deck_dir: Corpus directory (searched recursively).
    :return: Nested (archetype-subfolder) deck CSV paths, sorted; falls back to
             loose files if the corpus is flat.
    """
    all_csvs = sorted(deck_dir.rglob("*.csv"))
    nested = [p for p in all_csvs if p.parent != deck_dir]
    return nested or all_csvs


def prune_near_duplicates(deck_dir: Path, threshold: float, *, apply: bool) -> None:
    """
    Collapse each near-duplicate deck cluster down to a single representative.

    Decks whose pairwise weighted-Jaccard is ``>= threshold`` are unioned into
    clusters (the same transitive clustering the near-duplicate report counts).
    Each cluster is reduced to its medoid — the list most similar to the rest of
    its cluster — and the other files are deleted. This removes the ±1-2 tech-card
    variants that otherwise oversample popular archetypes under a uniform sampler
    and leak near-identical lists across the train/holdout split (which quietly
    inflates the unseen-deck eval). Archetypes and genuine variation are kept.

    :param deck_dir: Corpus directory (searched recursively).
    :param threshold: Weighted-Jaccard at/above which two decks are duplicates.
    :param apply: When False, only report what would be deleted (dry run); when
        True, delete the redundant CSVs, drop their manifest entries, and remove
        any archetype folders left empty.
    :return: None. Results are printed; files/manifest change only if ``apply``.
    """
    paths = _ordered_deck_paths(deck_dir)
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
    verb = "Deleting" if apply else "Would delete"
    console.print(
        f"[bold]{n}[/] decks -> [bold green]{kept}[/] kept, "
        f"[bold red]{len(delete_idx)}[/] {verb.lower()} "
        f"[dim](weighted-Jaccard >= {threshold:.2f})[/]"
    )
    if not delete_idx:
        return

    # Per-archetype before/after (archetype = parent folder name).
    before: Counter[str] = Counter(p.parent.name for p in kept_paths)
    removed: Counter[str] = Counter(kept_paths[i].parent.name for i in delete_idx)
    table = Table(
        box=ROUNDED,
        title=f"{verb} per archetype",
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

    if not apply:
        console.print(
            "[yellow]Dry run.[/] Re-run with [bold]--prune --apply[/] to delete."
        )
        return

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


# endregion


def main() -> None:
    """
    CLI entry point.
    """
    parser = argparse.ArgumentParser(
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
        "--plot-dir",
        default="outputs/deck_analysis",
        help="Directory the report and plot suite are written to.",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Collapse near-duplicate clusters (>= --dupe-threshold) to one "
        "representative each, then exit. Dry run unless --apply is given.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="With --prune, actually delete the redundant decks.",
    )
    args = parser.parse_args()

    deck_dir = (REPO_ROOT / args.dir).resolve()

    if args.prune:
        prune_near_duplicates(deck_dir, args.dupe_threshold, apply=args.apply)
        return

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
    out_dir = (REPO_ROOT / args.plot_dir).resolve()
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
