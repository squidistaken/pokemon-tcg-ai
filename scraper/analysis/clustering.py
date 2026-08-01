"""Unsupervised clustering of a similarity matrix and agreement with archetypes.

Pure-numpy silhouette, average-linkage agglomerative clustering, and the ARI /
NMI scores used to validate the manifest's archetype labels.
"""

from __future__ import annotations

import numpy as np


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
