"""Regression tests for small deck corpora."""

import numpy as np

from scraper.analysis.prune import near_duplicate_clusters
from scraper.analysis.reporting import report, report_diversity, report_near_duplicates


def test_pairwise_reports_accept_a_single_deck():
    similarity = np.ones((1, 1))
    presence = np.array([[True, False]])
    counts = np.array([[4, 0]])

    report(["only-deck"], similarity, ["Example"], top=10, label="test")
    report_diversity(presence, counts, similarity, ["Example"])
    report_near_duplicates(["only-deck"], similarity, ["Example"], threshold=0.9)


def test_near_duplicate_clusters_do_not_apply_similarity_transitively():
    similarity = np.array(
        [
            [1.00, 0.91, 0.80],
            [0.91, 1.00, 0.91],
            [0.80, 0.91, 1.00],
        ]
    )

    clusters = near_duplicate_clusters(similarity, threshold=0.9)

    assert sorted(map(len, clusters)) == [1, 2]
