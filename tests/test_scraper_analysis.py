"""Regression tests for small deck corpora."""

from pathlib import Path

import numpy as np

from scraper.analysis.__main__ import (
    analysis_output_dir,
    discover_corpora,
    required_sample_size,
    sample_decks,
)
from scraper.analysis.prune import near_duplicate_clusters
from scraper.analysis.reporting import report, report_diversity, report_near_duplicates


def test_pairwise_reports_accept_a_single_deck():
    similarity = np.ones((1, 1))
    presence = np.array([[True, False]])
    counts = np.array([[4, 0]])

    report(["only-deck"], similarity, ["Example"], top=10, label="test")
    report_diversity(presence, counts, similarity, ["Example"])
    report_near_duplicates(["only-deck"], similarity, ["Example"], threshold=0.9)


def test_diversity_uses_full_vectors_with_a_pairwise_sample():
    presence = np.array(
        [[True, False], [False, True], [True, True], [True, False]]
    )
    counts = presence.astype(np.int32)
    sampled_similarity = np.array([[1.0, 0.5], [0.5, 1.0]])

    report_diversity(
        presence,
        counts,
        sampled_similarity,
        ["Example", "Example"],
    )


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


def test_discovers_manifest_backed_strategy_corpora(tmp_path):
    mapping = tmp_path / "mapping-resolved"
    heuristic = tmp_path / "heuristic-resolved"
    unrelated = tmp_path / "other"
    for directory in (mapping, heuristic, unrelated):
        directory.mkdir()
    (mapping / "manifest.json").write_text("{}", encoding="utf-8")
    (heuristic / "manifest.json").write_text("{}", encoding="utf-8")

    assert discover_corpora(tmp_path) == [heuristic, mapping]


def test_explicit_manifest_directory_is_one_corpus(tmp_path):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    nested = tmp_path / "mapping-resolved"
    nested.mkdir()
    (nested / "manifest.json").write_text("{}", encoding="utf-8")

    assert discover_corpora(tmp_path) == [tmp_path]


def test_legacy_flat_directory_remains_one_corpus(tmp_path):
    assert discover_corpora(tmp_path) == [tmp_path]


def test_analysis_output_uses_corpus_name():
    output = analysis_output_dir(Path("decks/mapping-resolved"))

    assert output.name == "mapping-resolved"
    assert output.parent.name == "deck_analysis"


def test_large_analysis_sample_is_bounded_deterministic_and_aligned():
    names = [f"deck-{index}" for index in range(10)]
    decks = [[index] for index in range(10)]

    first = sample_decks(names, decks, max_decks=4, seed=7)
    second = sample_decks(names, decks, max_decks=4, seed=7)

    assert first == second
    assert len(first[0]) == 4
    assert all(name == f"deck-{deck[0]}" for name, deck in zip(*first, strict=True))


def test_zero_max_decks_keeps_full_corpus():
    names = ["one", "two"]
    decks = [[1], [2]]

    assert sample_decks(names, decks, max_decks=0, seed=0) == (names, decks)


def test_confidence_sample_size_uses_finite_population_correction():
    assert required_sample_size(30_053, 0.95, 0.05) == 380
    assert required_sample_size(30_053, 0.99, 0.05) == 650
    assert required_sample_size(28_670, 0.95, 0.05) == 380
    assert required_sample_size(28_670, 0.99, 0.05) == 649


def test_confidence_sample_never_exceeds_population():
    assert required_sample_size(12, 0.99, 0.01) == 12
