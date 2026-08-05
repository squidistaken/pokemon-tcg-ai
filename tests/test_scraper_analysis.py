"""Regression tests for small deck corpora."""

from pathlib import Path

import numpy as np

from scraper.analysis.__main__ import analysis_output_dir, discover_corpora
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
