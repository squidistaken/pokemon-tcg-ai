"""Deck-corpus analysis for the scraper.

Metrics over a directory of scraped deck CSVs: pairwise similarity (set /
weighted Jaccard / card-semantic), clustering agreement with the manifest's
archetype labels, corpus diversity, deck structure, metagame summaries, a plot
suite, and near-duplicate pruning.

Run it as ``python -m scraper.analysis`` (see ``--help``); the functions below
are also importable for ad-hoc use.
"""

from .clustering import clustering_agreement, silhouette_from_similarity
from .loading import (
    count_unique_decks,
    load_all_decks,
    load_archetypes,
    load_card_database,
    load_card_index,
    load_manifest,
    ordered_deck_paths,
)
from .matrices import build_count_matrix, build_presence_matrix
from .prune import prune_near_duplicates
from .similarity import (
    build_semantic_descriptors,
    jaccard_matrix,
    semantic_similarity_matrix,
    weighted_jaccard_matrix,
)
from .structure import deck_structure_stats

__all__ = [
    "build_count_matrix",
    "build_presence_matrix",
    "build_semantic_descriptors",
    "clustering_agreement",
    "count_unique_decks",
    "deck_structure_stats",
    "jaccard_matrix",
    "load_all_decks",
    "load_archetypes",
    "load_card_database",
    "load_card_index",
    "load_manifest",
    "ordered_deck_paths",
    "prune_near_duplicates",
    "semantic_similarity_matrix",
    "silhouette_from_similarity",
    "weighted_jaccard_matrix",
]
