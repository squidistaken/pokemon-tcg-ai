"""Pairwise deck-similarity metrics: set/weighted Jaccard and card-semantic cosine."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .matrices import build_count_matrix

if TYPE_CHECKING:
    from src.env.card_database import CardDatabase

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
    decks: list[list[int]], db: CardDatabase
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
