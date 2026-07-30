"""Per-deck compositional statistics (Pokemon/trainer/energy mix, HP, stages, ...)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.env.card_database import CardDatabase


def deck_structure_stats(
    decks: list[list[int]], db: CardDatabase
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
