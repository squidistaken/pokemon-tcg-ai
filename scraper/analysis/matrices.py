"""Deck x card-ID matrices the similarity/diversity metrics are built on."""

from __future__ import annotations

import numpy as np


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
