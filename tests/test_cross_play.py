import math
import random
from pathlib import Path

import pytest

from src.env.battle_handle import BattleHandle
from src.env.decks.deck import load_deck
from src.env.decks.deck_sampler import FixedDeckSampler
from src.env.opponents.random_opponent import RandomOpponent
from src.training.cross_play import (
    MatchResult,
    bradley_terry_elo,
    crossplay_matrix,
    play_match,
    play_series,
)

DECK = load_deck(str(Path(__file__).parents[1] / "decks" / "example.csv"))


def _mirror_sampler() -> FixedDeckSampler:
    """A sampler handing both seats the example deck (pure-skill mirror)."""
    return FixedDeckSampler(DECK, DECK)


def test_match_result_scores_draws_as_half() -> None:
    """
    ``score`` is win-equivalents over resolved games; an empty match reads 0.5.
    """
    result = MatchResult(wins=3, draws=2, losses=5)
    assert result.scored == 10
    assert result.score == (3 + 1.0) / 10
    empty = MatchResult(0, 0, 0)
    assert empty.scored == 0
    assert empty.score == 0.5


def test_bradley_terry_recovers_a_transitive_ranking() -> None:
    """
    Elo from a matrix where A beats B beats C recovers that strict order.
    """
    names = ["A", "B", "C"]
    scores = {
        "A": {"A": 0.5, "B": 0.8, "C": 0.95},
        "B": {"A": 0.2, "B": 0.5, "C": 0.8},
        "C": {"A": 0.05, "B": 0.2, "C": 0.5},
    }
    games = {a: {b: 10 for b in names} for a in names}
    elo = bradley_terry_elo(names, scores, games)
    assert elo["A"] > elo["B"] > elo["C"]
    assert sum(elo.values()) == pytest.approx(0.0, abs=1e-6)  # centered


def test_bradley_terry_keeps_undefeated_rating_finite() -> None:
    """
    The prior keeps a 100%-win policy at a finite (if high) rating, not infinity.
    """
    names = ["A", "B"]
    scores = {"A": {"A": 0.5, "B": 1.0}, "B": {"A": 0.0, "B": 0.5}}
    games = {a: {b: 10 for b in names} for a in names}
    elo = bradley_terry_elo(names, scores, games)
    assert elo["A"] > elo["B"]
    assert math.isfinite(elo["A"]) and math.isfinite(elo["B"])


def test_play_match_resolves_a_battle() -> None:
    """
    A random-vs-random battle on the example deck resolves within the cap.
    """
    result = play_match(
        BattleHandle(), DECK, DECK, RandomOpponent(1), RandomOpponent(2)
    )
    assert result is not None  # a seat won or the game drew, but it ended


def test_play_series_tallies_partition_the_resolved_games() -> None:
    """
    A series' win/draw/loss counts sum to the games that resolved (<= played).
    """
    result = play_series(
        BattleHandle(),
        RandomOpponent(1),
        RandomOpponent(2),
        _mirror_sampler(),
        n_games=4,
        rng=random.Random(0),
    )
    assert result.wins + result.draws + result.losses == result.scored
    assert result.scored <= 4


def test_crossplay_matrix_is_complementary_with_even_diagonal() -> None:
    """
    Every off-diagonal cell and its mirror sum to 1, and self-play cells are 0.5.
    """
    policies = {
        "p1": RandomOpponent(1),
        "p2": RandomOpponent(2),
        "p3": RandomOpponent(3),
    }
    names, scores, games = crossplay_matrix(
        policies, _mirror_sampler, n_games=3, seed=0
    )
    assert names == ["p1", "p2", "p3"]
    for a in names:
        assert scores[a][a] == 0.5
        for b in names:
            if a != b:
                assert scores[a][b] + scores[b][a] == pytest.approx(1.0)
                assert games[a][b] == games[b][a]
