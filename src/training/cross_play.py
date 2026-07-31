import itertools
import logging
import math
import random
from collections.abc import Callable, Mapping, Sequence

from cg.api import Observation
from src.env.battle_handle import BattleHandle
from src.env.deck_sampler import DeckSampler

logger = logging.getLogger(__name__)

Policy = Callable[[Observation], list[int]]


class MatchResult:
    """
    Aggregate outcome of a series of games between two policies, from A's view.

    Draws count as half a win for both sides, the standard scoring for
    pairwise-comparison rankings, so ``score`` is directly usable as A's
    expected result against B.
    """

    def __init__(self, wins: int, draws: int, losses: int) -> None:
        """
        :param wins: Games A won.
        :param draws: Games that ended without a decisive result.
        :param losses: Games A lost.
        """
        self.wins = wins
        self.draws = draws
        self.losses = losses

    @property
    def scored(self) -> int:
        """
        :return: Games that produced an outcome (excludes unresolved games).
        """
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """
        :return: A's win-equivalents over scored games (draws count as a half),
            or ``0.5`` when no game resolved so an empty cell reads as a tie.
        """
        if self.scored == 0:
            return 0.5
        return (self.wins + 0.5 * self.draws) / self.scored


def play_match(
        handle: BattleHandle,
        deck0: list[int],
        deck1: list[int],
        policy0: Policy,
        policy1: Policy,
        max_selections: int = 5000,
) -> int | None:
    """
    Play one battle to the end, each seat driven by its own policy.

    :param handle: An idle battle handle; finished before returning.
    :param deck0: 60 card IDs for seat 0.
    :param deck1: 60 card IDs for seat 1.
    :param policy0: Policy driving seat 0's selections.
    :param policy1: Policy driving seat 1's selections.
    :param max_selections: Safety cap on engine selections before giving up.
    :return: The winning seat (``0`` or ``1``), another value for a draw, or
        ``None`` if the game did not resolve within ``max_selections``.
    """
    policies = (policy0, policy1)
    observation = handle.start(deck0, deck1)
    try:
        for _ in range(max_selections):
            state = observation.current
            if state is not None and state.result != -1:
                return state.result
            select = observation.select
            if select is None:
                raise RuntimeError("Engine returned no selection while the battle is running.")
            if select.maxCount == 0:
                observation = handle.select([])
                continue
            seat = handle.select_player
            observation = handle.select(policies[seat](observation))
        return None
    finally:
        handle.finish()


def play_series(
        handle: BattleHandle,
        policy_a: Policy,
        policy_b: Policy,
        deck_sampler: DeckSampler,
        n_games: int,
        rng: random.Random,
        max_selections: int = 5000,
) -> MatchResult:
    """
    Play ``n_games`` between two policies, alternating which seat A takes.

    A's seat is randomized each game so the result is not confounded by any
    first/second-player advantage.

    :param handle: Battle handle reused across the series.
    :param policy_a: Policy whose results are tallied.
    :param policy_b: Opposing policy.
    :param deck_sampler: Source of each game's ``(deck0, deck1)`` matchup.
    :param n_games: Games to play.
    :param rng: Randomness for A's seat assignment.
    :param max_selections: Per-game engine-selection cap.
    :return: A's aggregate result over the games that resolved.
    """
    wins = draws = losses = 0
    for _ in range(n_games):
        deck0, deck1 = deck_sampler.sample()
        a_seat = rng.randint(0, 1)
        seat0, seat1 = (policy_a, policy_b) if a_seat == 0 else (policy_b, policy_a)
        result = play_match(handle, deck0, deck1, seat0, seat1, max_selections)
        if result is None:
            continue
        if result == a_seat:
            wins += 1
        elif result == 1 - a_seat:
            losses += 1
        else:
            draws += 1
    return MatchResult(wins, draws, losses)


def crossplay_matrix(
        policies: Mapping[str, Policy],
        deck_sampler_factory: Callable[[], DeckSampler],
        n_games: int = 20,
        seed: int = 0,
        max_selections: int = 5000,
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    """
    Score every pair of policies against each other.

    Each unordered pair plays one series; the reverse cell is the complement, so
    the engine cost is ``N*(N-1)/2`` series rather than ``N**2``. The diagonal is
    ``0.5``. A fresh, identically seeded deck sampler per pair keeps every pairing
    on a comparable deck sequence.

    :param policies: Named policies to rank (e.g. ``{"frames_000...": greedy}``).
    :param deck_sampler_factory: Builds a fresh deck sampler per pairing.
    :param n_games: Games per pairing.
    :param seed: Base seed for seat assignment.
    :param max_selections: Per-game engine-selection cap.
    :return: ``(names, score_matrix, games_matrix)`` where ``score_matrix[a][b]``
        is A's win-equivalent rate vs B and ``games_matrix[a][b]`` the games that
        resolved, both symmetric-by-construction with a ``0.5`` diagonal.
    """
    names = list(policies)
    scores: dict[str, dict[str, float]] = {a: {b: 0.5 for b in names} for a in names}
    games: dict[str, dict[str, int]] = {a: {b: 0 for b in names} for a in names}
    handle = BattleHandle()
    try:
        for offset, (a, b) in enumerate(itertools.combinations(names, 2)):
            sampler = deck_sampler_factory()
            result = play_series(
                handle,
                policies[a],
                policies[b],
                sampler,
                n_games,
                random.Random(seed + offset),
                max_selections,
            )
            scores[a][b] = result.score
            scores[b][a] = 1.0 - result.score
            games[a][b] = games[b][a] = result.scored
            logger.info(
                "Cross-play %s vs %s over %d game(s): score=%.3f",
                a, b, result.scored, result.score,
            )
    finally:
        handle.finish()
    return names, scores, games


def bradley_terry_elo(
        names: Sequence[str],
        scores: Mapping[str, Mapping[str, float]],
        games: Mapping[str, Mapping[str, int]],
        prior_games: float = 1.0,
        iterations: int = 200,
) -> dict[str, float]:
    """
    Fit Elo ratings from a cross-play score matrix (Bradley-Terry, order-free).

    :param names: Policies to rate.
    :param scores: ``scores[a][b]`` = A's win-equivalent rate vs B.
    :param games: ``games[a][b]`` = games that resolved between A and B.
    :param prior_games: Virtual even-result games added to each pairing to
        regularize the fit.
    :param iterations: MM iterations; the fit converges well within the default.
    :return: Elo rating per policy, centered so the mean rating is ``0``.
    """
    wins: dict[str, dict[str, float]] = {a: {} for a in names}
    played: dict[str, dict[str, float]] = {a: {} for a in names}
    for a, b in itertools.combinations(names, 2):
        n = games[a][b] + prior_games
        w_ab = scores[a][b] * games[a][b] + 0.5 * prior_games
        wins[a][b] = w_ab
        wins[b][a] = n - w_ab
        played[a][b] = played[b][a] = n

    strength = {name: 1.0 for name in names}
    for _ in range(iterations):
        updated: dict[str, float] = {}
        for a in names:
            total_wins = sum(wins[a][b] for b in names if b != a)
            denom = sum(
                played[a][b] / (strength[a] + strength[b])
                for b in names
                if b != a
            )
            updated[a] = total_wins / denom if denom > 0 else strength[a]
        # Normalize by the geometric mean so the scale cannot drift each pass.
        log_mean = sum(math.log(value) for value in updated.values()) / len(updated)
        norm = math.exp(log_mean)
        strength = {name: value / norm for name, value in updated.items()}

    return {name: 400.0 * math.log10(strength[name]) for name in names}
