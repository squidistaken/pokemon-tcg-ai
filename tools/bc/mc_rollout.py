"""
Rank actions by rolled-out game results instead of by the critic.

The critic cannot separate the options at a single decision: its spread across
them is 0.065 against a 0.549 spread across states, well inside its own error.
A rollout to terminal returns an actual win or loss, which has full per-action
resolution. Rollouts play randomly so the network stays out of the inner loop
and the cost is engine-only.
"""
import argparse
import collections
import random
import sys
from pathlib import Path

from cg import api
from cg.api import Observation
from src.env.battle_handle import BattleHandle
from src.env.decks.deck import load_deck
from src.policies.greedy_policy_opponent import GreedyPolicyOpponent

# lookahead.py sits next to this file, so resolve it from here rather than from
# the working directory: these tools are run from the repo root and from their
# own directory both.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lookahead import build_network, determinize  # noqa: E402


def random_playout(
    search_id: int, seat: int, rng: random.Random, max_steps: int = 600
) -> float | None:
    """
    Play a forked state to the end with uniform random legal selections.

    :param search_id: Search state to play out.
    :param seat: Our seat, for scoring the result.
    :param rng: Randomness for the selections.
    :param max_steps: Cap, so a pathological line cannot hang the search.
    :return: 1.0 win, 0.0 loss, 0.5 draw, or None if the line did not resolve.
    """
    current_id = search_id
    for _ in range(max_steps):
        try:
            state = api.search_step(current_id, _random_selection(current_id, rng))
        except Exception:
            return None
        observation = state.observation
        current_id = state.searchId
        if observation.current is None:
            return None
        result = observation.current.result
        if result != -1:
            if result == seat:
                return 1.0
            if result == 1 - seat:
                return 0.0
            return 0.5
        if observation.select is None:
            return None
        _PENDING[current_id] = observation
    return None


#: The selection for a search node has to be read from the observation that
#: produced it, which search_step returns rather than exposing by ID.
_PENDING: dict[int, Observation] = {}


def _random_selection(search_id: int, rng: random.Random) -> list[int]:
    """
    Draw a legal selection for a pending search node.

    :param search_id: Node to choose for.
    :param rng: Randomness.
    :return: Option indices satisfying minCount/maxCount without duplicates.
    """
    observation = _PENDING.get(search_id)
    if observation is None or observation.select is None:
        return [0]
    select = observation.select
    count = len(select.option)
    if count == 0:
        return []
    low = max(select.minCount, 0)
    high = min(select.maxCount, count)
    if high < low:
        high = low
    take = rng.randint(low, high) if high > low else low
    return rng.sample(range(count), min(take, count))


class RolloutPolicy:
    """
    Choose the option with the best mean rolled-out result.

    Falls back to the wrapped greedy policy whenever the search cannot be
    opened or no line resolves, so it is never worse by construction.
    """

    def __init__(self, network, encoder, own_list, opponent_list, rollouts=8, seed=0):
        """
        :param network: Trained actor-critic, used only for the fallback.
        :param encoder: Observation encoder matching training.
        :param own_list: Our full decklist, for determinization.
        :param opponent_list: Guessed opponent decklist.
        :param rollouts: Random playouts per option.
        :param seed: Seed for determinization and playouts.
        """
        self._own = own_list
        self._opponent = opponent_list
        self._rollouts = rollouts
        self._rng = random.Random(seed)
        self._fallback = GreedyPolicyOpponent(network, encoder)
        self.searched = 0
        self.fell_back = 0

    def __call__(self, observation: Observation) -> list[int]:
        """
        Pick a selection, using rollouts when the engine allows a search.

        :param observation: Live agent observation.
        :return: Option indices to submit.
        """
        select = observation.select
        state = observation.current
        if (
            select is None
            or state is None
            or observation.search_begin_input is None
            or len(select.option) < 2
            or select.minCount != 1
            or select.maxCount != 1
        ):
            self.fell_back += 1
            return self._fallback(observation)
        seat = state.yourIndex
        scores: dict[int, list[float]] = collections.defaultdict(list)
        try:
            for _ in range(self._rollouts):
                try:
                    root = api.search_begin(
                        observation,
                        **determinize(
                            observation, seat, self._own, self._opponent, self._rng
                        ),
                    )
                except Exception:
                    break
                for option in range(len(select.option)):
                    try:
                        child = api.search_step(root.searchId, [option])
                    except Exception:
                        continue
                    _PENDING[child.searchId] = child.observation
                    if child.observation.current is None:
                        continue
                    result = child.observation.current.result
                    if result != -1:
                        scores[option].append(
                            1.0 if result == seat else (0.0 if result == 1 - seat else 0.5)
                        )
                        continue
                    outcome = random_playout(child.searchId, seat, self._rng)
                    if outcome is not None:
                        scores[option].append(outcome)
                    try:
                        api.search_release(child.searchId)
                    except Exception:
                        pass
        finally:
            _PENDING.clear()
            try:
                api.search_end()
            except Exception:
                pass
        rated = {k: sum(v) / len(v) for k, v in scores.items() if v}
        if not rated:
            self.fell_back += 1
            return self._fallback(observation)
        self.searched += 1
        return [max(rated, key=lambda k: rated[k])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="outputs/deck-pinned-150m-local/tf-ptr-pinned-selfplay-10m-s42/"
        "checkpoints/snapshot_000175702016.pt",
    )
    parser.add_argument(
        "--deck", default="decks/top20/alakazam-dudunsparce/alakazam-dudunsparce-4.csv"
    )
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--rollouts", type=int, default=6)
    args = parser.parse_args()

    network, encoder = build_network(args.checkpoint)
    deck = load_deck(args.deck)
    searcher = RolloutPolicy(
        network, encoder, deck, deck, rollouts=args.rollouts, seed=5
    )
    raw = GreedyPolicyOpponent(network, encoder)

    wins = losses = draws = 0
    rng = random.Random(11)
    for game in range(args.games):
        our_seat = rng.randint(0, 1)
        handle = BattleHandle()
        try:
            observation = handle.start(deck, deck)
            for _ in range(2000):
                state = observation.current
                if state is None or state.result != -1 or observation.select is None:
                    break
                actor = searcher if state.yourIndex == our_seat else raw
                observation = handle.select(actor(observation))
            result = observation.current.result if observation.current else -1
        finally:
            handle.finish()
        if result == our_seat:
            wins += 1
        elif result == 1 - our_seat:
            losses += 1
        else:
            draws += 1
        print(f"  game {game + 1}: W{wins} L{losses} D{draws} "
              f"(searched {searcher.searched}, fell back {searcher.fell_back})",
              flush=True)

    scored = wins + losses + draws
    print(f"\nMC rollout search ({args.rollouts}/option) vs raw argmax, same weights")
    print(f"  games : {scored}")
    print(f"  score : {(wins + 0.5 * draws) / max(scored, 1):.3f} (0.50 = no gain)")
    print(f"  W/L/D : {wins}/{losses}/{draws}")


if __name__ == "__main__":
    main()
