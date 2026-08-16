"""
One-ply lookahead over the engine's determinized search, judged against the
raw policy.

The policy head scores an option from the current state alone. This instead
plays each legal option out in a forked copy of the battle
(``cg.api.search_begin`` / ``search_step``) and scores the *resulting* state
with the critic, which is the improvement operator plain PPO does not have.
Needs no retraining: it wraps a checkpoint that already exists.
"""

import argparse
import collections
import contextlib
import logging
import random

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.data import Binary, Categorical, Composite, Unbounded

from cg import api
from cg.api import Observation
from src.env.battle_handle import BattleHandle
from src.env.decks.deck import load_deck
from src.env.observation.structured_observation_encoder import (
    StructuredObservationEncoder,
)
from src.policies.greedy_policy_opponent import GreedyPolicyOpponent
from src.policies.ppo_actor import build_actor_critic

logger = logging.getLogger(__name__)

MAX_OPTIONS = 128


def build_network(path: str):
    """
    Rebuild a checkpoint's actor-critic and its encoder.

    :param path: Snapshot path.
    :return: The network in eval mode and the matching encoder.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = OmegaConf.create(checkpoint["config"])
    encoder = StructuredObservationEncoder(max_options=MAX_OPTIONS)
    n_actions = MAX_OPTIONS + 1
    obs_spec = Composite(
        observation=encoder.spec(),
        action_mask=Binary(n=n_actions, dtype=torch.bool),
        level_id=Unbounded(shape=(1,), dtype=torch.int64),
        opponent_is_anchor=Binary(1, dtype=torch.bool),
    )
    network = build_actor_critic(
        config, obs_spec, Categorical(n_actions, dtype=torch.int64)
    )
    network.load_state_dict(checkpoint["state_dict"], strict=True)
    return network.eval(), encoder


def _visible_cards(player) -> collections.Counter:
    """
    Count every card of one seat whose identity we can already see.

    :param player: A ``PlayerState``.
    :return: Multiset of visible card IDs.
    """
    seen: collections.Counter = collections.Counter()
    for card in player.hand or []:
        if card:
            seen[card.id] += 1
    for card in player.discard or []:
        if card:
            seen[card.id] += 1
    for mon in [m for m in (player.active or []) + (player.bench or []) if m]:
        seen[mon.id] += 1
        for group in (mon.energyCards or [], mon.tools or [], mon.preEvolution or []):
            for card in group:
                if card:
                    seen[card.id] += 1
    return seen


def determinize(
    observation: Observation,
    seat: int,
    own_list: list[int],
    opponent_list: list[int],
    rng: random.Random,
) -> dict:
    """
    Sample one consistent completion of the six hidden components.

    Our own decklist is known exactly, so the unseen part of it is the list
    minus what is already visible. The opponent's list is guessed from the
    corpus and completed the same way.

    :param observation: Live agent observation.
    :param seat: Our seat index.
    :param own_list: Our full 60-card decklist.
    :param opponent_list: A guessed 60-card list for the opponent.
    :param rng: Randomness for the shuffle.
    :return: Keyword arguments for :func:`cg.api.search_begin`.
    """
    state = observation.current
    me, them = state.players[seat], state.players[1 - seat]

    def hidden_pool(full: list[int], player) -> list[int]:
        remaining = collections.Counter(full) - _visible_cards(player)
        pool = list(remaining.elements())
        rng.shuffle(pool)
        return pool

    mine = hidden_pool(own_list, me)
    theirs = hidden_pool(opponent_list, them)
    need_mine = me.deckCount + len(me.prize or [])
    need_theirs = them.deckCount + len(them.prize or []) + (them.handCount or 0)
    # Pad from the full list rather than fail: a mis-guessed opponent list can
    # leave the pool short, and search_begin only requires "at least".
    while len(mine) < need_mine:
        mine.append(rng.choice(own_list))
    while len(theirs) < need_theirs:
        theirs.append(rng.choice(opponent_list))

    opponent_active = []
    actives = them.active or []
    if actives and actives[0] is None:
        opponent_active = [next((c for c in theirs if c), opponent_list[0])]
    return {
        "your_deck": mine[: me.deckCount],
        "your_prize": mine[me.deckCount : me.deckCount + len(me.prize or [])],
        "opponent_deck": theirs[: them.deckCount],
        "opponent_prize": theirs[
            them.deckCount : them.deckCount + len(them.prize or [])
        ],
        "opponent_hand": theirs[
            them.deckCount + len(them.prize or []) : them.deckCount
            + len(them.prize or [])
            + (them.handCount or 0)
        ],
        "opponent_active": opponent_active,
    }


class LookaheadPolicy:
    """
    Pick the option whose resulting state the critic likes best.

    Falls back to the wrapped greedy policy whenever a search cannot be opened
    (no ``search_begin_input``, an inconsistent determinization, or an engine
    error), so it is never worse than the policy it wraps by construction.
    """

    def __init__(
        self,
        network,
        encoder,
        own_list: list[int],
        opponent_list: list[int],
        seed: int = 0,
        minimize: bool = False,
    ) -> None:
        """
        :param network: Trained actor-critic supplying the value head.
        :param encoder: Observation encoder matching training.
        :param own_list: Our full decklist, for determinization.
        :param opponent_list: Guessed opponent decklist.
        :param seed: Seed for determinization shuffles.
        """
        self._network = network
        self._encoder = encoder
        self._own = own_list
        self._opponent = opponent_list
        self._rng = random.Random(seed)
        self._sign = -1.0 if minimize else 1.0
        self._fallback = GreedyPolicyOpponent(network, encoder)
        self.searched = 0
        self.fell_back = 0

    @torch.inference_mode()
    def _value(self, observation: Observation, seat: int) -> float:
        """
        Critic value of a state from our seat.

        :param observation: State to score.
        :param seat: Our seat index.
        :return: Scalar value estimate.
        """
        encoded = TensorDict(
            {"observation": self._encoder.encode(observation, seat, 0)},
            batch_size=torch.Size(()),
        )
        self._network(encoded)
        return float(encoded.get("state_value").reshape(-1)[0])

    def __call__(self, observation: Observation) -> list[int]:
        """
        Choose a selection, searching one ply when the engine allows it.

        :param observation: Live agent observation.
        :return: Option indices to submit.
        """
        select = observation.select
        state = observation.current
        if select is None or state is None or observation.search_begin_input is None:
            self.fell_back += 1
            return self._fallback(observation)
        seat = state.yourIndex
        n_options = len(select.option)
        if n_options < 2 or select.minCount != 1 or select.maxCount != 1:
            self.fell_back += 1
            return self._fallback(observation)
        try:
            root = api.search_begin(
                observation,
                **determinize(observation, seat, self._own, self._opponent, self._rng),
            )
        except Exception:
            logger.debug(
                "search_begin failed; falling back to the raw policy",
                exc_info=True,
            )
            self.fell_back += 1
            return self._fallback(observation)

        # The root survives being stepped, and every option must be judged
        # under the *same* sampled world or the ranking is determinization
        # noise rather than a comparison between actions.
        best_option, best_value = None, -float("inf")
        try:
            for option in range(min(n_options, MAX_OPTIONS)):
                try:
                    child = api.search_step(root.searchId, [option])
                except Exception:
                    logger.debug("search_step failed; skipping option", exc_info=True)
                    continue
                successor = child.observation
                if successor.current is not None:
                    value = self._sign * self._value(successor, seat)
                    if value > best_value:
                        best_option, best_value = option, value
                with contextlib.suppress(Exception):
                    api.search_release(child.searchId)
        finally:
            with contextlib.suppress(Exception):
                api.search_end()
        if best_option is None:
            self.fell_back += 1
            return self._fallback(observation)
        self.searched += 1
        return [best_option]


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
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Pick the lowest-valued successor, to test for an inverted ranking.",
    )
    args = parser.parse_args()

    network, encoder = build_network(args.checkpoint)
    deck = load_deck(args.deck)
    lookahead = LookaheadPolicy(
        network, encoder, deck, deck, seed=5, minimize=args.minimize
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
                actor = lookahead if state.yourIndex == our_seat else raw
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
        print(
            f"  game {game + 1}: W{wins} L{losses} D{draws} "
            f"(searched {lookahead.searched}, fell back {lookahead.fell_back})",
            flush=True,
        )

    scored = wins + losses + draws
    print("\n1-ply lookahead vs the same checkpoint playing raw argmax")
    print(f"  games   : {scored}")
    print(f"  score   : {(wins + 0.5 * draws) / max(scored, 1):.3f} (0.50 = no gain)")
    print(f"  W/L/D   : {wins}/{losses}/{draws}")
    total = lookahead.searched + lookahead.fell_back
    print(
        f"  searched: {lookahead.searched}/{total} decisions "
        f"({lookahead.searched / max(total, 1):.1%})"
    )


if __name__ == "__main__":
    main()
