"""Shared helpers for the head-to-head and search tools."""

import collections
import random

import torch
from omegaconf import OmegaConf
from torchrl.data import Binary, Categorical, Composite, Unbounded

from cg.api import Observation
from src.env.observation.structured_observation_encoder import (
    StructuredObservationEncoder,
)
from src.policies.ppo_actor import build_actor_critic

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
