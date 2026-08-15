"""
Check that the packaged Kaggle bundle plays the same moves as the trained policy.

Drives real battles with the training-side greedy policy and, at every agent
decision, asks the submission's torch-only runtime for its pick from the same
engine observation. Any disagreement means the deployed agent is not the agent
that was trained.
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchrl.data import Binary, Categorical, Composite, Unbounded

from src.env.battle_handle import BattleHandle
from src.env.decks.deck import load_deck
from src.env.observation.structured_observation_encoder import (
    StructuredObservationEncoder,
)
from src.env.opponents.random_opponent import RandomOpponent
from src.policies.greedy_policy_opponent import GreedyPolicyOpponent
from src.policies.ppo_actor import build_actor_critic
from submission.cg_api import to_observation_class
from submission.runtime import Policy

BUNDLE = Path("submissions/pinned-selfplay-158m-aladuduns4")
CHECKPOINT = (
    "outputs/deck-pinned-150m-local/tf-ptr-pinned-selfplay-10m-s42/"
    "checkpoints/snapshot_000158154752.pt"
)
DECK = "decks/top20/alakazam-dudunsparce/alakazam-dudunsparce-4.csv"
MAX_OPTIONS = 128


def load_bundle_policy() -> Policy:
    """
    Rebuild the exact policy the Kaggle bundle runs.

    :return: The bundle's torch-only inference policy.
    """
    config = json.loads((BUNDLE / "model_config.json").read_text())
    payload = torch.load(BUNDLE / "model.pt", map_location="cpu", weights_only=True)
    return Policy(payload, config)


def load_training_policy() -> GreedyPolicyOpponent:
    """
    Rebuild the training-side actor-critic from the same checkpoint.

    :return: A greedy policy over the training network and encoder.
    """
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = OmegaConf.create(checkpoint["config"])
    encoder = StructuredObservationEncoder(max_options=MAX_OPTIONS)
    n_actions = MAX_OPTIONS + 1
    obs_spec = Composite(
        observation=encoder.spec(),
        action_mask=Binary(n=n_actions, dtype=torch.bool),
        level_id=Unbounded(shape=(1,), dtype=torch.int64),
        opponent_is_anchor=Binary(1, dtype=torch.bool),
    )
    action_spec = Categorical(n_actions, dtype=torch.int64)
    actor_critic = build_actor_critic(config, obs_spec, action_spec)
    actor_critic.load_state_dict(checkpoint["state_dict"], strict=True)
    return GreedyPolicyOpponent(actor_critic, encoder)


def main() -> None:
    bundle_policy = load_bundle_policy()
    train_policy = load_training_policy()
    deck = load_deck(DECK)
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    agree = disagree = 0
    examples: list[tuple[int, list[int], list[int]]] = []
    for game in range(games):
        handle = BattleHandle()
        opponent = RandomOpponent(seed=1000 + game)
        observation = handle.start(deck, deck)
        try:
            for _ in range(600):
                state = observation.current
                if state is None or state.result != -1 or observation.select is None:
                    break
                if state.yourIndex == 0:
                    train_pick = list(train_policy(observation))
                    bundle_pick = list(
                        bundle_policy(to_observation_class(asdict(observation)))
                    )
                    if train_pick == bundle_pick:
                        agree += 1
                    else:
                        disagree += 1
                        if len(examples) < 6:
                            examples.append((game, train_pick, bundle_pick))
                    observation = handle.select(train_pick)
                else:
                    observation = handle.select(opponent(observation))
        finally:
            handle.finish()
    total = agree + disagree
    print(f"games played: {games}")
    print(f"agent decisions compared: {total}")
    print(f"  identical picks: {agree}")
    print(f"  divergent picks: {disagree} ({disagree / max(total, 1):.1%})")
    for game, train_pick, bundle_pick in examples:
        print(f"   game {game}: training {train_pick} vs bundle {bundle_pick}")


if __name__ == "__main__":
    main()
