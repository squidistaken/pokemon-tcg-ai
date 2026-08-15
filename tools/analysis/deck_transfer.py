"""
Test whether the pinned arm's advantage is real skill or home-turf advantage.

pin-173M trained 23M frames on the alakazam-dudunsparce-4 mirror and scored
0.85 to 0.97 against the whole weighted-field lineage in a round robin played
on that same list. This replays the head-to-head on decks it never trained on.
If the margin survives, the run learned to play; if it collapses to ~0.5, the
margin was the training distribution.
"""
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchrl.data import Binary, Categorical, Composite, Unbounded

from src.env.decks.deck import load_deck
from src.env.decks.deck_sampler import FixedDeckSampler
from src.env.observation.structured_observation_encoder import (
    StructuredObservationEncoder,
)
from src.policies.greedy_policy_opponent import GreedyPolicyOpponent
from src.policies.ppo_actor import build_actor_critic
from src.training.cross_play import play_series
from src.env.battle_handle import BattleHandle
import random

MAX_OPTIONS = 128
WEIGHTED = Path("outputs/weighted-field-20260808/tf-ptr-weighted-15m-s42/checkpoints")
PINNED = Path("outputs/deck-pinned-150m-local/tf-ptr-pinned-selfplay-10m-s42/checkpoints")

DECKS = {
    "alakazam-dudunsparce-4 (pin's home deck)":
        "decks/top20/alakazam-dudunsparce/alakazam-dudunsparce-4.csv",
    "dragapult-dudunsparce":
        "decks/top20/dragapult-dudunsparce",
    "rockets-honchkrow":
        "decks/top20/rockets-honchkrow",
    "lucario-hariyama":
        "decks/top20/lucario-hariyama",
}


def build_policy(path: Path) -> GreedyPolicyOpponent:
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
    return GreedyPolicyOpponent(network, encoder)


def resolve_deck(spec: str) -> str:
    path = Path(spec)
    if path.is_file():
        return str(path)
    lists = sorted(path.glob("*.csv"))
    return str(lists[0])


def main() -> None:
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    pin = build_policy(PINNED / "snapshot_000173670400.pt")
    wf150 = build_policy(WEIGHTED / "snapshot_000150011904.pt")
    wf181 = build_policy(WEIGHTED / "snapshot_000181518336.pt")
    handle = BattleHandle()
    print(f"pin-173M as player A, {n_games} games per cell, mirror matchups\n")
    print(f"{'deck':42} {'vs wf-150M':>12} {'vs wf-181M':>12}")
    try:
        for label, spec in DECKS.items():
            deck = load_deck(resolve_deck(spec))
            row = []
            for opponent in (wf150, wf181):
                result = play_series(
                    handle, pin, opponent, FixedDeckSampler(deck, deck),
                    n_games, random.Random(5),
                )
                row.append(result.score)
            print(f"{label:42} {row[0]:>12.2f} {row[1]:>12.2f}")
    finally:
        handle.finish()


if __name__ == "__main__":
    main()
