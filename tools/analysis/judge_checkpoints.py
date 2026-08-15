"""
Cross-play any two checkpoint files over several decks.

Takes checkpoint paths directly rather than run directories, so a behaviour
cloned policy (a single file, no training run behind it) can be judged in the
same harness as the RL arms. Reports the challenger's score against the
reference on each deck, with an even split meaning no difference.
"""
import argparse
import random
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchrl.data import Binary, Categorical, Composite, Unbounded

from src.env.battle_handle import BattleHandle
from src.env.decks.deck import load_deck
from src.env.decks.deck_sampler import FixedDeckSampler
from src.env.observation.structured_observation_encoder import (
    StructuredObservationEncoder,
)
from src.policies.greedy_policy_opponent import GreedyPolicyOpponent
from src.policies.ppo_actor import build_actor_critic
from src.training.cross_play import play_series

MAX_OPTIONS = 128
#: The archetypes the top-8 teams actually pilot, which is what a clone of
#: them can be fairly asked to play, plus our own deck as the control.
#: The exact 60-card lists the top teams piloted, reconstructed from their
#: replays. Evaluating on a corpus list of the same archetype is not the same
#: test: the nearest corpus list overlaps flg's real deck at only 0.92, so the
#: clone would be asked to pilot roughly five cards it never trained with.
DECKS = {
    "flg exact (100 games)": "decks/expert/flg.csv",
    "LiamK exact (122 games)": "decks/expert/LiamK.csv",
}


def build_policy(path: str) -> GreedyPolicyOpponent:
    """
    Rebuild a checkpoint as a greedy policy from its embedded config.

    :param path: Checkpoint file.
    :return: Greedy policy over those weights.
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
    return GreedyPolicyOpponent(network, encoder)


def resolve(spec: str) -> str:
    """
    Accept a decklist path or a directory holding one.

    :param spec: Path to a CSV or archetype directory.
    :return: A concrete decklist path.
    """
    path = Path(spec)
    return str(path) if path.is_file() else str(sorted(path.glob("*.csv"))[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("challenger")
    parser.add_argument(
        "--reference",
        default="outputs/deck-pinned-150m-local/tf-ptr-pinned-selfplay-10m-s42/"
        "checkpoints/snapshot_000175702016.pt",
    )
    parser.add_argument("--games", type=int, default=40)
    args = parser.parse_args()

    challenger = build_policy(args.challenger)
    reference = build_policy(args.reference)
    handle = BattleHandle()
    print(f"challenger : {args.challenger}")
    print(f"reference  : {args.reference}")
    print(f"\n{args.games} games per deck; 0.50 means no difference\n")
    print(f"{'deck':28} {'challenger score':>18} {'W-L':>10}")
    try:
        for label, spec in DECKS.items():
            deck = load_deck(resolve(spec))
            # Cap the per-game selection budget: a clone that stalls can push a
            # game to the 5000-selection default and dominate wall-clock.
            result = play_series(
                handle, challenger, reference, FixedDeckSampler(deck, deck),
                args.games, random.Random(23), max_selections=1500,
            )
            print(f"{label:28} {result.score:>18.3f} "
                  f"{f'{result.wins}-{result.losses}':>10}", flush=True)
    finally:
        handle.finish()


if __name__ == "__main__":
    main()
