"""
Test whether the pinned arm's advantage is real skill or home-turf advantage.

pin-173M trained 23M frames on the alakazam-dudunsparce-4 mirror and scored
0.85 to 0.97 against the whole weighted-field lineage in a round robin played
on that same list. This replays the head-to-head on decks it never trained on.
If the margin survives, the run learned to play; if it collapses to ~0.5, the
margin was the training distribution.
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
# Defaults for the runs this measurement came from, all overridable. They point
# into outputs/ and decks/, neither of which is in the repository, so on any
# other checkout every one of these has to be passed explicitly.
DEFAULT_SUBJECT = (
    "outputs/deck-pinned-150m-local/tf-ptr-pinned-selfplay-10m-s42/checkpoints/"
    "snapshot_000173670400.pt"
)
DEFAULT_OPPONENTS = (
    "outputs/weighted-field-20260808/tf-ptr-weighted-15m-s42/checkpoints/"
    "snapshot_000150011904.pt",
    "outputs/weighted-field-20260808/tf-ptr-weighted-15m-s42/checkpoints/"
    "snapshot_000181518336.pt",
)
# The first entry is the subject's home deck, the rest are lists it never
# trained on. That contrast is the whole measurement.
DEFAULT_DECKS = (
    "decks/top20/alakazam-dudunsparce/alakazam-dudunsparce-4.csv",
    "decks/top20/dragapult-dudunsparce",
    "decks/top20/rockets-honchkrow",
    "decks/top20/lucario-hariyama",
)


def build_policy(path: Path) -> GreedyPolicyOpponent:
    """
    Rebuild a checkpoint as a greedy policy, using its own embedded config.

    :param path: Snapshot to load.
    :return: Greedy policy over that checkpoint.
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


def resolve_deck(spec: str) -> str:
    """
    Accept either a decklist path or a directory holding one.

    :param spec: Path to a CSV or to an archetype directory.
    :return: A concrete decklist path.
    """
    path = Path(spec)
    return str(path) if path.is_file() else str(sorted(path.glob("*.csv"))[0])


def main() -> None:
    """
    Play the subject against each opponent on every deck and print the grid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument(
        "--subject",
        type=Path,
        default=Path(DEFAULT_SUBJECT),
        help="Snapshot under test, played as player A.",
    )
    parser.add_argument(
        "--opponents",
        type=Path,
        nargs="+",
        default=[Path(spec) for spec in DEFAULT_OPPONENTS],
        help="Snapshots to score it against, one column each.",
    )
    parser.add_argument(
        "--decks",
        nargs="+",
        default=list(DEFAULT_DECKS),
        help="Decks to replay on, home deck first. A file, or a directory to "
        "take the first list from.",
    )
    args = parser.parse_args()

    subject = build_policy(args.subject)
    opponents = [build_policy(path) for path in args.opponents]
    handle = BattleHandle()
    print(
        f"{args.subject.stem} as player A, {args.games} games per cell, "
        "mirror matchups\n"
    )
    header = "".join(f"{path.stem[-12:]:>16}" for path in args.opponents)
    print(f"{'deck':42}{header}")
    try:
        for spec in args.decks:
            deck = load_deck(resolve_deck(spec))
            cells = []
            for opponent in opponents:
                result = play_series(
                    handle,
                    subject,
                    opponent,
                    FixedDeckSampler(deck, deck),
                    args.games,
                    random.Random(5),
                )
                cells.append(f"{result.score:>16.2f}")
            print(f"{Path(spec).stem:42}{''.join(cells)}")
    finally:
        handle.finish()


if __name__ == "__main__":
    main()
