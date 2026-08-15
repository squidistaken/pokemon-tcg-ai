"""
Score the experiment arms against the baseline they have to beat.

Each arm is judged head to head, not by its own training curve: the 50-episode
self-play eval is the instrument that missed a whole run's worth of trouble.
Every arm plays the 181M weighted-field checkpoint on the deck it trained with
and on decks it never saw, so a win that is only local shows up as a win only
on the training deck.
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
# The arms' own starting point. Scoring against anything else measures the
# init rather than the training: pin-175.7M already beats the weighted-field
# lineage ~0.90 on Alakazam before a single frame of these runs.
BASELINE = Path(
    "outputs/deck-pinned-150m-local/tf-ptr-pinned-selfplay-10m-s42/checkpoints/"
    "snapshot_000175702016.pt"
)
TRAINING_DECK = "decks/heuristic-resolved/alakazam-dudunsparce/alakazam-dudunsparce-4.csv"
HELD_OUT_DECKS = {
    "dragapult-dudunsparce": "decks/top20/dragapult-dudunsparce",
    "rockets-honchkrow": "decks/top20/rockets-honchkrow",
    "lucario-hariyama": "decks/top20/lucario-hariyama",
}


def build_policy(path: Path) -> GreedyPolicyOpponent:
    """
    Rebuild a checkpoint as a greedy policy, using its own embedded config.

    Each arm carries its own architecture in the snapshot, so the card-effect
    arms rebuild at their wider card representation without any extra flag.

    :param path: Snapshot to load.
    :return: Greedy policy over that checkpoint.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = OmegaConf.create(checkpoint["config"])
    effects = bool(config.model.get("adapter", {}).get("card_effect_features", False))
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
    print(f"  loaded {path.name} (card_effect_features={effects})")
    return GreedyPolicyOpponent(network, encoder)


def latest_snapshot(run_dir: Path) -> Path | None:
    """
    Find a run's final snapshot.

    :param run_dir: Hydra run directory for one arm.
    :return: The highest-frame snapshot, or None when the arm produced none.
    """
    snaps = sorted((run_dir / "checkpoints").glob("snapshot_*.pt"))
    return snaps[-1] if snaps else None


def resolve_deck(spec: str) -> str:
    """
    Accept either a decklist path or a directory holding one.

    :param spec: Path to a CSV or to an archetype directory.
    :return: A concrete decklist path.
    """
    path = Path(spec)
    return str(path) if path.is_file() else str(sorted(path.glob("*.csv"))[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="Hydra run directories to judge.")
    parser.add_argument("--games", type=int, default=40)
    args = parser.parse_args()

    print("baseline:")
    baseline = build_policy(BASELINE)
    arms: dict[str, GreedyPolicyOpponent] = {}
    for spec in args.run_dirs:
        run_dir = Path(spec)
        snapshot = latest_snapshot(run_dir)
        if snapshot is None:
            print(f"skip {run_dir}: no snapshots")
            continue
        print(f"{run_dir.name}:")
        arms[run_dir.name] = build_policy(snapshot)

    decks = {"TRAINING alakazam-dudunsparce-4": TRAINING_DECK} | HELD_OUT_DECKS
    handle = BattleHandle()
    print(f"\nscore vs their own init (pin-175.7M), {args.games} games per cell "
          f"(0.50 = no change, >0.50 = the arm is better)\n")
    header = "".join(f"{name[:20]:>22}" for name in arms)
    print(f"{'deck':32}{header}")
    try:
        for label, spec in decks.items():
            deck = load_deck(resolve_deck(spec))
            cells = []
            for policy in arms.values():
                result = play_series(
                    handle, policy, baseline, FixedDeckSampler(deck, deck),
                    args.games, random.Random(17),
                )
                cells.append(f"{result.score:>22.2f}")
            print(f"{label:32}{''.join(cells)}")
    finally:
        handle.finish()


if __name__ == "__main__":
    main()
