"""
Test whether the self-play ladder is transitive.

Self-play reports progress as "the new policy beats the old one". That is only
evidence of increasing strength if the ladder is transitive: if a later
checkpoint beats an earlier one, it should also beat everything that earlier
one beat. This plays a full round robin between checkpoints spanning the run
and reports the score matrix, the Bradley-Terry ratings and every intransitive
triple it finds.
"""
import itertools
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
from src.training.cross_play import bradley_terry_elo, crossplay_matrix

MAX_OPTIONS = 128
DECK = "decks/top20/alakazam-dudunsparce/alakazam-dudunsparce-4.csv"

WEIGHTED = "outputs/weighted-field-20260808/tf-ptr-weighted-15m-s42/checkpoints"
PINNED = "outputs/deck-pinned-150m-local/tf-ptr-pinned-selfplay-10m-s42/checkpoints"

CANDIDATES = {
    "wf-030M": f"{WEIGHTED}/snapshot_000030015488.pt",
    "wf-060M": f"{WEIGHTED}/snapshot_000060030976.pt",
    "wf-090M": f"{WEIGHTED}/snapshot_000090046464.pt",
    "wf-120M": f"{WEIGHTED}/snapshot_000120061952.pt",
    "wf-150M": f"{WEIGHTED}/snapshot_000150011904.pt",
    "wf-181M": f"{WEIGHTED}/snapshot_000181518336.pt",
    "pin-173M": f"{PINNED}/snapshot_000173670400.pt",
}


def build_policy(path: str) -> GreedyPolicyOpponent:
    """
    Load a checkpoint as a greedy policy over its own trained network.

    :param path: Checkpoint file to load.
    :return: Greedy policy playing that checkpoint's weights.
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
    actor_critic = build_actor_critic(
        config, obs_spec, Categorical(n_actions, dtype=torch.int64)
    )
    actor_critic.load_state_dict(checkpoint["state_dict"], strict=True)
    return GreedyPolicyOpponent(actor_critic, encoder)


def nearest_existing(path: str) -> str | None:
    """
    Resolve a requested snapshot to the closest one that exists on disk.

    :param path: Desired checkpoint path.
    :return: An existing sibling snapshot, or None when the directory is empty.
    """
    target = Path(path)
    if target.exists():
        return str(target)
    siblings = sorted(target.parent.glob("snapshot_*.pt"))
    if not siblings:
        return None
    wanted = int(target.stem.split("_")[-1])
    return str(min(siblings, key=lambda p: abs(int(p.stem.split("_")[-1]) - wanted)))


def main() -> None:
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    resolved: dict[str, str] = {}
    for name, path in CANDIDATES.items():
        found = nearest_existing(path)
        if found is None:
            print(f"skip {name}: no snapshots in {Path(path).parent}")
            continue
        resolved[name] = found
        print(f"{name:10} {Path(found).name}")

    policies = {name: build_policy(path) for name, path in resolved.items()}
    deck = load_deck(DECK)
    names, scores, games = crossplay_matrix(
        policies,
        lambda: FixedDeckSampler(deck, deck),
        n_games=n_games,
        seed=7,
    )

    print(f"\nscore matrix (row's win-equivalent rate vs column, {n_games} games/pair)")
    header = "".join(f"{n:>10}" for n in names)
    print(f"{'':10}{header}")
    for a in names:
        row = "".join(f"{scores[a][b]:>10.2f}" for b in names)
        print(f"{a:10}{row}")

    elo = bradley_terry_elo(names, scores, games)
    print("\nBradley-Terry rating")
    for name, rating in sorted(elo.items(), key=lambda kv: -kv[1]):
        print(f"  {name:10} {rating:8.1f}")

    print("\nintransitive triples (A>B, B>C, but C>=A):")
    found = 0
    for a, b, c in itertools.permutations(names, 3):
        if scores[a][b] > 0.5 and scores[b][c] > 0.5 and scores[c][a] >= 0.5:
            print(f"  {a} > {b} ({scores[a][b]:.2f}), {b} > {c} ({scores[b][c]:.2f}), "
                  f"{c} >= {a} ({scores[c][a]:.2f})")
            found += 1
    total = len(names) * (len(names) - 1) * (len(names) - 2)
    print(f"  {found} of {total} ordered triples are intransitive")


if __name__ == "__main__":
    main()
