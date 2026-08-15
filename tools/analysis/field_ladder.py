"""
Rank the weighted-field lineage on the field it was actually trained on.

The mirror round robin ran on one pinned decklist, which is the pinned arm's
training distribution and none of the weighted-field checkpoints'. This repeats
the round robin with independent draws from the full top20 corpus, the
distribution those checkpoints trained against, so the comparison is fair to
them.
"""
import glob
import itertools
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchrl.data import Binary, Categorical, Composite, Unbounded

from src.env.decks.deck import load_deck
from src.env.decks.deck_sampler import PoolDeckSampler
from src.env.observation.structured_observation_encoder import (
    StructuredObservationEncoder,
)
from src.policies.greedy_policy_opponent import GreedyPolicyOpponent
from src.policies.ppo_actor import build_actor_critic
from src.training.cross_play import bradley_terry_elo, crossplay_matrix

MAX_OPTIONS = 128
WEIGHTED = Path("outputs/weighted-field-20260808/tf-ptr-weighted-15m-s42/checkpoints")


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


def nearest(frames: int) -> Path:
    snaps = sorted(WEIGHTED.glob("snapshot_*.pt"))
    return min(snaps, key=lambda p: abs(int(p.stem.split("_")[-1]) - frames))


def main() -> None:
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    wanted = [10_000_000, 30_000_000, 75_000_000, 120_000_000, 150_000_000, 181_000_000]
    chosen: dict[str, Path] = {}
    for frames in wanted:
        path = nearest(frames)
        actual = int(path.stem.split("_")[-1])
        name = f"wf-{actual/1e6:06.1f}M"
        if name not in chosen:
            chosen[name] = path
    for name, path in chosen.items():
        print(f"{name}: {path.name}")

    decks = [
        load_deck(p) for p in sorted(glob.glob("decks/top20/**/*.csv", recursive=True))
    ]
    print(f"field: {len(decks)} lists, independent draws\n")
    policies = {name: build_policy(path) for name, path in chosen.items()}
    names, scores, games = crossplay_matrix(
        policies,
        lambda: PoolDeckSampler(decks, matchup="independent", mode="uniform", seed=11),
        n_games=n_games,
        seed=3,
    )
    print(f"score matrix ({n_games} games/pair, independent top20 draws)")
    print(f"{'':12}" + "".join(f"{n:>12}" for n in names))
    for a in names:
        print(f"{a:12}" + "".join(f"{scores[a][b]:>12.2f}" for b in names))
    print("\nBradley-Terry rating")
    for name, rating in sorted(bradley_terry_elo(names, scores, games).items(), key=lambda kv: -kv[1]):
        print(f"  {name:12} {rating:8.1f}")
    bad = sum(
        1
        for a, b, c in itertools.permutations(names, 3)
        if scores[a][b] > 0.5 and scores[b][c] > 0.5 and scores[c][a] >= 0.5
    )
    print(f"\nintransitive ordered triples: {bad}")


if __name__ == "__main__":
    main()
