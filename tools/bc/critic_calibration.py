"""
Check whether the critic predicts outcomes on the states search asks it about.

Training only ever shows the critic states where our seat is about to act,
because TCGEnv plays the opponent's moves internally. A one-ply search scores
the state *after* our move, which is usually a state where the opponent acts
next. This plays self-play games, records the critic's value at both kinds of
state, and correlates each against the game's actual result.
"""

import argparse
import random

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.data import Binary, Categorical, Composite, Unbounded

from src.env.battle_handle import BattleHandle
from src.env.decks.deck import load_deck
from src.env.observation.structured_observation_encoder import (
    StructuredObservationEncoder,
)
from src.policies.greedy_policy_opponent import GreedyPolicyOpponent
from src.policies.ppo_actor import build_actor_critic

MAX_OPTIONS = 128


def build(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = OmegaConf.create(checkpoint["config"])
    encoder = StructuredObservationEncoder(max_options=MAX_OPTIONS)
    n = MAX_OPTIONS + 1
    obs_spec = Composite(
        observation=encoder.spec(),
        action_mask=Binary(n=n, dtype=torch.bool),
        level_id=Unbounded(shape=(1,), dtype=torch.int64),
        opponent_is_anchor=Binary(1, dtype=torch.bool),
    )
    network = build_actor_critic(config, obs_spec, Categorical(n, dtype=torch.int64))
    network.load_state_dict(checkpoint["state_dict"], strict=True)
    return network.eval(), encoder


def correlation(values: list[float], outcomes: list[float]) -> float:
    if len(values) < 3:
        return float("nan")
    mean_v = sum(values) / len(values)
    mean_o = sum(outcomes) / len(outcomes)
    cov = sum(
        (v - mean_v) * (o - mean_o) for v, o in zip(values, outcomes, strict=True)
    )
    var_v = sum((v - mean_v) ** 2 for v in values) ** 0.5
    var_o = sum((o - mean_o) ** 2 for o in outcomes) ** 0.5
    return cov / (var_v * var_o) if var_v and var_o else float("nan")


@torch.inference_mode()
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
    args = parser.parse_args()

    network, encoder = build(args.checkpoint)
    policy = GreedyPolicyOpponent(network, encoder)
    deck = load_deck(args.deck)
    rng = random.Random(4)

    def value(observation, seat: int) -> float:
        td = TensorDict(
            {"observation": encoder.encode(observation, seat, 0)},
            batch_size=torch.Size(()),
        )
        network(td)
        return float(td.get("state_value").reshape(-1)[0])

    our_move, opp_move = ([], []), ([], [])
    for _game in range(args.games):
        our_seat = rng.randint(0, 1)
        handle = BattleHandle()
        samples: list[tuple[bool, float]] = []
        try:
            observation = handle.start(deck, deck)
            for _ in range(2000):
                state = observation.current
                if state is None or state.result != -1 or observation.select is None:
                    break
                acting = state.yourIndex
                # Score every state from our seat, tagging who is to move.
                samples.append((acting == our_seat, value(observation, our_seat)))
                observation = handle.select(policy(observation))
            result = observation.current.result if observation.current else -1
        finally:
            handle.finish()
        outcome = (
            1.0 if result == our_seat else (-1.0 if result == 1 - our_seat else 0.0)
        )
        for is_ours, v in samples:
            bucket = our_move if is_ours else opp_move
            bucket[0].append(v)
            bucket[1].append(outcome)

    print(f"games: {args.games}\n")
    print(f"{'state type':34} {'n':>7} {'mean V':>9} {'corr with result':>18}")
    for label, (values, outcomes) in (
        ("our seat to move (trained on)", our_move),
        ("opponent to move (search asks)", opp_move),
    ):
        print(
            f"{label:34} {len(values):>7} {sum(values) / max(len(values), 1):>9.3f} "
            f"{correlation(values, outcomes):>18.3f}"
        )


if __name__ == "__main__":
    main()
