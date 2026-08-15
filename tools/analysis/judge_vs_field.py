"""
Score policies by how they do against a common opponent, not against each other.

Head-to-head cross-play on one decklist is a mirror match, and the RL
checkpoint was trained with ``agent_deck_mirror: true`` -- mirrors are the only
thing it has ever played. The experts never played one. This instead has every
policy pilot the same deck against the same third-party opponent drawn from the
field, so the number is an absolute win rate rather than a matchup that favours
one side's training distribution.
"""
import argparse
import glob
import random

import torch
from omegaconf import OmegaConf
from torchrl.data import Binary, Categorical, Composite, Unbounded

from src.env.battle_handle import BattleHandle
from src.env.decks.deck import load_deck
from src.env.observation.structured_observation_encoder import (
    StructuredObservationEncoder,
)
from src.policies.greedy_policy_opponent import GreedyPolicyOpponent
from src.policies.ppo_actor import build_actor_critic
from src.training.cross_play import play_match

MAX_OPTIONS = 128


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", default="decks/expert/flg.csv")
    parser.add_argument(
        "--opponent",
        default="outputs/weighted-field-20260808/tf-ptr-weighted-15m-s42/"
        "checkpoints/snapshot_000030064640.pt",
        help="Common third party both contestants face.",
    )
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("contestants", nargs="+")
    args = parser.parse_args()

    deck = load_deck(args.deck)
    field = [
        load_deck(path)
        for path in sorted(glob.glob("decks/top20/**/*.csv", recursive=True))
    ]
    opponent = build_policy(args.opponent)
    print(f"deck under test : {args.deck}")
    print(f"common opponent : {args.opponent.split('/')[-1]} on {len(field)} field decks")
    print(f"\n{args.games} games each; higher is better, both face identical draws\n")
    print(f"{'contestant':44} {'win rate':>10} {'W-L':>10}")

    handle = BattleHandle()
    try:
        for path in args.contestants:
            policy = build_policy(path)
            wins = losses = draws = 0
            rng = random.Random(31)
            for game in range(args.games):
                # Same seed sequence per contestant, so both meet the same
                # opposing decks and the same seat assignment.
                opposing = field[rng.randrange(len(field))]
                our_seat = rng.randint(0, 1)
                deck0, deck1 = (deck, opposing) if our_seat == 0 else (opposing, deck)
                seat0, seat1 = (
                    (policy, opponent) if our_seat == 0 else (opponent, policy)
                )
                result = play_match(
                    handle, deck0, deck1, seat0, seat1, max_selections=1500
                )
                if result is None:
                    continue
                if result == our_seat:
                    wins += 1
                elif result == 1 - our_seat:
                    losses += 1
                else:
                    draws += 1
            scored = wins + losses + draws
            rate = (wins + 0.5 * draws) / max(scored, 1)
            name = path.split("/")[-1]
            print(f"{name[:44]:44} {rate:>10.3f} {f'{wins}-{losses}':>10}", flush=True)
    finally:
        handle.finish()


if __name__ == "__main__":
    main()
