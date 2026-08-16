"""
Play two checkpoints against each other in the engine.

Validation accuracy says how often a clone reproduces an expert's move, which
is not the same question as whether it wins. This plays real games with seats
alternated, so neither side gets the first-player advantage for free.
"""

import argparse
import sys
from pathlib import Path

from src.env.battle_handle import BattleHandle
from src.env.decks.deck import load_deck
from src.policies.greedy_policy_opponent import GreedyPolicyOpponent

# lookahead.py sits next to this file, so resolve it from here rather than from
# the working directory: these tools are run from the repo root and from their
# own directory both.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lookahead import build_network


def play(deck_a, deck_b, policy_a, policy_b, seat_a: int) -> int:
    """
    Play one battle and report the winning seat.

    :param deck_a: Decklist for policy A.
    :param deck_b: Decklist for policy B.
    :param policy_a: Callable answering a selection for A.
    :param policy_b: Callable answering a selection for B.
    :param seat_a: Seat index policy A occupies.
    :return: Winning seat index, or -1 when the battle did not resolve.
    """
    handle = BattleHandle()
    try:
        decks = [None, None]
        decks[seat_a] = deck_a
        decks[1 - seat_a] = deck_b
        observation = handle.start(decks[0], decks[1])
        for _ in range(2000):
            state = observation.current
            if state is None or state.result != -1 or observation.select is None:
                break
            actor = policy_a if state.yourIndex == seat_a else policy_b
            observation = handle.select(actor(observation))
        return observation.current.result if observation.current else -1
    finally:
        handle.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="Checkpoint under test.")
    parser.add_argument(
        "--b",
        default="outputs/deck-pinned-150m-local/tf-ptr-pinned-selfplay-10m-s42/"
        "checkpoints/snapshot_000175702016.pt",
        help="Baseline checkpoint.",
    )
    parser.add_argument(
        "--deck",
        default="decks/top20/alakazam-dudunsparce/alakazam-dudunsparce-4.csv",
        help="Deck for A, and for B unless --deck-b is given.",
    )
    parser.add_argument(
        "--deck-b",
        default=None,
        help="Deck for B. Use this to let each side pilot the deck it is best "
        "at, which is the matchup a submission actually faces.",
    )
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    network_a, encoder_a = build_network(args.a)
    network_b, encoder_b = build_network(args.b)
    policy_a = GreedyPolicyOpponent(network_a, encoder_a)
    policy_b = GreedyPolicyOpponent(network_b, encoder_b)
    deck_a = load_deck(args.deck)
    deck_b = load_deck(args.deck_b) if args.deck_b else deck_a

    wins = losses = draws = 0
    for game in range(args.games):
        # Alternating rather than random keeps the seat split exactly even.
        seat_a = game % 2
        result = play(deck_a, deck_b, policy_a, policy_b, seat_a)
        if result == seat_a:
            wins += 1
        elif result == 1 - seat_a:
            losses += 1
        else:
            draws += 1
        if (game + 1) % 10 == 0:
            played = wins + losses + draws
            print(
                f"  {played} games: W{wins} L{losses} D{draws} "
                f"score {(wins + 0.5 * draws) / played:.3f}",
                flush=True,
            )

    played = max(wins + losses + draws, 1)
    score = (wins + 0.5 * draws) / played
    stderr = (0.25 / played) ** 0.5
    print(f"\nA: {args.a}")
    print(f"B: {args.b}")
    print(f"deck A: {args.deck}")
    print(f"deck B: {args.deck_b or args.deck}")
    print(f"games : {played}")
    print(f"score : {score:.3f} +/- {stderr:.3f}  (0.50 = even)")
    print(f"W/L/D : {wins}/{losses}/{draws}")


if __name__ == "__main__":
    main()
