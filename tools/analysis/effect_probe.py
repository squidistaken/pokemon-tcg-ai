"""
Test whether the observation can predict what an action does.

Plays real battles and records, for every agent decision, the encoder's option
features for the chosen action and the state change that action produced. Then
fits ridge probes:

  A. option features only  -- what the network can compute from the observation
  B. card id one-hot       -- what a per-card learned embedding could memorize
  C. both

If A is near zero and B is high, the effect of a card is knowable only by
identity, so the network must memorize each card from reward rather than read
its effect off the observation.
"""
import sys

import numpy as np
import torch

from src.env.battle_handle import BattleHandle
from src.env.decks.deck import load_deck
from src.env.observation.structured_observation_encoder import (
    StructuredObservationEncoder,
)
from src.env.opponents.random_opponent import RandomOpponent

MAX_OPTIONS = 128
DECK = "decks/top20/alakazam-dudunsparce/alakazam-dudunsparce-4.csv"


def board_summary(state, seat: int) -> np.ndarray:
    """
    Summarize the observable board from one seat.

    :param state: Engine state.
    :param seat: Seat to summarize.
    :return: Vector of hand size, board HP, prize counts, energy and bench size.
    """
    me = state.players[seat]
    opp = state.players[1 - seat]

    def board_hp(player):
        mons = [m for m in ([*(player.active or []), *(player.bench or [])]) if m]
        return float(sum(m.hp for m in mons)), float(len(mons)), float(
            sum(len(m.energies or []) for m in mons)
        )

    my_hp, my_n, my_energy = board_hp(me)
    opp_hp, opp_n, opp_energy = board_hp(opp)
    return np.array(
        [
            float(me.handCount or 0),
            float(me.deckCount or 0),
            float(len(me.prize or [])),
            float(len(opp.prize or [])),
            float(len(me.discard or [])),
            my_hp, my_n, my_energy,
            opp_hp, opp_n, opp_energy,
        ],
        dtype=np.float64,
    )


def ridge_r2(features: np.ndarray, targets: np.ndarray, alpha: float = 1.0) -> float:
    """
    Fit a ridge regression and report out-of-sample R2 on a held-out half.

    :param features: Design matrix.
    :param targets: Multi-output targets.
    :param alpha: Ridge penalty.
    :return: Mean R2 across target columns, floored at 0.
    """
    n = len(features)
    split = n // 2
    order = np.random.RandomState(0).permutation(n)
    train, test = order[:split], order[split:]
    x_tr = np.hstack([features[train], np.ones((len(train), 1))])
    x_te = np.hstack([features[test], np.ones((len(test), 1))])
    y_tr, y_te = targets[train], targets[test]
    gram = x_tr.T @ x_tr + alpha * np.eye(x_tr.shape[1])
    weights = np.linalg.solve(gram, x_tr.T @ y_tr)
    pred = x_te @ weights
    ss_res = ((y_te - pred) ** 2).sum(axis=0)
    ss_tot = ((y_te - y_te.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-9)
    return float(np.clip(r2, 0.0, 1.0).mean())


def main() -> None:
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    encoder = StructuredObservationEncoder(max_options=MAX_OPTIONS)
    deck = load_deck(DECK)
    option_rows, card_ids, deltas = [], [], []

    for game in range(games):
        handle = BattleHandle()
        opponent = RandomOpponent(seed=500 + game)
        policy = RandomOpponent(seed=9000 + game)
        observation = handle.start(deck, deck)
        try:
            for _ in range(400):
                state = observation.current
                if state is None or state.result != -1 or observation.select is None:
                    break
                seat = state.yourIndex
                if seat != 0:
                    observation = handle.select(opponent(observation))
                    continue
                before = board_summary(state, seat)
                encoded = encoder.encode(observation, seat, 0)
                picks = policy(observation)
                if not picks:
                    observation = handle.select(picks)
                    continue
                slot = picks[0]
                options = encoded["options"]
                row = torch.cat(
                    [
                        options["cats"][slot].float(),
                        options["scalars"][slot].float(),
                        options["owner"][slot].reshape(-1).float(),
                        options["target_state"][slot].reshape(-1).float(),
                    ]
                ).numpy()
                cid = int(options["card_id"][slot].item())
                observation = handle.select(picks)
                after_state = observation.current
                if after_state is None or not after_state.players:
                    break
                # Only score transitions the opponent did not act in between,
                # otherwise the delta mixes our action with their reply.
                if after_state.yourIndex != seat:
                    continue
                after = board_summary(after_state, seat)
                option_rows.append(row)
                card_ids.append(cid)
                deltas.append(after - before)
        finally:
            handle.finish()

    features = np.asarray(option_rows, dtype=np.float64)
    targets = np.asarray(deltas, dtype=np.float64)
    ids = np.asarray(card_ids)
    unique = sorted(set(ids.tolist()))
    lookup = {c: i for i, c in enumerate(unique)}
    onehot = np.zeros((len(ids), len(unique)))
    onehot[np.arange(len(ids)), [lookup[c] for c in ids]] = 1.0

    print(f"transitions: {len(features)}, distinct cards chosen: {len(unique)}")
    print(f"option feature width: {features.shape[1]}")
    print()
    print(f"A. option features only        R2 = {ridge_r2(features, targets):.3f}")
    print(f"B. card id one-hot only        R2 = {ridge_r2(onehot, targets):.3f}")
    print(f"C. both                        R2 = {ridge_r2(np.hstack([features, onehot]), targets):.3f}")


if __name__ == "__main__":
    main()
