"""
Verify the behaviour-cloning inputs match what the agent sees when it plays.

The clone is trained on observations parsed out of Kaggle replay JSON and then
deployed against observations produced by the live engine. If those two differ
in shape, scale, or completeness, the clone is fitting one distribution and
being tested on another, and no amount of training fixes it. This encodes both
and compares them group by group.
"""
import collections
import glob
import json
import random

import torch

from cg.api import to_observation_class
from src.env.battle_handle import BattleHandle
from src.env.decks.deck import load_deck
from src.env.observation.structured_observation_encoder import (
    StructuredObservationEncoder,
)
from src.env.opponents.random_opponent import RandomOpponent

MAX_OPTIONS = 128


def collect_replay_observations(limit: int) -> list[tuple[dict, int]]:
    """
    Pull observations the experts actually acted on.

    :param limit: Maximum observations to return.
    :return: Pairs of raw observation dict and the acting seat.
    """
    out: list[tuple[dict, int]] = []
    files = sorted(glob.glob("logs/expert_replays/*/episode-*.json"))
    random.Random(0).shuffle(files)
    for path in files:
        team_dir = path.rsplit("/", 1)[0]
        episode = path.split("episode-")[1].split("-replay")[0]
        try:
            manifest = json.load(open(f"{team_dir}/manifest.json"))
        except Exception:
            continue
        meta = manifest.get(episode)
        if not meta:
            continue
        seat = meta["expert_index"]
        data = json.load(open(path))
        for step in data["steps"]:
            agent = step[seat]
            observation = agent.get("observation") or {}
            if observation.get("select") and observation.get("current") and agent.get("action"):
                out.append((observation, seat))
                if len(out) >= limit:
                    return out
    return out


def collect_live_observations(limit: int) -> list[tuple[object, int]]:
    """
    Pull observations straight from the engine under random play.

    :param limit: Maximum observations to return.
    :return: Pairs of Observation and the acting seat.
    """
    out = []
    deck = load_deck("decks/top20/dragapult/dragapult-10.csv")
    for game in range(200):
        handle = BattleHandle()
        policy = RandomOpponent(seed=game)
        observation = handle.start(deck, deck)
        try:
            for _ in range(400):
                state = observation.current
                if state is None or state.result != -1 or observation.select is None:
                    break
                out.append((observation, state.yourIndex))
                if len(out) >= limit:
                    return out
                observation = handle.select(policy(observation))
        finally:
            handle.finish()
    return out


def summarize(encoded: list) -> dict[str, tuple[float, float, float]]:
    """
    Per-group mean, standard deviation and share of exactly-zero entries.

    :param encoded: Encoded observations.
    :return: Group name to (mean, std, zero share).
    """
    keys = list(encoded[0].keys())
    summary = {}
    for key in keys:
        values = torch.stack([item[key].float().reshape(-1) for item in encoded])
        summary[key] = (
            float(values.mean()),
            float(values.std()),
            float((values == 0).float().mean()),
        )
    return summary


def flatten(entry, prefix=""):
    """
    Flatten a nested tensordict into name -> tensor.

    :param entry: TensorDict or tensor.
    :param prefix: Name prefix.
    :return: Flat dict of leaf tensors.
    """
    out = {}
    for key in entry.keys():
        value = entry.get(key)
        if hasattr(value, "keys"):
            out.update(flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def main() -> None:
    encoder = StructuredObservationEncoder(max_options=MAX_OPTIONS)
    print("collecting...", flush=True)
    replay_rows = collect_replay_observations(600)
    live_rows = collect_live_observations(600)
    print(f"  replay observations: {len(replay_rows)}")
    print(f"  live observations  : {len(live_rows)}\n")

    replay_encoded, failures = [], 0
    for observation, seat in replay_rows:
        try:
            replay_encoded.append(flatten(encoder.encode(to_observation_class(observation), seat, 0)))
        except Exception:
            failures += 1
    live_encoded = [flatten(encoder.encode(observation, seat, 0)) for observation, seat in live_rows]
    print(f"replay observations that failed to encode: {failures}\n")

    replay_summary = summarize(replay_encoded)
    live_summary = summarize(live_encoded)

    print(f"{'group':34} {'replay mean':>12} {'live mean':>12} {'replay std':>11} "
          f"{'live std':>10} {'zero% r/l':>14}")
    suspicious = []
    for key in sorted(replay_summary):
        r_mean, r_std, r_zero = replay_summary[key]
        l_mean, l_std, l_zero = live_summary.get(key, (float("nan"),) * 3)
        print(f"{key[:34]:34} {r_mean:>12.4f} {l_mean:>12.4f} {r_std:>11.4f} "
              f"{l_std:>10.4f} {r_zero:>6.2f}/{l_zero:<6.2f}")
        # A group that is populated live but empty in replays is missing data.
        if l_std > 1e-6 and r_std < 1e-6:
            suspicious.append(f"{key}: constant in replays, varies live")
        elif r_zero > 0.99 > l_zero:
            suspicious.append(f"{key}: all zero in replays, populated live")

    print()
    if suspicious:
        print("MISMATCHES:")
        for line in suspicious:
            print(f"  {line}")
    else:
        print("no group is populated live and empty in replays")


if __name__ == "__main__":
    main()
