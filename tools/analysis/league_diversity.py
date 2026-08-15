"""
Measure how much behavioural diversity the self-play league actually contains.

Every league member is a GreedyPolicyOpponent, an argmax over a frozen
checkpoint, so a member's behaviour is a deterministic function of the state.
The pool holds 24 snapshots spanning 24M frames. This scores several snapshots
across that span on the same real game states and reports how often they pick
the same action, which is the effective number of distinct opponents the
learner faces.
"""
import glob
import itertools
import json
import random
import sys

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.data import Binary, Categorical, Composite, Unbounded

from src.env.observation.structured_observation_encoder import (
    StructuredObservationEncoder,
)
from src.policies.ppo_actor import build_actor_critic
from submission.cg_api import to_observation_class

MAX_OPTIONS = 128
WEIGHTED = "outputs/weighted-field-20260808/tf-ptr-weighted-15m-s42/checkpoints"
PINNED = "outputs/deck-pinned-150m-local/tf-ptr-pinned-selfplay-10m-s42/checkpoints"


def load_network(path: str, encoder: StructuredObservationEncoder):
    """
    Rebuild one checkpoint's network.

    :param path: Snapshot to load.
    :param encoder: Shared observation encoder.
    :return: The network in eval mode.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = OmegaConf.create(checkpoint["config"])
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
    return network.eval()


def collect_states(limit: int):
    """
    Gather real observation dicts with their acting seat.

    :param limit: Maximum states to return.
    :return: List of (observation dict, seat) pairs.
    """
    states = []
    files = sorted(glob.glob("logs/replays/*/episode-*.json"))
    random.Random(1).shuffle(files)
    for path in files:
        sub = path.split("/")[2]
        episode = path.split("episode-")[1].split("-replay")[0]
        meta = json.load(open(f"logs/replays/{sub}/manifest.json")).get(episode)
        if not meta:
            continue
        for step in json.load(open(path))["steps"]:
            agent = step[meta["our_index"]]
            obs = agent["observation"]
            if obs.get("select") and obs.get("current") and agent.get("action"):
                states.append((obs, meta["our_index"]))
                if len(states) >= limit:
                    return states
    return states


@torch.inference_mode()
def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    encoder = StructuredObservationEncoder(max_options=MAX_OPTIONS)
    members = {
        "126M": f"{WEIGHTED}/snapshot_000126009344.pt",
        "138M": f"{WEIGHTED}/snapshot_000138010624.pt",
        "150M": f"{WEIGHTED}/snapshot_000150011904.pt",
        "162M": f"{PINNED}/snapshot_000162004992.pt",
        "173M": f"{PINNED}/snapshot_000173670400.pt",
    }
    resolved = {}
    for name, path in members.items():
        from pathlib import Path
        target = Path(path)
        if not target.exists():
            siblings = sorted(target.parent.glob("snapshot_*.pt"))
            if not siblings:
                continue
            wanted = int(target.stem.split("_")[-1])
            target = min(siblings, key=lambda p: abs(int(p.stem.split("_")[-1]) - wanted))
        resolved[name] = str(target)
        print(f"{name}: {target.name}")

    networks = {name: load_network(path, encoder) for name, path in resolved.items()}
    states = collect_states(limit)
    picks: dict[str, list[int]] = {name: [] for name in networks}
    scored = 0
    for obs_dict, seat in states:
        try:
            observation = to_observation_class(obs_dict)
            encoded = TensorDict(
                {"observation": encoder.encode(observation, seat, 0)},
                batch_size=torch.Size(()),
            )
        except Exception:
            continue
        n_options = min(len(observation.select.option), MAX_OPTIONS)
        if n_options < 2:
            continue
        scored += 1
        for name, network in networks.items():
            logits = network.policy_logits(encoded)
            legal = torch.full_like(logits, float("-inf"))
            legal[:n_options] = logits[:n_options]
            legal[MAX_OPTIONS] = logits[MAX_OPTIONS]
            picks[name].append(int(torch.argmax(legal).item()))

    print(f"\nstates scored: {scored}")
    names = list(networks)
    print("\npairwise action agreement between league members:")
    print(f"{'':8}" + "".join(f"{n:>8}" for n in names))
    for a in names:
        row = ""
        for b in names:
            same = sum(1 for x, y in zip(picks[a], picks[b]) if x == y)
            row += f"{same / max(scored, 1):>8.2f}"
        print(f"{a:8}{row}")

    spans = [(a, b) for a, b in itertools.combinations(names, 2)]
    agreements = [
        sum(1 for x, y in zip(picks[a], picks[b]) if x == y) / max(scored, 1)
        for a, b in spans
    ]
    print(f"\nmean off-diagonal agreement: {sum(agreements)/len(agreements):.3f}")
    unique_profiles = len({tuple(picks[n]) for n in names})
    print(f"distinct action profiles among {len(names)} members: {unique_profiles}")


if __name__ == "__main__":
    main()
