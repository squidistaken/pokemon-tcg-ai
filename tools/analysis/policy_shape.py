"""
Measure the deployed policy's decision distribution on real game states.

Replays the observations our agent actually faced on Kaggle through the 158M
checkpoint and reports how peaked the action distribution is, how much mass
sits on the synthetic stop action, and how often the greedy pick is the first
legal option. A policy that has collapsed onto a narrow habit shows up here.
"""
import glob
import json
import random
import sys

import torch
from omegaconf import OmegaConf
from torchrl.data import Binary, Categorical, Composite, Unbounded
from tensordict import TensorDict

from src.env.observation.structured_observation_encoder import (
    StructuredObservationEncoder,
)
from src.policies.ppo_actor import build_actor_critic
from submission.cg_api import to_observation_class

MAX_OPTIONS = 128
CHECKPOINT = (
    "outputs/deck-pinned-150m-local/tf-ptr-pinned-selfplay-10m-s42/"
    "checkpoints/snapshot_000158154752.pt"
)


def load_network():
    """
    Rebuild the trained actor-critic and its encoder.

    :return: The network in eval mode and the matching observation encoder.
    """
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
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
    return network.eval(), encoder


def collect_states(limit: int) -> list[tuple[dict, int]]:
    """
    Gather observation dicts our agent faced, with the acting seat.

    :param limit: Maximum states to return.
    :return: Pairs of raw observation dict and acting seat index.
    """
    states: list[tuple[dict, int]] = []
    rng = random.Random(0)
    files = sorted(glob.glob("logs/replays/*/episode-*.json"))
    rng.shuffle(files)
    for path in files:
        sub = path.split("/")[2]
        manifest = json.load(open(f"logs/replays/{sub}/manifest.json"))
        episode = path.split("episode-")[1].split("-replay")[0]
        meta = manifest.get(episode)
        if not meta:
            continue
        seat = meta["our_index"]
        data = json.load(open(path))
        for step in data["steps"]:
            agent = step[seat]
            obs = agent["observation"]
            if not obs.get("select") or not obs.get("current"):
                continue
            if not agent.get("action"):
                continue
            states.append((obs, seat))
            if len(states) >= limit:
                return states
    return states


@torch.inference_mode()
def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    network, encoder = load_network()
    states = collect_states(limit)
    top1, entropies, stop_mass, first_option, stop_chosen = [], [], [], 0, 0
    option_counts = []
    usable = 0
    for obs_dict, seat in states:
        try:
            observation = to_observation_class(obs_dict)
            encoded = TensorDict(
                {"observation": encoder.encode(observation, seat, 0)},
                batch_size=torch.Size(()),
            )
            logits = network.policy_logits(encoded)
        except Exception:
            continue
        n_options = min(len(observation.select.option), MAX_OPTIONS)
        if n_options < 2:
            continue
        usable += 1
        legal = torch.full_like(logits, float("-inf"))
        legal[:n_options] = logits[:n_options]
        legal[MAX_OPTIONS] = logits[MAX_OPTIONS]
        probs = torch.softmax(legal, dim=-1)
        live = probs[probs > 0]
        entropies.append(float(-(live * live.log()).sum()))
        top1.append(float(probs.max()))
        stop_mass.append(float(probs[MAX_OPTIONS]))
        best = int(torch.argmax(legal).item())
        if best == MAX_OPTIONS:
            stop_chosen += 1
        elif best == 0:
            first_option += 1
        option_counts.append(n_options)

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"states scored: {usable} (of {len(states)} collected)")
    print(f"mean legal options offered : {mean(option_counts):.2f}")
    print(f"mean policy entropy        : {mean(entropies):.3f} nats "
          f"(uniform over {mean(option_counts):.1f}+stop would be "
          f"{torch.tensor(mean(option_counts) + 1).log().item():.3f})")
    print(f"mean top-1 probability     : {mean(top1):.3f}")
    print(f"mean probability on stop   : {mean(stop_mass):.3f}")
    print(f"greedy pick == stop        : {stop_chosen}/{usable} ({stop_chosen/max(usable,1):.1%})")
    print(f"greedy pick == option 0    : {first_option}/{usable} ({first_option/max(usable,1):.1%})")


if __name__ == "__main__":
    main()
