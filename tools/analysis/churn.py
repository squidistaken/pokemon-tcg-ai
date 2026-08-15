"""
Track whether the policy settles down as training proceeds.

A policy converging on a strategy makes nearly the same decisions from one
snapshot to the next. This scores pairs of snapshots roughly 2M frames apart,
sampled across the whole run, on the same real game states, and reports how
often the pair agrees. Flat agreement across the run means the policy is still
churning at 170M frames as much as it was at 20M.
"""
import glob
import json
import random
import sys
from pathlib import Path

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
WEIGHTED = Path("outputs/weighted-field-20260808/tf-ptr-weighted-15m-s42/checkpoints")


def load_network(path: Path, encoder):
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
    states = []
    files = sorted(glob.glob("logs/replays/*/episode-*.json"))
    random.Random(2).shuffle(files)
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


def nearest(frames: int) -> Path:
    snaps = sorted(WEIGHTED.glob("snapshot_*.pt"))
    return min(snaps, key=lambda p: abs(int(p.stem.split("_")[-1]) - frames))


@torch.inference_mode()
def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    encoder = StructuredObservationEncoder(max_options=MAX_OPTIONS)
    states = collect_states(limit)
    encoded_states = []
    for obs_dict, seat in states:
        try:
            observation = to_observation_class(obs_dict)
            n_options = min(len(observation.select.option), MAX_OPTIONS)
            if n_options < 2:
                continue
            encoded_states.append(
                (
                    TensorDict(
                        {"observation": encoder.encode(observation, seat, 0)},
                        batch_size=torch.Size(()),
                    ),
                    n_options,
                )
            )
        except Exception:
            continue
    print(f"states scored: {len(encoded_states)}\n")

    def picks_for(path: Path):
        network = load_network(path, encoder)
        out = []
        for encoded, n_options in encoded_states:
            logits = network.policy_logits(encoded)
            legal = torch.full_like(logits, float("-inf"))
            legal[:n_options] = logits[:n_options]
            legal[MAX_OPTIONS] = logits[MAX_OPTIONS]
            out.append(int(torch.argmax(legal).item()))
        return out

    print(f"{'training point':>16}  {'gap':>6}  {'agreement':>10}")
    for frames in (20_000_000, 50_000_000, 80_000_000, 110_000_000, 140_000_000, 170_000_000):
        early, late = nearest(frames), nearest(frames + 2_000_000)
        early_frames = int(early.stem.split("_")[-1])
        late_frames = int(late.stem.split("_")[-1])
        if early_frames == late_frames:
            continue
        agree = sum(1 for a, b in zip(picks_for(early), picks_for(late)) if a == b)
        gap = (late_frames - early_frames) / 1e6
        print(f"{early_frames/1e6:>13.1f}M  {gap:>5.1f}M  {agree/len(encoded_states):>10.2f}")


if __name__ == "__main__":
    main()
