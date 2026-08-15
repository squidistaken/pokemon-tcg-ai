"""
Turn harvested Kaggle replays into a supervised decision dataset.

Each replay step records the acting agent's full observation next to the
action it submitted, so the expert's decisions can be encoded with the very
same ``StructuredObservationEncoder`` training uses. Nothing is re-simulated.

A multi-select step is decomposed into the sequential single picks the
environment produces, so the cloned policy sees the same decision shape it
will face at play time. Every sample also carries the episode's final result
from the acting seat, which trains the value head on real outcomes.
"""
import argparse
import json
import random
from pathlib import Path

import torch
from cg.api import Observation
from cg.utils import to_dataclass
from tensordict import TensorDict

from src.env.observation.structured_observation_encoder import (
    StructuredObservationEncoder,
)

MAX_OPTIONS = 128
REPLAY_ROOT = Path("logs/expert_replays")


def iter_decisions(replay: dict, expert_index: int | None, both_seats: bool):
    """
    Yield every usable decision in one replay.

    :param replay: Parsed Kaggle replay JSON.
    :param expert_index: Seat the harvested team played, or None for any seat.
    :param both_seats: Keep the opponent's decisions too, tagged by result.
    :yield: ``(observation dict, chosen option indices, seat, outcome)``.
    """
    rewards = replay.get("rewards") or [0, 0]
    for step in replay.get("steps") or []:
        for agent in step:
            payload = agent.get("observation") or {}
            select = payload.get("select")
            state = payload.get("current")
            if not select or not state:
                continue
            options = select.get("option") or []
            action = agent.get("action")
            if not options or not isinstance(action, list) or not action:
                continue
            # The opening step submits a decklist through the same field, so
            # its "action" holds card ids far outside the option range.
            if any(
                not isinstance(index, int) or not 0 <= index < len(options)
                for index in action
            ):
                continue
            seat = state.get("yourIndex")
            if seat is None:
                continue
            if not both_seats and expert_index is not None and seat != expert_index:
                continue
            reward = rewards[seat] if seat < len(rewards) else 0
            outcome = 1.0 if (reward or 0) > 0 else (-1.0 if (reward or 0) < 0 else 0.0)
            yield payload, action, seat, outcome


def build(
    output: Path,
    max_replays: int,
    both_seats: bool,
    winners_only: bool,
    seed: int,
) -> None:
    """
    Encode the replay corpus into a single tensor file.

    :param output: Destination ``.pt`` path.
    :param max_replays: Cap on replays read, for quick smoke runs.
    :param both_seats: Include the non-harvested seat's decisions.
    :param winners_only: Drop decisions made by the side that lost.
    :param seed: Shuffle seed, so the split is reproducible.
    """
    encoder = StructuredObservationEncoder(max_options=MAX_OPTIONS)
    samples: list[dict] = []
    files: list[tuple[Path, int | None]] = []
    for team_dir in sorted(REPLAY_ROOT.iterdir()):
        manifest_path = team_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        for episode_id, meta in manifest.items():
            path = team_dir / f"episode-{episode_id}-replay.json"
            if path.is_file():
                files.append((path, meta.get("expert_index")))
    random.Random(seed).shuffle(files)
    files = files[:max_replays]
    print(f"reading {len(files)} replays", flush=True)

    skipped = 0
    for count, (path, expert_index) in enumerate(files, start=1):
        try:
            replay = json.loads(path.read_text())
        except Exception:
            skipped += 1
            continue
        for payload, action, seat, outcome in iter_decisions(
            replay, expert_index, both_seats
        ):
            if winners_only and outcome <= 0.0:
                continue
            try:
                observation = to_dataclass(payload, Observation)
            except Exception:
                continue
            # A pick of k options becomes k single-choice decisions, matching
            # how TCGEnv accumulates a multi-select at training time.
            for position, chosen in enumerate(action):
                try:
                    encoded = encoder.encode(observation, seat, position)
                except Exception:
                    break
                samples.append(
                    {
                        "observation": encoded,
                        "action": torch.tensor(chosen, dtype=torch.int64),
                        "outcome": torch.tensor(outcome, dtype=torch.float32),
                        "n_options": torch.tensor(
                            len(observation.select.option), dtype=torch.int64
                        ),
                    }
                )
        if count % 25 == 0:
            print(f"  {count}/{len(files)} replays -> {len(samples)} decisions",
                  flush=True)

    if not samples:
        raise SystemExit("no decisions extracted")
    dataset = TensorDict(
        {
            "observation": torch.stack([s["observation"] for s in samples]),
            "action": torch.stack([s["action"] for s in samples]),
            "outcome": torch.stack([s["outcome"] for s in samples]),
            "n_options": torch.stack([s["n_options"] for s in samples]),
        },
        batch_size=torch.Size((len(samples),)),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output)
    print(f"\nreplays read : {len(files) - skipped}")
    print(f"decisions    : {len(samples)}")
    print(f"written to   : {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("logs/bc_dataset.pt"))
    parser.add_argument("--max-replays", type=int, default=10_000)
    parser.add_argument("--both-seats", action="store_true")
    parser.add_argument("--winners-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build(args.output, args.max_replays, args.both_seats, args.winners_only, args.seed)


if __name__ == "__main__":
    main()
