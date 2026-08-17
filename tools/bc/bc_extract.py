"""
Turn Kaggle's daily episode export into a supervised decision dataset.

Each replay step records the acting agent's full observation next to the
action it submitted, so the expert's decisions can be encoded with the very
same ``StructuredObservationEncoder`` training uses. Nothing is re-simulated.

A multi-select step is decomposed into the sequential single picks the
environment produces, so the cloned policy sees the same decision shape it
will face at play time. Every sample also carries the episode's final result
from the acting seat, which trains the value head on real outcomes.
"""

import argparse
import glob
import json
import logging
import random
from pathlib import Path

import torch
from tensordict import TensorDict

from cg.api import Observation
from cg.utils import to_dataclass
from src.env.observation.structured_observation_encoder import (
    StructuredObservationEncoder,
)

logger = logging.getLogger(__name__)

MAX_OPTIONS = 128


STOP_INDEX = MAX_OPTIONS


def legal_mask(n_options: int, chosen: list[int], min_count: int) -> torch.Tensor:
    """
    Reproduce ``TCGEnv._build_mask`` for one accumulation position.

    The clone is deployed through ``GreedyPolicyOpponent``, which compares the
    stop logit against the best option logit to decide when a selection is
    finished. Training therefore has to see exactly the mask the environment
    builds, stop slot included, or that comparison is against a logit no
    gradient ever reached.

    :param n_options: Options the selection offers.
    :param chosen: Option indices already picked in this accumulation.
    :param min_count: Minimum picks the selection demands.
    :return: Bool tensor of shape ``(MAX_OPTIONS + 1,)``.
    """
    mask = torch.zeros(MAX_OPTIONS + 1, dtype=torch.bool)
    mask[: min(n_options, MAX_OPTIONS)] = True
    for index in chosen:
        mask[index] = False
    if len(chosen) >= min_count:
        mask[STOP_INDEX] = True
    return mask


def iter_decisions(replay: dict):
    """
    Yield every usable decision in one replay, one per accumulation position.

    A selection of k options becomes k pick rows plus, when the expert could
    have taken more and chose not to, a terminating stop row. Declining an
    optional selection outright is that stop row at position zero, which is a
    real strategic choice and not an absence of one.

    :param replay: Parsed Kaggle replay JSON.
    :yield: ``(observation dict, position, target, mask, seat, outcome)``.
    """
    rewards = replay.get("rewards") or [0, 0]
    steps = replay.get("steps") or []
    for step_index, step in enumerate(steps):
        for agent_index, agent in enumerate(step):
            # A replay carries an observation and an action field for both
            # seats at every step, but only the ACTIVE one is being asked to
            # move here. The idle seat's fields are stale.
            if agent.get("status") != "ACTIVE":
                continue
            payload = agent.get("observation") or {}
            select = payload.get("select")
            state = payload.get("current")
            if not select or not state:
                continue
            options = select.get("option") or []
            # Kaggle records an agent's answer on the *following* step: step 0
            # poses the deck prompt and the decklist appears as step 1's
            # action. Reading this step's action pairs every state with the
            # previous decision's answer.
            if step_index + 1 >= len(steps):
                continue
            following = steps[step_index + 1]
            if agent_index >= len(following):
                continue
            action = following[agent_index].get("action")
            if not options or not isinstance(action, list):
                continue
            # The opening step submits a decklist through the same field, so
            # its "action" holds card ids far outside the option range.
            if any(
                not isinstance(index, int) or not 0 <= index < len(options)
                for index in action
            ):
                continue
            if len(set(action)) != len(action) or len(options) > MAX_OPTIONS:
                continue
            seat = state.get("yourIndex")
            if seat is None:
                continue
            reward = rewards[seat] if seat < len(rewards) else 0
            outcome = 1.0 if (reward or 0) > 0 else (-1.0 if (reward or 0) < 0 else 0.0)

            min_count = int(select.get("minCount") or 0)
            max_count = min(int(select.get("maxCount") or 1), len(options))
            for position, chosen in enumerate(action):
                yield (
                    payload,
                    position,
                    chosen,
                    legal_mask(len(options), action[:position], min_count),
                    seat,
                    outcome,
                )
            # greedy_select stops querying once max_count picks are in hand, so
            # only a short selection records a deliberate stop.
            if min_count <= len(action) < max_count:
                yield (
                    payload,
                    len(action),
                    STOP_INDEX,
                    legal_mask(len(options), list(action), min_count),
                    seat,
                    outcome,
                )


def write_shard(samples: list[dict], destination: Path) -> int:
    """
    Stack one batch of rows and memory-map it to disk.

    Rows are ~28 KB each, so a full corpus does not fit in RAM. Writing in
    shards keeps peak memory at one shard and lets training page rows from
    disk instead of holding the corpus resident.

    :param samples: Rows to write.
    :param destination: Directory to memory-map into.
    :return: Number of rows written.
    """
    if not samples:
        return 0
    shard = TensorDict(
        {
            "observation": torch.stack([s["observation"] for s in samples]),
            "action": torch.stack([s["action"] for s in samples]),
            "action_mask": torch.stack([s["action_mask"] for s in samples]),
            "outcome": torch.stack([s["outcome"] for s in samples]),
            "episode": torch.stack([s["episode"] for s in samples]),
        },
        batch_size=torch.Size((len(samples),)),
    )
    destination.mkdir(parents=True, exist_ok=True)
    shard.memmap_(str(destination))
    return len(samples)


def load_team_whitelist(path: Path | None) -> set[str] | None:
    """
    Read the teams whose decisions are kept.

    One team name per line, matching ``TeamName`` in the Kaggle leaderboard
    export exactly. Blank lines and ``#`` comments are ignored. Names carry
    commas and non-ASCII characters, so this is a file rather than a delimited
    command-line value.

    :param path: File to read, or None for no filtering.
    :return: The set of team names, or None when unfiltered.
    """
    if path is None:
        return None
    names = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not names:
        raise ValueError(f"team whitelist {path} is empty")
    return names


def build(
    output: Path,
    max_replays: int,
    winners_only: bool,
    seed: int,
    replay_glob: str,
    shard_size: int = 1200,
    team_whitelist: set[str] | None = None,
) -> None:
    """
    Encode the replay corpus into a single tensor file.

    :param output: Destination ``.pt`` path.
    :param max_replays: Cap on replays read, for quick smoke runs.
    :param winners_only: Drop decisions made by the side that lost.
    :param seed: Shuffle seed, so the split is reproducible.
    :param replay_glob: Glob for the flat daily-export directory of
        ``<episode>.json`` files.
    :param shard_size: Replays per memory-mapped shard.
    :param team_whitelist: Keep only decisions made by a seat belonging to one
        of these teams. None clones both seats of every game, which fits the
        whole ladder field rather than the players worth imitating.
    """
    encoder = StructuredObservationEncoder(max_options=MAX_OPTIONS)
    samples: list[dict] = []
    # The daily export is a flat directory of <episode>.json with no manifest.
    # Without a whitelist both seats are cloned, so the corpus is the field
    # average; `info.TeamNames` is indexed by the same seat number the decision
    # carries, which is what makes a per-seat rating filter possible. A
    # duplicate path is dropped so a repeated glob never double counts.
    by_episode: dict[str, Path] = {}
    duplicates = 0
    for path_str in glob.glob(replay_glob):
        path = Path(path_str)
        episode_id = path.stem
        if episode_id in by_episode:
            duplicates += 1
            continue
        by_episode[episode_id] = path
    files = list(by_episode.items())
    random.Random(seed).shuffle(files)
    files = files[:max_replays]
    print(
        f"reading {len(files)} distinct replays "
        f"({duplicates} duplicate copies dropped)",
        flush=True,
    )

    skipped = 0
    stop_rows = 0
    shards = 0
    total_rows = 0
    kept_out = 0
    for count, (episode_id, path) in enumerate(files, start=1):
        try:
            replay = json.loads(path.read_text())
        except Exception:
            logger.debug("could not parse %s; skipping", path, exc_info=True)
            skipped += 1
            continue
        game = int(episode_id) if str(episode_id).isdigit() else count
        team_names = (replay.get("info") or {}).get("TeamNames") or []
        for payload, position, target, mask, seat, outcome in iter_decisions(replay):
            if winners_only and outcome <= 0.0:
                continue
            if team_whitelist is not None and (
                seat >= len(team_names) or team_names[seat] not in team_whitelist
            ):
                kept_out += 1
                continue
            try:
                observation = to_dataclass(payload, Observation)
                encoded = encoder.encode(observation, seat, position)
            except Exception:
                logger.debug(
                    "could not encode a decision from %s; skipping", path, exc_info=True
                )
                continue
            stop_rows += target == STOP_INDEX
            samples.append(
                {
                    "observation": encoded,
                    "action": torch.tensor(target, dtype=torch.int64),
                    "action_mask": mask,
                    "outcome": torch.tensor(outcome, dtype=torch.float32),
                    "episode": torch.tensor(game, dtype=torch.int64),
                }
            )
        if count % shard_size == 0:
            written = write_shard(samples, output / f"shard_{shards:04d}")
            if written:
                total_rows += written
                shards += 1
            samples = []
        if count % 200 == 0:
            print(
                f"  {count}/{len(files)} replays -> "
                f"{total_rows + len(samples)} decisions",
                flush=True,
            )

    written = write_shard(samples, output / f"shard_{shards:04d}")
    if written:
        total_rows += written
        shards += 1
    if not total_rows:
        raise SystemExit("no decisions extracted")
    (output / "meta.json").write_text(
        json.dumps({"shards": shards, "rows": total_rows}, indent=1)
    )
    print(f"\nreplays read : {len(files) - skipped}")
    print(f"decisions    : {total_rows}")
    if team_whitelist is not None:
        print(
            f"off-team rows: {kept_out} dropped "
            f"({kept_out / max(kept_out + total_rows, 1):.1%} of the corpus)"
        )
    print(f"stop rows    : {stop_rows} ({stop_rows / max(total_rows, 1):.2%})")
    print(f"shards       : {shards}")
    print(f"written to   : {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("logs/bc_dataset.pt"))
    parser.add_argument("--max-replays", type=int, default=10_000)
    parser.add_argument("--winners-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--replay-glob",
        required=True,
        help="Glob for the flat daily-export directory of <episode>.json files, "
        "e.g. 'logs/kaggle_episodes/*/*.json'.",
    )
    parser.add_argument("--shard-size", type=int, default=1200)
    parser.add_argument(
        "--team-whitelist",
        type=Path,
        default=None,
        help="File of team names, one per line, matching the leaderboard "
        "export. Only decisions made by a listed team's seat are kept.",
    )
    args = parser.parse_args()
    build(
        args.output,
        args.max_replays,
        args.winners_only,
        args.seed,
        args.replay_glob,
        args.shard_size,
        load_team_whitelist(args.team_whitelist),
    )


if __name__ == "__main__":
    main()
