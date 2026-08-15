"""
Turn expert replays into a behaviour-cloning index.

One replay decision is several training examples, because ``TCGEnv`` decomposes
a selection into one pick at a time (``src/env/tcg_env.py:266-276``): each pick
is an action, options already taken are masked out, and an explicit **stop**
follows when the expert took fewer than ``maxCount``. This walks the replays and
records one row per sub-decision, keeping only pointers into the JSON so the
index stays small and observations are encoded lazily at training time.

With ``--both-seats`` it learns from the opponent's seat too. Kaggle matches by
rating, so a top team's opponent is also near the top, and reading only the
scouted seat discards 45% of the expert decisions already on disk.
"""
import argparse
import collections
import json
import pickle
from pathlib import Path

REPLAY_ROOT = Path("logs/expert_replays")
OUTPUT = Path("logs/bc_index.pkl")
MAX_OPTIONS = 128
STOP_INDEX = MAX_OPTIONS


def rows_for_selection(
    action: list[int], min_count: int, max_count: int, n_options: int
) -> list[tuple[tuple[int, ...], int]]:
    """
    Expand one expert selection into the env's sequence of single picks.

    :param action: The option indices the expert submitted.
    :param min_count: Selection's ``minCount``.
    :param max_count: Selection's ``maxCount``.
    :param n_options: Number of options offered.
    :return: ``(already_chosen_prefix, label)`` per sub-decision.
    """
    picks = [a for a in action if 0 <= a < n_options]
    rows: list[tuple[tuple[int, ...], int]] = []
    for index, option in enumerate(picks):
        rows.append((tuple(picks[:index]), option))
    # The env auto-submits once maxCount picks are in; short of that the expert
    # is choosing to stop, and that choice is itself a training signal.
    if len(picks) < max_count and len(picks) >= min_count:
        rows.append((tuple(picks), STOP_INDEX))
    return rows


def extract_seat(
    data: dict,
    seat: int,
    team: str | None,
    rank: int | None,
    outcome: float,
    replay_path: Path,
    index: list[dict],
    stats: collections.Counter,
    per_team: collections.Counter,
) -> None:
    """
    Append every representable sub-decision one seat made in one episode.

    :param data: Parsed replay.
    :param seat: Seat to extract.
    :param team: Team name playing that seat.
    :param rank: That team's leaderboard rank, when known.
    :param outcome: Episode result from this seat's view.
    :param replay_path: Replay path, used as the episode key for the split.
    :param index: Output list, appended in place.
    :param stats: Counters, updated in place.
    :param per_team: Row counts per team, updated in place.
    """
    for step_index, step in enumerate(data["steps"]):
        agent = step[seat]
        observation = agent.get("observation") or {}
        select = observation.get("select")
        action = agent.get("action")
        if not select or action is None or not observation.get("current"):
            continue
        # Both seats carry a select and an action at every step, and a seat
        # that was never asked shows an empty action, identical to a genuine
        # decline. Counting those invents a stop from a position the expert
        # never faced (measured: 800 fabricated against 72 real declines), so
        # a step only counts when this seat actually submitted something.
        if not action:
            stats["skipped_not_acting"] += 1
            continue
        n_options = len(select.get("option") or [])
        if n_options == 0 or n_options > MAX_OPTIONS:
            stats["skipped_option_count"] += 1
            continue
        if not isinstance(action, list) or any(not isinstance(a, int) for a in action):
            stats["skipped_action_shape"] += 1
            continue
        expanded = rows_for_selection(
            action,
            int(select.get("minCount", 1)),
            int(select.get("maxCount", 1)),
            n_options,
        )
        for chosen, label in expanded:
            index.append(
                {
                    "replay": str(replay_path),
                    "step": step_index,
                    "seat": seat,
                    "chosen": chosen,
                    "label": label,
                    "n_options": n_options,
                    "min_count": int(select.get("minCount", 1)),
                    "outcome": outcome,
                    "team": team,
                    "rank": rank,
                }
            )
            stats["stop_labels" if label == STOP_INDEX else "pick_labels"] += 1
        per_team[team] += len(expanded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winners-only", action="store_true",
                        help="Keep only seats that won their episode.")
    parser.add_argument("--both-seats", action="store_true",
                        help="Also learn from the opponent's seat in each replay.")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    index: list[dict] = []
    stats: collections.Counter = collections.Counter()
    per_team: collections.Counter = collections.Counter()
    # An episode played between two scouted teams is downloaded into both of
    # their directories. Counting it once per directory would weight games
    # between strong teams several times over, so episodes are deduplicated by
    # id rather than by path.
    seen_episodes: set[str] = set()

    for manifest_path in sorted(REPLAY_ROOT.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        team_dir = manifest_path.parent
        for episode_id, meta in manifest.items():
            replay_path = team_dir / f"episode-{episode_id}-replay.json"
            if not replay_path.is_file():
                continue
            if episode_id in seen_episodes:
                stats["duplicate_episodes"] += 1
                continue
            seen_episodes.add(episode_id)
            try:
                data = json.loads(replay_path.read_text())
            except Exception:
                stats["unreadable"] += 1
                continue
            stats["episodes"] += 1
            names = (data.get("info") or {}).get("TeamNames") or []
            rewards = data.get("rewards") or []
            scouted = meta["expert_index"]
            seats = [scouted] + ([1 - scouted] if args.both_seats else [])
            for seat in seats:
                outcome = (
                    float(rewards[seat])
                    if len(rewards) > seat and rewards[seat] is not None
                    else 0.0
                )
                if args.winners_only and outcome <= 0:
                    stats["seats_skipped_lost"] += 1
                    continue
                team = (
                    meta.get("team_name")
                    if seat == scouted
                    else (names[seat] if len(names) > seat else None)
                )
                extract_seat(
                    data, seat, team, meta.get("rank"), outcome,
                    replay_path, index, stats, per_team,
                )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump(index, handle)

    print(f"episodes read        : {stats['episodes']:,}")
    print(f"  duplicate episodes skipped: {stats['duplicate_episodes']:,}")
    print(f"seats extracted      : {'both' if args.both_seats else 'scouted only'}")
    if args.winners_only:
        print(f"  seats dropped (lost): {stats['seats_skipped_lost']:,}")
    print(f"training rows        : {len(index):,}")
    print(f"  option picks       : {stats['pick_labels']:,}")
    print(f"  stop actions       : {stats['stop_labels']:,}")
    print(f"  skipped (not acting): {stats['skipped_not_acting']:,}")
    print(f"  skipped (options)  : {stats['skipped_option_count']:,}")
    print(f"  skipped (action)   : {stats['skipped_action_shape']:,}")
    wins = sum(1 for row in index if row["outcome"] > 0)
    print(f"rows from won games  : {wins:,} ({wins / max(len(index), 1):.1%})")
    print(f"distinct teams       : {len(per_team)}")
    print(f"\ntop teams by rows:")
    for team, count in per_team.most_common(8):
        print(f"  {str(team)[:32]:34} {count:>9,}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
