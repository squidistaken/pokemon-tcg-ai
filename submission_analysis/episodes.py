"""Summarize outcomes of episodes our own Kaggle submissions have played."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from rich.box import ROUNDED
from rich.console import Console
from rich.table import Table

from submission_analysis.submissions import (
    COMPETITION,
    FetchError,
    fetch_submissions,
    repo_root,
    resolve_local_path,
    select_rows,
    to_rows,
)

REPLAYS_DIR = "logs/replays"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class EpisodeOutcome:
    """One episode a target submission played, from its own point of view."""

    episode_id: int
    create_time: datetime
    result: str
    our_reward: float
    opponent_team: str
    opponent_reward: float
    our_index: int


@dataclass(frozen=True)
class OutcomeSummary:
    """Aggregate win/loss/draw record for one submission, overall and by opponent."""

    submission_ref: int
    total: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    by_opponent: dict[str, dict[str, int]]


def fetch_episodes(submission_ref: int) -> list[Any]:
    """
    Fetch every episode a submission has played, through the Kaggle API.

    :param submission_ref: Kaggle submission ID.
    :return: Raw episode objects from the Kaggle API.
    :raises FetchError: If Kaggle API authentication fails.
    """
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as error:
        raise FetchError(
            f"Could not authenticate with the Kaggle API: {error}\n"
            "Put KAGGLE_API_TOKEN in the untracked .env, run "
            "'uv run kaggle auth login', or configure another credential "
            "method supported by the CLI."
        ) from error
    return api.competition_list_episodes(submission_ref) or []


def _result_for_reward(reward: float) -> str:
    """
    Classify a submission's episode reward as win/loss/draw.

    :param reward: This submission's reward for the episode.
    :return: ``"win"``, ``"loss"``, or ``"draw"``.
    """
    if reward > 0:
        return "win"
    if reward < 0:
        return "loss"
    return "draw"


def to_outcomes(submission_ref: int, episodes: Iterable[Any]) -> list[EpisodeOutcome]:
    """
    Reduce raw episodes to this submission's own outcomes, skipping any
    episode missing either side (still pending, errored, or malformed).

    :param submission_ref: Kaggle submission ID these episodes belong to.
    :param episodes: Raw episode objects from :func:`fetch_episodes`.
    :return: One outcome per playable episode.
    """
    outcomes = []
    for episode in episodes:
        mine = next(
            (
                agent
                for agent in episode.agents
                if agent.submission_id == submission_ref
            ),
            None,
        )
        theirs = next(
            (
                agent
                for agent in episode.agents
                if agent.submission_id != submission_ref
            ),
            None,
        )
        if mine is None or theirs is None:
            continue
        outcomes.append(
            EpisodeOutcome(
                episode_id=episode.id,
                create_time=episode.create_time,
                result=_result_for_reward(mine.reward),
                our_reward=mine.reward,
                opponent_team=theirs.team_name,
                opponent_reward=theirs.reward,
                our_index=mine.index,
            )
        )
    return outcomes


_RESULT_BUCKETS = {"win": "wins", "loss": "losses", "draw": "draws"}


def summarize(
    submission_ref: int, outcomes: Sequence[EpisodeOutcome]
) -> OutcomeSummary:
    """
    Aggregate outcomes into overall and per-opponent win/loss/draw counts.

    :param submission_ref: Kaggle submission ID these outcomes belong to.
    :param outcomes: Per-episode outcomes from :func:`to_outcomes`.
    :return: The aggregated summary.
    """
    wins = sum(1 for outcome in outcomes if outcome.result == "win")
    losses = sum(1 for outcome in outcomes if outcome.result == "loss")
    draws = sum(1 for outcome in outcomes if outcome.result == "draw")
    total = len(outcomes)
    by_opponent: dict[str, dict[str, int]] = {}
    for outcome in outcomes:
        bucket = by_opponent.setdefault(
            outcome.opponent_team, {"wins": 0, "losses": 0, "draws": 0}
        )
        bucket[_RESULT_BUCKETS[outcome.result]] += 1
    return OutcomeSummary(
        submission_ref=submission_ref,
        total=total,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=wins / total if total else 0.0,
        by_opponent=by_opponent,
    )


def resolve_submission_refs(competition: str, *, most_recent_n: int) -> list[int]:
    """
    Resolve the N most recently submitted successful submissions' refs.

    :param competition: Kaggle competition slug.
    :param most_recent_n: Number of most-recent successful submissions to resolve.
    :return: Matching submission refs, newest first.
    """
    submissions = fetch_submissions(competition)
    rows = select_rows(to_rows(submissions), most_recent_n=most_recent_n)
    return [row.ref for row in rows]


def download_replays(
    submission_ref: int, outcomes: Sequence[EpisodeOutcome], directory: Path
) -> None:
    """
    Download the full replay JSON for each outcome's episode.

    :param submission_ref: Kaggle submission ID these outcomes belong to.
    :param outcomes: Episodes to download replays for.
    :param directory: Root replay directory (one subdirectory per submission ref).
    :return: None.
    """
    api = KaggleApi()
    api.authenticate()
    submission_dir = directory / str(submission_ref)
    submission_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, Any]] = {}
    manifest_path = submission_dir / MANIFEST_FILENAME
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    downloaded = 0
    for outcome in outcomes:
        replay_path = submission_dir / f"episode-{outcome.episode_id}-replay.json"
        if not replay_path.is_file():
            api.competition_episode_replay(
                outcome.episode_id, path=str(submission_dir), quiet=True
            )
            downloaded += 1
        manifest[str(outcome.episode_id)] = {
            "our_index": outcome.our_index,
            "result": outcome.result,
            "our_reward": outcome.our_reward,
            "opponent_team": outcome.opponent_team,
            "opponent_reward": outcome.opponent_reward,
            "create_time": outcome.create_time.isoformat(),
        }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"Downloaded {downloaded} new replay(s) ({len(outcomes) - downloaded} already cached) "
        f"for submission {submission_ref} to {submission_dir}"
    )


def render_table(summary: OutcomeSummary) -> None:
    """
    Print one submission's outcome summary as Rich tables.

    :param summary: Aggregated outcome summary to print.
    :return: None.
    """
    console = Console()
    overall = Table(
        box=ROUNDED,
        title=f"Submission {summary.submission_ref}: outcomes",
        title_justify="left",
    )
    overall.add_column("episodes")
    overall.add_column("wins", justify="right")
    overall.add_column("losses", justify="right")
    overall.add_column("draws", justify="right")
    overall.add_column("win rate", justify="right")
    overall.add_row(
        str(summary.total),
        str(summary.wins),
        str(summary.losses),
        str(summary.draws),
        f"{summary.win_rate:.1%}",
    )
    console.print(overall)

    if summary.by_opponent:
        by_opponent = Table(box=ROUNDED, title="By opponent", title_justify="left")
        by_opponent.add_column("opponent team")
        by_opponent.add_column("wins", justify="right")
        by_opponent.add_column("losses", justify="right")
        by_opponent.add_column("draws", justify="right")
        for opponent, record in sorted(summary.by_opponent.items()):
            by_opponent.add_row(
                opponent,
                str(record["wins"]),
                str(record["losses"]),
                str(record["draws"]),
            )
        console.print(by_opponent)


def _parser() -> argparse.ArgumentParser:
    """
    :return: The argument parser for the ``episodes`` subcommand.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Summarize win/loss/draw outcomes for episodes our own Kaggle "
            "submissions have played, by opponent team."
        )
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--submission-ref",
        type=int,
        action="append",
        metavar="REF",
        help="Analyze this Kaggle submission ref. Repeatable.",
    )
    target.add_argument(
        "--most-recent-n",
        type=int,
        default=1,
        help="Analyze the N most recently submitted successful submissions (default: 1).",
    )
    parser.add_argument(
        "--competition", help=f"Kaggle competition (default: {COMPETITION})."
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--download-replays",
        action="store_true",
        help=(
            "Also download each analyzed episode's full replay JSON "
            f"(default: {REPLAYS_DIR}/<submission ref>/)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI entry point.

    :param argv: Argument vector; defaults to ``sys.argv`` when None.
    :return: Process exit code.
    """
    args = _parser().parse_args(argv)
    root = repo_root()
    env_path = root / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)

    competition = args.competition or os.environ.get("KAGGLE_COMPETITION", COMPETITION)
    try:
        refs = args.submission_ref or resolve_submission_refs(
            competition, most_recent_n=args.most_recent_n
        )
        if not refs:
            print("No successful submissions found.")
            return 0
        summaries = []
        for ref in refs:
            episodes = fetch_episodes(ref)
            outcomes = to_outcomes(ref, episodes)
            summaries.append(summarize(ref, outcomes))
            if args.download_replays:
                download_replays(ref, outcomes, resolve_local_path(REPLAYS_DIR, root))
    except FetchError as error:
        print(f"error: {error}")
        return 2

    if args.format == "json":
        print(json.dumps([asdict(summary) for summary in summaries], indent=2))
    else:
        for summary in summaries:
            render_table(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
