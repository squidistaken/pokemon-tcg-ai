"""Fetch live Kaggle submission status for this competition."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import (
    ApiGetLeaderboardRequest,
)
from kagglesdk.competitions.types.competition_enums import (
    SubmissionGroup,
    SubmissionSortBy,
)
from rich.box import ROUNDED
from rich.console import Console
from rich.table import Table

from src.checkpoint_registry import append_rating_history

COMPETITION = "pokemon-tcg-ai-battle"
RATING_HISTORY_FILE = "logs/kaggle_rating_history.csv"


class FetchError(RuntimeError):
    """User-facing failure while fetching or rendering submission status."""


@dataclass(frozen=True)
class SubmissionStatusRow:
    """One Kaggle submission, normalized for filtering, sorting, and display."""

    ref: int
    label: str
    file_name: str
    date: datetime
    status: str
    successful: bool
    error_description: str
    public_score: float | None


def repo_root() -> Path:
    """
    :return: The repository root containing this package.
    """
    return Path(__file__).resolve().parents[1]


def resolve_local_path(value: str | Path, root: Path) -> Path:
    """
    Resolve a user/config path relative to the repository root.

    :param value: A path, absolute or relative to ``root``.
    :param root: Repository root to resolve relative paths against.
    :return: The resolved absolute path.
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def fetch_submissions(competition: str, *, page_size: int = 100) -> list[Any]:
    """
    Fetch every submission for ``competition`` through the Kaggle API.

    :param competition: Kaggle competition slug.
    :param page_size: Submissions requested per API page.
    :return: Every submission, across all pages, newest-first.
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

    submissions: list[Any] = []
    page_number = 1
    while True:
        page = (
            api.competition_submissions(
                competition,
                group=SubmissionGroup.SUBMISSION_GROUP_ALL,
                sort=SubmissionSortBy.SUBMISSION_SORT_BY_DATE,
                page_size=page_size,
                page_number=page_number,
            )
            or []
        )
        submissions.extend(page)
        if len(page) < page_size:
            break
        page_number += 1
    return submissions


def _to_row(submission: Any) -> SubmissionStatusRow:
    """
    Normalize one Kaggle API submission into a :class:`SubmissionStatusRow`.

    :param submission: Raw submission object from the Kaggle API.
    :return: The normalized row.
    """
    status = submission.status.name
    public_score: float | None = None
    if submission.public_score:
        try:
            public_score = float(submission.public_score)
        except ValueError:
            public_score = None
    return SubmissionStatusRow(
        ref=submission.ref,
        label=submission.description,
        file_name=submission.file_name,
        date=submission.date,
        status=status,
        successful=status == "COMPLETE" and not submission.error_description,
        error_description=submission.error_description,
        public_score=public_score,
    )


def to_rows(submissions: Iterable[Any]) -> list[SubmissionStatusRow]:
    """
    Normalize raw Kaggle API submissions into display rows.

    :param submissions: Raw submission objects from the Kaggle API.
    :return: Normalized rows.
    """
    return [_to_row(submission) for submission in submissions]


def select_rows(
    rows: Sequence[SubmissionStatusRow],
    *,
    all_statuses: bool = False,
    best_n: int | None = None,
    most_recent_n: int | None = None,
) -> list[SubmissionStatusRow]:
    """
    Filter to successful submissions (unless overridden), then sort/limit.

    :param rows: Rows to filter/sort/limit.
    :param all_statuses: Include PENDING/ERROR submissions, not just
        successful ones.
    :param best_n: If set, keep only the N highest-scoring rows.
    :param most_recent_n: If set, keep only the N most recently submitted
        rows. Ignored when ``best_n`` is set.
    :return: The filtered, sorted, limited rows.
    """
    pool = list(rows) if all_statuses else [row for row in rows if row.successful]
    if best_n is not None:
        scored = [row for row in pool if row.public_score is not None]
        return sorted(scored, key=lambda row: row.public_score, reverse=True)[:best_n]
    ordered = sorted(pool, key=lambda row: row.date, reverse=True)
    if most_recent_n is not None:
        return ordered[:most_recent_n]
    return ordered


def record_rating_history(
    path: Path,
    rows: Sequence[SubmissionStatusRow],
    *,
    leaderboard_rank: int | None = None,
) -> None:
    """
    Snapshot every successful submission's current rating to the local history.

    :param path: Rating-history CSV path.
    :param rows: Rows to record.
    :param leaderboard_rank: Our team's current leaderboard rank, or None if
        not fetched this run.
    :return: None.
    """
    for row in rows:
        append_rating_history(
            path,
            kaggle_ref=row.ref,
            label=row.label,
            status=row.status,
            public_score=row.public_score,
            leaderboard_rank=leaderboard_rank,
        )


def fetch_leaderboard_rank(
    competition: str, team_name: str, *, page_size: int = 200, max_pages: int = 100
) -> int | None:
    """
    Find our team's current 1-based rank on the live leaderboard.

    :param competition: Kaggle competition slug.
    :param team_name: Our team's display name, as shown on the leaderboard.
    :param page_size: Leaderboard rows requested per API page.
    :param max_pages: Maximum pages to scan before giving up.
    :return: Our team's 1-based rank, or None if not found within
        ``max_pages``.
    """
    api = KaggleApi()
    api.authenticate()

    rank = 0
    page_token = ""
    for _ in range(max_pages):
        with api.build_kaggle_client() as kaggle:
            request = ApiGetLeaderboardRequest()
            request.competition_name = competition
            request.page_size = page_size
            request.page_token = page_token
            response = kaggle.competitions.competition_api_client.get_leaderboard(
                request
            )
        rows = response.submissions or []
        for row in rows:
            rank += 1
            if row and row.team_name == team_name:
                return rank
        if not response.next_page_token:
            break
        page_token = response.next_page_token
    return None


def _format_score(score: float | None) -> str:
    """
    :param score: Live skill rating, or None if not yet scored.
    :return: The score to one decimal place, or ``"-"`` if unscored.
    """
    return "-" if score is None else f"{score:.1f}"


def render_table(rows: Sequence[SubmissionStatusRow]) -> None:
    """
    Print submission rows as a Rich table.

    :param rows: Rows to print.
    :return: None.
    """
    table = Table(box=ROUNDED, title="Kaggle submissions", title_justify="left")
    table.add_column("date")
    table.add_column("status")
    table.add_column("score", justify="right")
    table.add_column("label")
    for row in rows:
        table.add_row(
            row.date.strftime("%Y-%m-%d %H:%M"),
            row.status,
            _format_score(row.public_score),
            row.label,
        )
    Console().print(table)


def rows_to_json(rows: Sequence[SubmissionStatusRow]) -> str:
    """
    Serialize submission rows as a JSON array.

    :param rows: Rows to serialize.
    :return: Pretty-printed JSON.
    """

    def _row_dict(row: SubmissionStatusRow) -> dict[str, Any]:
        payload = asdict(row)
        payload["date"] = row.date.isoformat()
        return payload

    return json.dumps([_row_dict(row) for row in rows], indent=2)


def _parser() -> argparse.ArgumentParser:
    """
    :return: The argument parser for the ``status`` subcommand.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fetch live Kaggle submission status, filtered to successful "
            "submissions by default."
        )
    )
    parser.add_argument(
        "--competition", help=f"Kaggle competition (default: {COMPETITION})."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--best-n",
        type=int,
        metavar="N",
        help="Show the N successful submissions with the highest skill rating.",
    )
    selection.add_argument(
        "--most-recent-n",
        type=int,
        metavar="N",
        help=(
            "Show the N most recently submitted (2 mirrors the competition's "
            "active-submission window)."
        ),
    )
    parser.add_argument(
        "--all-statuses",
        action="store_true",
        help="Include PENDING and ERROR submissions instead of only successful ones.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--rank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fetch our team's current leaderboard rank by walking the live "
            "leaderboard until our team turns up (default: on; ~5-10s)."
        ),
    )
    parser.add_argument(
        "--log-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Append every successful submission's current rating to a local "
            "history CSV, building a time series across repeated runs "
            "(default: on)."
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
        submissions = fetch_submissions(competition)
        all_rows = to_rows(submissions)
        leaderboard_rank: int | None = None
        if args.rank and submissions:
            try:
                leaderboard_rank = fetch_leaderboard_rank(
                    competition, submissions[0].team_name
                )
                if leaderboard_rank is None:
                    print(
                        "warning: team not found on the leaderboard "
                        "(no completed submission yet, or beyond the page scan limit).",
                        file=sys.stderr,
                    )
            except Exception as error:  # noqa: BLE001 - report and continue; rank is best-effort
                print(
                    f"warning: could not fetch leaderboard rank: {error}",
                    file=sys.stderr,
                )
        if args.log_history:
            history_path = resolve_local_path(RATING_HISTORY_FILE, root)
            try:
                record_rating_history(
                    history_path,
                    select_rows(all_rows, all_statuses=False),
                    leaderboard_rank=leaderboard_rank,
                )
            except Exception as error:  # noqa: BLE001 - report and continue; history is best-effort
                print(
                    f"warning: could not record rating history: {error}",
                    file=sys.stderr,
                )
        rows = select_rows(
            all_rows,
            all_statuses=args.all_statuses,
            best_n=args.best_n,
            most_recent_n=args.most_recent_n,
        )
    except FetchError as error:
        print(f"error: {error}")
        return 2

    if leaderboard_rank is not None:
        print(f"Leaderboard rank: #{leaderboard_rank} ({submissions[0].team_name})")
    if args.format == "json":
        print(rows_to_json(rows))
    else:
        render_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
