from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from submission_analysis.submissions import (
    record_rating_history,
    rows_to_json,
    select_rows,
    to_rows,
)


@dataclass
class _FakeStatus:
    name: str


@dataclass
class _FakeSubmission:
    ref: int
    description: str
    file_name: str
    date: datetime
    status: _FakeStatus
    public_score: str
    error_description: str = ""
    team_name: str = "test-team"


def _submission(
    *,
    ref: int = 1,
    description: str = "checkpoint-5ac45db92f3e",
    date: datetime,
    status: str = "COMPLETE",
    public_score: str = "",
    error_description: str = "",
) -> _FakeSubmission:
    return _FakeSubmission(
        ref=ref,
        description=description,
        file_name=f"{description}.tar.gz",
        date=date,
        status=_FakeStatus(status),
        public_score=public_score,
        error_description=error_description,
    )


NOW = datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC)


def test_to_rows_marks_complete_without_error_as_successful() -> None:
    submission = _submission(date=NOW, status="COMPLETE", public_score="446.8")
    (row,) = to_rows([submission])
    assert row.successful is True
    assert row.public_score == pytest.approx(446.8)
    assert row.status == "COMPLETE"


def test_to_rows_marks_error_status_as_unsuccessful() -> None:
    submission = _submission(
        date=NOW, status="ERROR", public_score="", error_description="crashed"
    )
    (row,) = to_rows([submission])
    assert row.successful is False
    assert row.public_score is None


def test_to_rows_treats_complete_with_error_description_as_unsuccessful() -> None:
    submission = _submission(
        date=NOW, status="COMPLETE", public_score="1.0", error_description="flaky"
    )
    (row,) = to_rows([submission])
    assert row.successful is False


def test_to_rows_handles_unparseable_score() -> None:
    submission = _submission(date=NOW, status="COMPLETE", public_score="n/a")
    (row,) = to_rows([submission])
    assert row.public_score is None


def test_select_rows_defaults_to_successful_sorted_by_date_desc() -> None:
    older = _submission(ref=1, date=NOW - timedelta(days=1), status="COMPLETE")
    newer = _submission(ref=2, date=NOW, status="COMPLETE")
    errored = _submission(ref=3, date=NOW, status="ERROR")
    rows = to_rows([older, newer, errored])

    selected = select_rows(rows)

    assert [row.ref for row in selected] == [2, 1]


def test_select_rows_all_statuses_includes_errors() -> None:
    errored = _submission(ref=1, date=NOW, status="ERROR")
    rows = to_rows([errored])

    selected = select_rows(rows, all_statuses=True)

    assert [row.ref for row in selected] == [1]


def test_select_rows_best_n_sorts_by_score_descending() -> None:
    low = _submission(ref=1, date=NOW, public_score="100.0")
    high = _submission(ref=2, date=NOW, public_score="500.0")
    rows = to_rows([low, high])

    selected = select_rows(rows, best_n=1)

    assert [row.ref for row in selected] == [2]


def test_select_rows_most_recent_n_limits_and_orders() -> None:
    oldest = _submission(ref=1, date=NOW - timedelta(days=2))
    middle = _submission(ref=2, date=NOW - timedelta(days=1))
    newest = _submission(ref=3, date=NOW)
    rows = to_rows([oldest, middle, newest])

    selected = select_rows(rows, most_recent_n=2)

    assert [row.ref for row in selected] == [3, 2]


def test_record_rating_history_writes_one_row_per_row_regardless_of_display_filter(
    tmp_path,
) -> None:
    """History captures every successful submission, not just what --best-n
    or --most-recent-n narrowed the display to."""
    older = _submission(ref=1, date=NOW - timedelta(days=1), public_score="100.0")
    newer = _submission(ref=2, date=NOW, public_score="500.0")
    rows = to_rows([older, newer])
    history_path = tmp_path / "kaggle_rating_history.csv"

    record_rating_history(history_path, rows)

    with history_path.open(encoding="utf-8", newline="") as history_file:
        written = list(csv.DictReader(history_file))
    assert [row["kaggle_ref"] for row in written] == ["1", "2"]
    assert [row["public_score"] for row in written] == ["100.0", "500.0"]


def test_record_rating_history_stamps_the_same_leaderboard_rank_on_every_row(
    tmp_path,
) -> None:
    """Rank is a team-wide leaderboard position, not per-submission, so every
    row from one run carries the same value."""
    older = _submission(ref=1, date=NOW - timedelta(days=1), public_score="100.0")
    newer = _submission(ref=2, date=NOW, public_score="500.0")
    rows = to_rows([older, newer])
    history_path = tmp_path / "kaggle_rating_history.csv"

    record_rating_history(history_path, rows, leaderboard_rank=42)

    with history_path.open(encoding="utf-8", newline="") as history_file:
        written = list(csv.DictReader(history_file))
    assert [row["leaderboard_rank"] for row in written] == ["42", "42"]


def test_rows_to_json_round_trips_date_and_score() -> None:
    submission = _submission(date=NOW, public_score="446.8")
    rows = to_rows([submission])

    payload = json.loads(rows_to_json(rows))

    assert payload[0]["public_score"] == pytest.approx(446.8)
    assert payload[0]["date"] == NOW.isoformat()
