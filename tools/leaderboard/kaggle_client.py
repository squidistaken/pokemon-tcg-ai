"""
Read submissions and the public leaderboard through the Kaggle CLI.
"""

import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from leaderboard_row import LeaderboardRow
from submission_record import SubmissionRecord

DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
)
UNKNOWN_DATE = datetime(1970, 1, 1, tzinfo=UTC)


def parse_timestamp(raw: str) -> datetime | None:
    """
    Parse the timestamp formats the Kaggle CLI emits.

    Kaggle reports in UTC, so the result is stamped UTC rather than left naive.

    :param raw: Timestamp text, possibly empty or with a trailing `Z`.
    :return: The parsed datetime, or None when the field is empty or unknown.
    """
    text = raw.strip().rstrip("Z")
    if not text:
        return None
    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(text, date_format).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def parse_float(raw: str) -> float | None:
    """
    Parse a score field that is empty for submissions that never scored.

    :param raw: Score text from the CSV.
    :return: The score, or None when the field is empty or not a number.
    """
    text = raw.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


class KaggleCompetitionClient:
    """
    Fetch the live competition data with the `kaggle` command line client.

    Every call goes out to Kaggle, so a rerun of the report picks up the ratings
    as they stand at that moment. The CLI is called with PYTHONPATH removed,
    because the repo root on PYTHONPATH shadows the packaged `submission`
    module and breaks unrelated imports inside the tool.
    """

    def __init__(self, competition: str, executable: str | None = None) -> None:
        """
        :param competition: Kaggle competition slug, for example `pokemon-tcg-ai-battle`.
        :param executable: Path to the `kaggle` binary; resolved from PATH when omitted.
        """
        self.competition = competition
        self.executable = executable or self._resolve_executable()

    @staticmethod
    def _resolve_executable() -> str:
        """
        Find the `kaggle` binary, preferring the one next to the running interpreter.
        """
        candidate = Path(sys.executable).parent / "kaggle"
        if candidate.exists():
            return str(candidate)
        found = shutil.which("kaggle")
        if found is None:
            raise RuntimeError(
                "the kaggle CLI is not on PATH; install it with `uv pip install kaggle`"
            )
        return found

    def _run(self, arguments: Iterable[str], cwd: Path | None = None) -> str:
        """
        Run the CLI and return its standard output.

        :param arguments: Arguments after the `kaggle` executable.
        :param cwd: Working directory for the call.
        :return: Captured standard output.
        """
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [self.executable, *arguments],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
            cwd=None if cwd is None else str(cwd),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"kaggle {' '.join(arguments)} failed:\n{completed.stderr.strip()}"
            )
        return completed.stdout

    @staticmethod
    def _csv_rows(payload: str, first_column: str) -> list[dict]:
        """
        Parse CLI output that holds a CSV table after optional progress lines.

        :param payload: Raw standard output.
        :param first_column: Name of the first CSV column, used to find the header.
        :return: Rows as dictionaries.
        """
        lines = payload.splitlines()
        for index, line in enumerate(lines):
            if line.lstrip("﻿").startswith(first_column + ","):
                table = "\n".join(lines[index:])
                return list(csv.DictReader(io.StringIO(table.lstrip("﻿"))))
        raise RuntimeError(
            f"no CSV table with a `{first_column}` column in the kaggle output:\n{payload[:400]}"
        )

    @staticmethod
    def username() -> str | None:
        """
        Read the Kaggle account name from the environment or the stored credentials.

        :return: The account name, or None when no credential file is present.
        """
        from_environment = os.environ.get("KAGGLE_USERNAME")
        if from_environment:
            return from_environment
        for path in (
            Path.home() / ".kaggle" / "credentials.json",
            Path.home() / ".kaggle" / "kaggle.json",
        ):
            if path.exists():
                try:
                    stored = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue
                name = stored.get("username")
                if name:
                    return str(name)
        return None

    def fetch_submissions(self) -> list[SubmissionRecord]:
        """
        List our own submissions with the score each one holds right now.

        :return: Records ordered oldest first.
        """
        rows = self._csv_rows(
            self._run(["competitions", "submissions", "-c", self.competition, "-v"]),
            "ref",
        )
        records = [
            SubmissionRecord(
                ref=row.get("ref", "").strip(),
                file_name=row.get("fileName", "").strip(),
                submitted_at=parse_timestamp(row.get("date", "")) or UNKNOWN_DATE,
                description=row.get("description", "").strip(),
                status=row.get("status", "").strip(),
                score=parse_float(row.get("publicScore", "")),
            )
            for row in rows
        ]
        return sorted(records, key=lambda record: record.submitted_at)

    def fetch_leaderboard(self) -> list[LeaderboardRow]:
        """
        Download the full public leaderboard snapshot.

        :return: Rows ordered by rank.
        """
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            self._run(
                [
                    "competitions",
                    "leaderboard",
                    "-c",
                    self.competition,
                    "-d",
                    "-p",
                    str(target),
                ]
            )
            archives = sorted(target.glob("*.zip"))
            if not archives:
                raise RuntimeError("the kaggle CLI downloaded no leaderboard archive")
            with zipfile.ZipFile(archives[0]) as archive:
                member = next(
                    name for name in archive.namelist() if name.endswith(".csv")
                )
                payload = archive.read(member).decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(payload)))
        entries = []
        for row in rows:
            score = parse_float(row.get("Score", ""))
            if score is None:
                continue
            members = tuple(
                part
                for part in row.get("TeamMemberUserNames", "").split(",")
                if part.strip()
            )
            entries.append(
                LeaderboardRow(
                    rank=int(row.get("Rank", "0") or 0),
                    team_id=row.get("TeamId", "").strip(),
                    team_name=row.get("TeamName", "").strip(),
                    last_submission_at=parse_timestamp(
                        row.get("LastSubmissionDate", "")
                    ),
                    score=score,
                    submission_count=int(row.get("SubmissionCount", "0") or 0),
                    member_usernames=members,
                )
            )
        return sorted(entries, key=lambda entry: entry.rank)
