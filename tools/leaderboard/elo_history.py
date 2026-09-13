"""
Turn raw submissions and the public leaderboard into the numbers the report shows.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from statistics import median
from typing import Any

from leaderboard_row import LeaderboardRow
from submission_record import SubmissionRecord


class EloHistory:
    """
    Combine our submission list with the field, and expose it as a plain payload.

    The submission list gives one rating per agent we uploaded. The leaderboard
    gives the field those ratings sit in: the leader, the middle of the pack and
    our own rank. Both come from the same fetch, so the two agree in time.
    """

    def __init__(
        self,
        competition: str,
        submissions: Sequence[SubmissionRecord],
        leaderboard: Sequence[LeaderboardRow],
        username: str | None = None,
        team_name: str | None = None,
    ) -> None:
        """
        :param competition: Competition slug the data comes from.
        :param submissions: Our own submissions, in any order.
        :param leaderboard: Public leaderboard snapshot.
        :param username: Kaggle account used to find our team on the leaderboard.
        :param team_name: Team name override, used when the account lookup fails.
        """
        self.competition = competition
        self.submissions = sorted(submissions, key=lambda record: record.submitted_at)
        self.leaderboard = list(leaderboard)
        self.username = username
        self.team_name = team_name
        self.fetched_at = datetime.now(UTC)

    @property
    def scored(self) -> list[SubmissionRecord]:
        """
        Give the submissions that carry a rating, oldest first.
        """
        return [record for record in self.submissions if record.is_scored]

    @property
    def unscored(self) -> list[SubmissionRecord]:
        """
        Give the submissions that never produced a rating.
        """
        return [record for record in self.submissions if not record.is_scored]

    @property
    def best(self) -> SubmissionRecord | None:
        """
        Give the highest rated submission.
        """
        scored = self.scored
        return max(scored, key=lambda record: record.score or 0.0) if scored else None

    @property
    def team_row(self) -> LeaderboardRow | None:
        """
        Find our team on the leaderboard, by account name first and by team name second.
        """
        if self.username:
            for row in self.leaderboard:
                if row.has_member(self.username):
                    return row
        if self.team_name:
            wanted = self.team_name.strip().lower()
            for row in self.leaderboard:
                if row.team_name.strip().lower() == wanted:
                    return row
        return None

    def _field(self) -> dict[str, Any]:
        """
        Summarise the field: the leader, the middle of the pack and our own place in it.
        """
        if not self.leaderboard:
            return {}
        scores = [row.score for row in self.leaderboard]
        leader = self.leaderboard[0]
        row = self.team_row
        field: dict[str, Any] = {
            "top_score": leader.score,
            "top_team": leader.team_name,
            "median_score": round(median(scores), 1),
            "team_count": len(self.leaderboard),
        }
        if row is not None:
            field.update(
                {
                    "team_name": row.team_name,
                    "team_score": row.score,
                    "team_rank": row.rank,
                    "team_percentile": round(
                        100.0 * row.rank / len(self.leaderboard), 1
                    ),
                }
            )
        return field

    def to_payload(self) -> dict[str, Any]:
        """
        Build the JSON-ready structure the HTML report renders from.

        Points carry a running best so the chart can draw the envelope without
        recomputing it, and the raw description so the tooltip can show what the
        agent actually was.

        :return: Dictionary with the points, the failed uploads and the field context.
        """
        points = []
        running_best = float("-inf")
        previous_best = None
        for record in self.scored:
            score = float(record.score or 0.0)
            gain = None if previous_best is None else round(score - previous_best, 1)
            running_best = max(running_best, score)
            previous_best = running_best
            points.append(
                {
                    "ref": record.ref,
                    "label": record.label,
                    "date": record.submitted_at.strftime("%Y-%m-%d %H:%M"),
                    "timestamp": record.submitted_at.timestamp() * 1000.0,
                    "score": score,
                    "best": running_best,
                    "gain_vs_best": gain,
                    "status": record.short_status,
                    "description": record.description,
                }
            )
        failures = [
            {
                "label": record.label,
                "date": record.submitted_at.strftime("%Y-%m-%d %H:%M"),
                "status": record.short_status,
                "description": record.description,
            }
            for record in self.unscored
        ]
        return {
            "competition": self.competition,
            "fetched_at": self.fetched_at.strftime("%Y-%m-%d %H:%M UTC"),
            "points": points,
            "failures": failures,
            "field": self._field(),
        }
