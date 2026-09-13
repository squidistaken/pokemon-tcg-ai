"""
One team on the public leaderboard.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class LeaderboardRow:
    """
    A public leaderboard entry, used as field context for our own scores.
    """

    rank: int
    team_id: str
    team_name: str
    last_submission_at: datetime | None
    score: float
    submission_count: int
    member_usernames: tuple[str, ...]

    def has_member(self, username: str) -> bool:
        """
        Tell whether a Kaggle username belongs to this team.

        :param username: Kaggle account name to look for.
        :return: True when the account is one of the team members.
        """
        wanted = username.strip().lower()
        return any(member.strip().lower() == wanted for member in self.member_usernames)
