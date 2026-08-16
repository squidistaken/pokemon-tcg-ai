"""
One Kaggle submission and the competition score it currently holds.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SubmissionRecord:
    """
    A single row of the Kaggle submission list.

    The score is the rating the agent holds right now, not the rating it had on
    the submission date: a simulation competition keeps replaying every active
    agent, so the value moves after the upload.
    """

    ref: str
    file_name: str
    submitted_at: datetime
    description: str
    status: str
    score: float | None

    @property
    def is_scored(self) -> bool:
        """
        Tell whether the submission has a rating to plot.
        """
        return self.score is not None

    @property
    def label(self) -> str:
        """
        Give the short agent name, which is the bundle name without archive suffixes.
        """
        name = self.file_name
        for suffix in (".tar.gz", ".tgz", ".zip", ".tar"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    @property
    def short_status(self) -> str:
        """
        Give the status without the `SubmissionStatus.` prefix the API adds.
        """
        return self.status.rsplit(".", 1)[-1].lower()
