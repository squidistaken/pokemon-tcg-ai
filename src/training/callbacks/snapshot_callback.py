import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.models.actor_critic import ActorCritic
from src.policies.greedy_policy_opponent import save_actor_critic

from .base import TrainingCallback

logger = logging.getLogger(__name__)

#: Zero-padding width of the frame counter in snapshot filenames, chosen so
#: lexicographic order matches numeric order for any realistic run length.
_FRAME_DIGITS = 12


class SnapshotCallback(TrainingCallback):
    """
    Freezes the learner into the self-play league at a fixed frame interval.

    This is the writing half of the self-play loop: every ``interval``
    collected frames it snapshots the actor-critic into ``checkpoint_dir``,
    where the environment workers'
    :class:`~src.env.snapshot_opponent_pool.SnapshotOpponentPool` instances
    discover it and start playing against it.

    Writes go to a temporary file that is then atomically renamed into place,
    so a worker scanning the directory concurrently never observes a partially
    written snapshot.

    Filenames embed the zero-padded frame count, so sorting them lexicographic-
    ally orders them by recency.
    """

    def __init__(
            self,
            actor_critic: ActorCritic,
            checkpoint_dir: str | Path,
            interval: int,
    ) -> None:
        """
        :param actor_critic: Learner to snapshot; shared with the trainer, so
            each write captures the parameters as of that moment.
        :param checkpoint_dir: Directory snapshots are written to.
        :param interval: Frames between snapshots; ``0`` disables writing.
        """
        self._actor_critic = actor_critic
        self._checkpoint_dir = Path(checkpoint_dir)
        self._interval = interval
        self._last_snapshot_frames = 0

    def on_train_start(self, run_config: Mapping[str, Any]) -> None:  # noqa: ARG002
        """
        Create the snapshot directory up front so workers can scan it.

        :param run_config: Opaque run metadata; unused.
        """
        if self._interval > 0:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Self-play snapshots every %d frames -> %s", self._interval, self._checkpoint_dir)

    def on_rollout_start(self, step: int) -> None:
        """
        Unused; snapshots are taken after a rollout's update has been applied.

        :param step: Total frames collected before this rollout.
        """

    def on_rollout_end(self, step: int, metrics: Mapping[str, float]) -> None:  # noqa: ARG002
        """
        Snapshot the learner if ``interval`` frames have passed since the last.

        :param step: Total frames collected so far.
        :param metrics: Running training metrics; unused.
        """
        if self._interval <= 0 or step - self._last_snapshot_frames < self._interval:
            return
        self._last_snapshot_frames = step
        self._write(step)

    def on_eval_end(self, step: int, metrics: Mapping[str, float]) -> None:
        """
        Unused; evaluation does not change the learner.

        :param step: Frames collected at the time of the evaluation.
        :param metrics: Evaluation metrics; unused.
        """

    def on_train_end(self, summary: Mapping[str, float]) -> None:
        """
        Write a final snapshot so the finished policy is always on disk.

        Skipped when the run happened to end exactly on a snapshot boundary,
        which would otherwise rewrite the same parameters under the same name.

        :param summary: Aggregate run statistics, read for the frame count.
        """
        frames = int(summary.get("frames", self._last_snapshot_frames))
        if self._interval > 0 and frames > self._last_snapshot_frames:
            self._write(frames)

    def _write(self, frames: int) -> None:
        """
        Atomically write one snapshot named after the frame count.

        :param frames: Frame count to embed in the filename.
        """
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        final_path = self._checkpoint_dir / f"snapshot_{frames:0{_FRAME_DIGITS}d}.pt"
        # Same directory (os.replace is only atomic within a filesystem) and a
        # suffix the pool's "*.pt" scan does not match.
        staging_path = final_path.parent / f"{final_path.name}.tmp"
        save_actor_critic(self._actor_critic, staging_path)
        os.replace(staging_path, final_path)
        logger.info("Wrote self-play snapshot %s", final_path.name)
