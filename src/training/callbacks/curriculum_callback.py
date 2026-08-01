import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.training.curriculum import Curriculum

from .base import TrainingCallback

logger = logging.getLogger(__name__)

#: Zero-padding width of the frame counter in dump filenames, matching the
#: snapshot writer so both sort lexicographically by recency.
_FRAME_DIGITS = 12


class CurriculumStateCallback(TrainingCallback):
    """
    Dumps the level buffer to disk at a fixed frame interval.

    Deck selection ranks archetypes on win rates drawn from a *window* of
    training rather than the whole run, because early games reflect a far
    weaker policy. The buffer only carries cumulative tallies, and a window
    cannot be recovered from a single end-of-run total, so the tallies are
    snapshotted periodically and differenced afterwards.

    Writes are atomic, so a dump read concurrently is never partial.
    """

    def __init__(
            self,
            curriculum: Curriculum,
            state_dir: str | Path,
            interval: int,
    ) -> None:
        """
        :param curriculum: Curriculum whose buffer is dumped; shared with the
            trainer, so each write captures the tallies as of that moment.
        :param state_dir: Directory the dumps are written to.
        :param interval: Frames between dumps; ``0`` disables writing.
        """
        self._curriculum = curriculum
        self._state_dir = Path(state_dir)
        self._interval = interval
        self._last_dump_frames = 0

    def on_train_start(self, run_config: Mapping[str, Any]) -> None:  # noqa: ARG002
        """
        Create the dump directory up front.

        :param run_config: Opaque run metadata; unused.
        """
        if self._interval > 0:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Curriculum state every %d frames -> %s", self._interval, self._state_dir
            )

    def on_rollout_start(self, step: int) -> None:
        """
        Unused; dumps are taken after a rollout has been scored.

        :param step: Total frames collected before this rollout.
        """

    def on_rollout_end(self, step: int, metrics: Mapping[str, float]) -> None:  # noqa: ARG002
        """
        Dump the buffer if ``interval`` frames have passed since the last one.

        :param step: Total frames collected so far.
        :param metrics: Running training metrics; unused.
        """
        if self._interval <= 0 or step - self._last_dump_frames < self._interval:
            return
        self._last_dump_frames = step
        self._write(step)

    def on_eval_end(self, step: int, metrics: Mapping[str, float]) -> None:
        """
        Unused; evaluation does not touch the curriculum.

        :param step: Frames collected at the time of the evaluation.
        :param metrics: Evaluation metrics; unused.
        """

    def on_train_end(self, summary: Mapping[str, float]) -> None:
        """
        Write a final dump so the finished tallies are always on disk.

        :param summary: Aggregate run statistics, read for the frame count.
        """
        frames = int(summary.get("frames", self._last_dump_frames))
        if frames > self._last_dump_frames or self._last_dump_frames == 0:
            self._write(frames)

    def _write(self, frames: int) -> None:
        """
        Write one dump named after the frame count.

        :param frames: Frame count to embed in the filename.
        """
        path = self._state_dir / f"curriculum_{frames:0{_FRAME_DIGITS}d}.pt"
        self._curriculum.save_state(path)
        logger.info("Wrote curriculum state %s", path.name)
