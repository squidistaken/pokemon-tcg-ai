import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from src.training.callbacks.base import TrainingCallback

logger = logging.getLogger(__name__)


class TrainStateCallback(TrainingCallback):
    """
    Periodically writes everything a run needs to be picked up where it stopped.

    The league snapshots :class:`~src.training.callbacks.snapshot_callback.SnapshotCallback`
    writes hold weights only, which is enough to *serve* a policy and enough to
    warm-start one, but not to continue optimizing it: Adam's moment estimates
    live in the optimizer, and restarting them from zero makes the first
    rollouts after a restart noisier than the ones that preceded it. This
    callback closes that gap.

    It writes one rolling file rather than a numbered series. A resume only ever
    wants the newest state, and the optimizer's two moment tensors per parameter
    make each write roughly three times the size of a weights-only snapshot --
    keeping every one of them would cost far more disk than the league does, for
    states nothing will ever read.

    The write is staged through a temporary file and renamed into place, so a
    run killed mid-write leaves the previous state intact rather than a
    truncated file that fails to load.
    """

    def __init__(
            self,
            actor_critic: nn.Module,
            optimizer: Optimizer,
            path: str | Path,
            interval: int,
    ) -> None:
        """
        :param actor_critic: Network whose weights accompany the optimizer
            state, so the pair is guaranteed to come from the same step.
        :param optimizer: Optimizer whose state is being preserved.
        :param path: Destination file, overwritten on every write.
        :param interval: Frames between writes; ``0`` disables them, leaving
            only the end-of-run write.
        """
        self._actor_critic = actor_critic
        self._optimizer = optimizer
        self._path = Path(path)
        self._interval = interval
        self._last_write = 0

    def on_train_start(self, run_config: Mapping[str, Any]) -> None:
        """
        Announce where the state will be written.

        :param run_config: Resolved run config; unused.
        """
        del run_config
        if self._interval > 0:
            logger.info(
                "Training state every %d frames -> %s", self._interval, self._path
            )

    def on_rollout_start(self, step: int) -> None:
        """
        Ignore the rollout start; state is written once a rollout completes.

        :param step: Total frames collected before this rollout.
        """

    def on_eval_end(self, step: int, metrics: Mapping[str, float]) -> None:
        """
        Ignore evaluation; the state is keyed to collection, not scoring.

        :param step: Total frames collected so far.
        :param metrics: Evaluation metrics.
        """

    def on_rollout_end(self, step: int, metrics: Mapping[str, float]) -> None:
        """
        Write the state when the interval has elapsed.

        :param step: Total frames collected so far.
        :param metrics: Running training metrics; unused.
        """
        del metrics
        if self._interval <= 0 or step - self._last_write < self._interval:
            return
        self._last_write = step
        self._write(step)

    def on_train_end(self, summary: Mapping[str, float]) -> None:
        """
        Write a final state, so a run that ends on a partial interval is
        still resumable from where it actually stopped.

        :param summary: Aggregate run statistics, read for the frame count.
        """
        self._write(int(summary.get("frames", self._last_write)))

    def _write(self, frames: int) -> None:
        """
        Stage the state to a temporary file and rename it into place.

        :param frames: Absolute frame count this state was captured at.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "format_version": 1,
            "state_dict": self._actor_critic.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "frames": frames,
            "torch_rng_state": torch.get_rng_state(),
        }
        staging = self._path.with_suffix(self._path.suffix + ".tmp")
        torch.save(state, staging)
        staging.replace(self._path)
        logger.info("Wrote training state at %d frames -> %s", frames, self._path)
