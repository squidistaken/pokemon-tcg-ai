from __future__ import annotations

import logging
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

from src.training.callbacks.base import TrainingCallback

logger = logging.getLogger(__name__)

_ARCHETYPE_PREFIX = "archetype_win_rate/"

WandbMode = Literal["online", "offline", "disabled"]
_VALID_MODES: tuple[str, ...] = get_args(WandbMode)


class WeightsAndBiases(TrainingCallback):
    """
    Weights & Biases backend for training metrics.

    The only module in the project that imports ``wandb``, so dropping or
    swapping the backend touches this file and a config group, nothing else.
    Metrics are namespaced ``train/`` and ``eval/``; run aggregates go to the run
    summary rather than the step series.

    Credentials come from ``WANDB_API_KEY`` (which :mod:`src.train` loads from an
    untracked ``.env``) or ``wandb login``. Online mode is strict: missing or
    invalid credentials and connection failures stop training.
    """

    def __init__(
            self,
            project: str,
            entity: str | None = None,
            name: str | None = None,
            group: str | None = None,
            job_type: str | None = None,
            tags: Sequence[str] | None = None,
            mode: str = "online",
            notes: str | None = None,
            dir: str | None = None,
            log_checkpoints: bool = True,
    ) -> None:
        """
        :param project: W&B project to log the run under.
        :param entity: W&B team or user; None uses the account default.
        :param name: Display name for the run; None lets W&B generate one.
        :param group: Group label, keeping related runs together in the UI.
        :param job_type: Job-type label within the group (e.g. ``train``).
        :param tags: Free-form tags attached to the run.
        :param mode: ``online``, ``offline`` (record for a later ``wandb sync``)
            or ``disabled`` (drop everything).
        :param notes: Free-text note attached to the run.
        :param dir: Parent directory for W&B's local run files.
        :param log_checkpoints: Mirror written model checkpoints as versioned
            W&B model artifacts.
        :raises ValueError: If ``mode`` is not a mode W&B accepts. Checked here so
            a config typo fails before the environments are built.
        """
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid W&B mode {mode!r}; expected one of {', '.join(_VALID_MODES)}."
            )
        self._project = project
        self._entity = entity
        self._name = name
        self._group = group
        self._job_type = job_type
        self._tags = list(tags) if tags is not None else None
        self._mode: WandbMode = cast(WandbMode, mode)
        self._notes = notes
        self._dir = dir
        self._log_checkpoints = log_checkpoints
        self._run: Run | None = None
        self._latest_archetype_rates: dict[str, float] = {}
        self._failure: BaseException | None = None

    def on_train_start(self, run_config: Mapping[str, Any]) -> None:
        """
        Start the W&B run and record the run config as its hyperparameters.

        ``wandb`` is imported here, not at module scope: the import is slow and
        has side effects, and this keeps it off the path of every non-W&B run.

        :param run_config: Opaque run metadata, recorded as the run's config.
        """
        import wandb

        self._run = wandb.init(
            project=self._project,
            entity=self._entity,
            name=self._name,
            group=self._group,
            job_type=self._job_type,
            tags=self._tags,
            mode=self._mode,
            notes=self._notes,
            dir=self._dir,
            config=dict(run_config),
            force=self._mode == "online",
            # ParallelEnv workers inherit the parent's file descriptors. Using
            # low-level redirection captures their stdout and stderr too, while
            # W&B's default stream wrapping only sees writes in this process.
            settings=wandb.Settings(
                console="redirect",
                # ``redirect`` preserves low-level worker output in output.log,
                # but those records may not populate W&B's Logs tab. Send the
                # application's Python logs there through W&B's supported
                # logger integration as well.
                capture_loggers={"root": "INFO"},
            ),
        )
        logger.info(
            "W&B run started: %s (%s, mode=%s)",
            self._run.name,
            self._run.url or self._run.id,
            self._mode,
        )

    def on_rollout_start(self, step: int) -> None:
        """
        Trigger on rollout start; W&B logs a training point at :meth:`on_rollout_end`,
        so there is nothing to record when a rollout begins.

        :param step: Total frames collected before this rollout; unused.
        """

    def on_rollout_end(self, step: int, metrics: Mapping[str, float]) -> None:
        """
        Log one training point, x-axed by collected frames.

        :param step: Total frames collected so far.
        :param metrics: Running training metrics at ``step``.
        """
        self._log("train", step, metrics)

    def on_eval_end(self, step: int, metrics: Mapping[str, float]) -> None:
        """
        Log one evaluation point against the same frame axis as training.

        :param step: Frames collected at the time of the evaluation.
        :param metrics: Evaluation metrics.
        """
        archetype_rates = {
            key[len(_ARCHETYPE_PREFIX):]: value
            for key, value in metrics.items()
            if key.startswith(_ARCHETYPE_PREFIX)
        }
        if archetype_rates:
            self._latest_archetype_rates = archetype_rates
        summary = {
            key: value for key, value in metrics.items() if not key.startswith(_ARCHETYPE_PREFIX)
        }
        self._log("eval", step, summary)

    def log_table(self, key: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        """
        Log a one-shot table to the run.
        
        :param key: W&B key the table is logged under.
        :param columns: Column names.
        :param rows: Table rows, one sequence of cell values per row.
        """
        if self._run is None:
            return
        import wandb

        table = wandb.Table(columns=list(columns), data=[list(row) for row in rows])
        self._run.log({key: table})

    def log_checkpoint(self, path: Path, digest: str, frames: int) -> None:
        """
        Log a local checkpoint as a version of this run's model artifact.

        The stable artifact collection is keyed by W&B run ID. ``latest`` and
        the short SHA-256 key make either the newest or an exact checkpoint
        easy to retrieve from W&B without changing the local submission flow.

        :param path: Local checkpoint path.
        :param digest: Full SHA-256 digest of the checkpoint.
        :param frames: Collected-frame counter captured by the checkpoint.
        """
        if self._run is None or not self._log_checkpoints:
            return
        key = digest[:12]
        self._run.log_artifact(
            str(path),
            name=f"checkpoint-{self._run.id}",
            type="model",
            aliases=["latest", f"sha-{key}", f"frames-{frames}"],
        )
        logger.info("W&B checkpoint artifact logged: sha-%s", key)

    def _log(self, prefix: str, step: int, metrics: Mapping[str, float]) -> None:
        """
        Send namespaced metrics to the active run, if there is one.

        No-ops without a run so teardown remains safe if startup failed before
        W&B returned a run object.

        :param prefix: Namespace for the metric keys (``train`` or ``eval``).
        :param step: Value for the ``frames`` x-axis.
        :param metrics: Metrics to log under ``prefix``.
        """
        if self._run is None:
            return
        self._run.log({f"{prefix}/{key}": value for key, value in metrics.items()}, step=step)

    def on_train_error(self, error: BaseException) -> None:
        """
        Remember that the run is ending in failure.

        Recorded rather than acted on immediately, because the run still has to
        be closed by :meth:`on_train_end`; this only decides the exit code it
        closes with.

        :param error: The exception that ended the run.
        """
        self._failure = error
        if self._run is not None:
            self._run.summary["summary/error"] = f"{type(error).__name__}: {error}"
            # The run is finished during callback teardown, before the
            # interpreter prints an uncaught exception. Publish the traceback
            # now so it is visible in W&B's Logs tab while the run is active.
            self._run.write_logs("".join(traceback.format_exception(error)).rstrip())

    def on_train_end(self, summary: Mapping[str, float]) -> None:
        """
        Record the run aggregates and close the run.

        Aggregates go to the summary, not the step series: they describe the
        whole run, so they are what the W&B run table sorts on.

        A run that ended in failure is finished with a non-zero exit code, so
        W&B marks it ``crashed`` rather than ``finished``. Without that, a run
        that died partway through is indistinguishable in the UI from a short
        successful one -- which is exactly how a mid-run environment crash came
        to look like a completed 475k-frame run.

        :param summary: Aggregate statistics for the whole run.
        """
        if self._run is None:
            return
        if self._latest_archetype_rates:
            self.log_table(
                "eval/archetype_win_rate_table",
                ["archetype", "win_rate"],
                sorted(self._latest_archetype_rates.items()),
            )
        for key, value in summary.items():
            self._run.summary[f"summary/{key}"] = value
        run_id = self._run.id
        if self._failure is None:
            self._run.finish()
            logger.info("W&B run finished: %s", run_id)
        else:
            self._run.finish(exit_code=1)
            logger.warning("W&B run marked failed: %s (%s)", run_id, self._failure)
        self._run = None
