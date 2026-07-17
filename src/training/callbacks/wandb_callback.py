from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from src.training.callbacks.base import TrainingCallback

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

logger = logging.getLogger(__name__)

WandbMode = Literal["online", "offline", "disabled", "shared"]
_VALID_MODES: tuple[str, ...] = get_args(WandbMode)


class WeightsAndBiases(TrainingCallback):
    """
    Weights & Biases backend for training metrics.

    The only module in the project that imports ``wandb``, so dropping or
    swapping the backend touches this file and a config group, nothing else.
    Metrics are namespaced ``train/`` and ``eval/``; run aggregates go to the run
    summary rather than the step series.

    Credentials come from ``WANDB_API_KEY`` (which :mod:`src.train` loads from an
    untracked ``.env``) or ``wandb login``. See :meth:`_resolve_mode` for what
    happens without them.
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
    ) -> None:
        """
        :param project: W&B project to log the run under.
        :param entity: W&B team or user; None uses the account default.
        :param name: Display name for the run; None lets W&B generate one.
        :param group: Group label, keeping related runs together in the UI.
        :param job_type: Job-type label within the group (e.g. ``train``).
        :param tags: Free-form tags attached to the run.
        :param mode: ``online``, ``offline`` (record for a later ``wandb sync``),
            ``disabled`` (drop everything) or ``shared``.
        :param notes: Free-text note attached to the run.
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
        # Annotated so the narrowing in _resolve_mode keeps the Literal wandb wants.
        self._mode: WandbMode = cast(WandbMode, mode)
        self._notes = notes
        self._run: Run | None = None

    @staticmethod
    def _has_credentials() -> bool:
        """
        Report whether W&B can authenticate on this machine.

        Uses ``wandb.api.api_key``, which resolves ``WANDB_API_KEY`` *and*
        ``~/.netrc``. Do not swap this for the environment variable alone: it
        would downgrade anyone who authenticated with ``wandb login``.

        :return: True if a key is resolvable, or if the probe itself fails —
            ``wandb.api`` is semi-internal, so defer to wandb rather than guess.
        """
        import wandb

        try:
            return bool(wandb.api.api_key)
        except Exception:
            logger.debug("Could not probe W&B credentials; assuming present.", exc_info=True)
            return True

    def _resolve_mode(self) -> WandbMode:
        """
        Adapt the configured mode to the credentials actually available.

        An ``online`` run without credentials blocks on a login prompt in a
        terminal and fails in CI, so record offline instead — the metrics survive
        and stay syncable. Other modes need no credentials and are left alone.

        :return: The mode to open the run with.
        """
        if self._mode != "online" or self._has_credentials():
            return self._mode
        logger.warning(
            "W&B mode is 'online' but no credentials were found (no WANDB_API_KEY "
            "and no `wandb login` entry in ~/.netrc): recording this run OFFLINE "
            "instead. The metrics are kept — push them with `wandb sync` on the "
            "run directory below. To log online, put WANDB_API_KEY in .env (see "
            ".env.example) or run `wandb login`."
        )
        return "offline"

    def on_train_start(self, run_config: Mapping[str, Any]) -> None:
        """
        Start the W&B run and record the run config as its hyperparameters.

        ``wandb`` is imported here, not at module scope: the import is slow and
        has side effects, and this keeps it off the path of every non-W&B run.

        :param run_config: Opaque run metadata, recorded as the run's config.
        """
        import wandb

        mode = self._resolve_mode()
        self._run = wandb.init(
            project=self._project,
            entity=self._entity,
            name=self._name,
            group=self._group,
            job_type=self._job_type,
            tags=self._tags,
            mode=mode,
            notes=self._notes,
            config=dict(run_config),
        )
        logger.info(
            "W&B run started: %s (%s, mode=%s)",
            self._run.name,
            self._run.url or self._run.id,
            mode,
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
        self._log("eval", step, metrics)

    def _log(self, prefix: str, step: int, metrics: Mapping[str, float]) -> None:
        """
        Send namespaced metrics to the active run, if there is one.

        No-ops without a run: a failed :meth:`on_train_start` is logged and
        training continues, so the later hooks must not raise on every batch.

        :param prefix: Namespace for the metric keys (``train`` or ``eval``).
        :param step: Value for the ``frames`` x-axis.
        :param metrics: Metrics to log under ``prefix``.
        """
        if self._run is None:
            return
        self._run.log({f"{prefix}/{key}": value for key, value in metrics.items()}, step=step)

    def on_train_end(self, summary: Mapping[str, float]) -> None:
        """
        Record the run aggregates and close the run.

        Aggregates go to the summary, not the step series: they describe the
        whole run, so they are what the W&B run table sorts on.

        :param summary: Aggregate statistics for the whole run.
        """
        if self._run is None:
            return
        for key, value in summary.items():
            self._run.summary[f"summary/{key}"] = value
        self._run.finish()
        logger.info("W&B run finished: %s", self._run.id)
        self._run = None
