from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)


class TrainingCallback:
    """
    Interface for observers of a training run, which emits metric mappings at
    fixed points.
    """

    def on_train_start(self, run_config: Mapping[str, Any]) -> None:
        """
        Called once before the first batch is collected.

        :param run_config: Opaque run metadata (in practice the resolved Hydra
            config) for backends that record hyperparameters alongside metrics.
        """

    def on_batch_end(self, step: int, metrics: Mapping[str, float]) -> None:
        """
        Called after every collected batch and its algorithm update.

        :param step: Monotonically increasing x-axis for the metrics; the total
            number of frames collected so far.
        :param metrics: Running training metrics at ``step``.
        """

    def on_eval_end(self, step: int, metrics: Mapping[str, float]) -> None:
        """
        Called after an evaluation round. Nothing calls this yet — the Evaluator
        is not built; the hook exists so eval metrics reach the same backends.

        :param step: Frames collected at the time of the evaluation.
        :param metrics: Evaluation metrics (e.g. win rate against a baseline).
        """

    def on_train_end(self, summary: Mapping[str, float]) -> None:
        """
        Called once after the last batch, including on failure.

        The teardown hook (closing files, finishing runs). Must tolerate being
        called after :meth:`on_train_start` raised or never ran.

        :param summary: Aggregate statistics for the whole run.
        """


class CallbackList(TrainingCallback):
    """
    Composite that fans each hook out to its members in order.

    Being a :class:`TrainingCallback` itself lets the trainer hold exactly one
    callback and stay unaware of how many are attached.

    A raising member is logged with its traceback and skipped for that hook, not
    propagated: logging is a side channel, and a W&B blip should not destroy an
    otherwise healthy run.
    """

    def __init__(self, callbacks: Iterable[TrainingCallback] = ()) -> None:
        """
        :param callbacks: Members to fan out to, invoked in iteration order.
        """
        self._callbacks: list[TrainingCallback] = list(callbacks)

    @property
    def callbacks(self) -> Sequence[TrainingCallback]:
        """
        The attached members.

        :return: Members in invocation order.
        """
        return tuple(self._callbacks)

    def __len__(self) -> int:
        """
        :return: Number of attached members.
        """
        return len(self._callbacks)

    def _dispatch(self, hook: str, *args: Any) -> None:
        """
        Call ``hook`` on every member, isolating failures.

        :param hook: Name of the :class:`TrainingCallback` method to invoke.
        :param args: Positional arguments forwarded to the hook.
        """
        for callback in self._callbacks:
            try:
                getattr(callback, hook)(*args)
            except Exception:
                logger.exception(
                    "Callback %s.%s raised; continuing without it for this hook.",
                    type(callback).__name__,
                    hook,
                )

    def on_train_start(self, run_config: Mapping[str, Any]) -> None:
        """
        Forward the run start to every member.

        :param run_config: Opaque run metadata.
        """
        self._dispatch("on_train_start", run_config)

    def on_batch_end(self, step: int, metrics: Mapping[str, float]) -> None:
        """
        Forward the batch metrics to every member.

        :param step: Total frames collected so far.
        :param metrics: Running training metrics at ``step``.
        """
        self._dispatch("on_batch_end", step, metrics)

    def on_eval_end(self, step: int, metrics: Mapping[str, float]) -> None:
        """
        Forward the evaluation metrics to every member.

        :param step: Frames collected at the time of the evaluation.
        :param metrics: Evaluation metrics.
        """
        self._dispatch("on_eval_end", step, metrics)

    def on_train_end(self, summary: Mapping[str, float]) -> None:
        """
        Forward the run summary to every member.

        :param summary: Aggregate statistics for the whole run.
        """
        self._dispatch("on_train_end", summary)
