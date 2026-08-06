from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


class TrainingCallback(ABC):
    """
    Interface for observers of a training run, which emits metric mappings at fixed points.

    The lifecycle brackets each rollout the :meth:`~src.training.trainer.Trainer.train` collects and
    hands to a single ``_update``. A rollout is deliberately distinct from the minibatch iterations
    an algorithm may run *inside* that update; the ``on_batch_*`` name is left free for that finer-grained level.
    """

    @abstractmethod
    def on_train_start(self, run_config: Mapping[str, Any]) -> None:
        """
        Called once before the first rollout is collected.

        :param run_config: Opaque run metadata (in practice the resolved Hydra
            config) for backends that record hyperparameters alongside metrics.
        """

    @abstractmethod
    def on_rollout_start(self, step: int) -> None:
        """
        Called at the start of each rollout, before it is collected and updated.

        :param step: Total frames collected before this rollout (its x-axis lower
            bound); ``0`` for the first rollout.
        """

    @abstractmethod
    def on_rollout_end(self, step: int, metrics: Mapping[str, float]) -> None:
        """
        Called after each rollout has been collected and its algorithm update
        applied.

        :param step: Monotonically increasing x-axis for the metrics; the total
            number of frames collected so far, including this rollout.
        :param metrics: Running training metrics at ``step``.
        """

    @abstractmethod
    def on_eval_end(self, step: int, metrics: Mapping[str, float]) -> None:
        """
        Called after an evaluation round. Nothing calls this yet — the Evaluator
        is not built; the hook exists so eval metrics reach the same backends.

        :param step: Frames collected at the time of the evaluation.
        :param metrics: Evaluation metrics (e.g. win rate against a baseline).
        """

    # Deliberately concrete and empty, not abstract: adding a required hook
    # would break every existing backend for a signal most of them have no use
    # for, and ignoring a failure is the correct default.
    def on_train_error(self, error: BaseException) -> None:  # noqa: B027
        """
        Called when training is ending because of an unhandled exception.

        Always followed by :meth:`on_train_end`, so this is where a backend
        records *that* the run failed while the summary hook stays the single
        teardown path. Deliberately concrete rather than abstract: a backend
        with no notion of failure needs no code, and the default is to ignore
        it.

        :param error: The exception that ended the run.
        """

    @abstractmethod
    def on_train_end(self, summary: Mapping[str, float]) -> None:
        """
        Called once after the last rollout, including on failure.

        The teardown hook (closing files, finishing runs). Must tolerate being
        called after :meth:`on_train_start` raised or never ran, and after
        :meth:`on_train_error`.

        :param summary: Aggregate statistics for the whole run.
        """


class CallbackList(TrainingCallback):
    """
    Composite that fans each hook out to its members in order.

    Being a :class:`TrainingCallback` itself lets the trainer hold exactly one
    callback and stay unaware of how many are attached.

    Selected callbacks are part of the run contract. A failure propagates so a
    requested metric backend cannot silently stop recording.
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
        Call ``hook`` on every member in order.

        :param hook: Name of the :class:`TrainingCallback` method to invoke.
        :param args: Positional arguments forwarded to the hook.
        """
        for callback in self._callbacks:
            getattr(callback, hook)(*args)

    def on_train_start(self, run_config: Mapping[str, Any]) -> None:
        """
        Forward the run start to every member.

        :param run_config: Opaque run metadata.
        """
        self._dispatch("on_train_start", run_config)

    def on_rollout_start(self, step: int) -> None:
        """
        Forward the rollout start to every member.

        :param step: Total frames collected before this rollout.
        """
        self._dispatch("on_rollout_start", step)

    def on_rollout_end(self, step: int, metrics: Mapping[str, float]) -> None:
        """
        Forward the rollout metrics to every member.

        :param step: Total frames collected so far.
        :param metrics: Running training metrics at ``step``.
        """
        self._dispatch("on_rollout_end", step, metrics)

    def on_eval_end(self, step: int, metrics: Mapping[str, float]) -> None:
        """
        Forward the evaluation metrics to every member.

        :param step: Frames collected at the time of the evaluation.
        :param metrics: Evaluation metrics.
        """
        self._dispatch("on_eval_end", step, metrics)

    def on_train_error(self, error: BaseException) -> None:
        """
        Forward the failure to every member.

        :param error: The exception that ended the run.
        """
        self._dispatch("on_train_error", error)

    def on_train_end(self, summary: Mapping[str, float]) -> None:
        """
        Forward the run summary to every member.

        :param summary: Aggregate statistics for the whole run.
        """
        self._dispatch("on_train_end", summary)
