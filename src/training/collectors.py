"""
Collector construction and batch-layout normalization.

The collectors used here differ along two independent axes (how many copies of
the policy exist, and whether the learner waits for every environment at every
step) and they do not agree on the shape of the batch they hand back.

======================  ==================  ==================  ================
kind                    policy copies       learner waits?      raw batch shape
======================  ==================  ==================  ================
``sync``                1, in the parent    yes, for all envs   ``(N, T)``
``multi_sync``          one per worker      yes, for all envs   ``(N, T)``
``multi_async``         one per worker      no                  ``(T,)``
======================  ==================  ==================  ================

TorchRL's ``AsyncBatchedCollector`` is deliberately absent: its transport maps a
fresh shared-memory segment per leaf tensor per transition, and this
environment's ~288-tensor observation exhausts Linux's ``vm.max_map_count``
within a few hundred steps, so it cannot collect a batch at all
(``docs/training-performance.md`` section 6.6).
"""

import logging
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import torch
import torch.multiprocessing as torch_mp
from tensordict import (
    TensorDictBase,
)
from tensordict import (
    stack as stack_tensordicts,
)
from torch import nn
from torchrl.collectors import (
    BaseCollector,
    Collector,
    MultiAsyncCollector,
    MultiSyncCollector,
)
from torchrl.envs import EnvBase

logger = logging.getLogger(__name__)


class CollectorKind(StrEnum):
    """
    Which TorchRL collector drives data collection.

    :cvar SYNC: One :class:`~torchrl.collectors.Collector` over a
        :class:`~torchrl.envs.ParallelEnv`. The policy lives in the parent
        process and every environment waits for the slowest one at every step.
        The historical default, and the only fully on-policy option.
    :cvar MULTI_SYNC: :class:`~torchrl.collectors.MultiSyncCollector`. One
        worker process per environment, each running its own copy of the policy,
        so the per-step round trip through the parent disappears; the learner
        still waits for every worker before it gets a batch, which keeps the
        batch on-policy.
    :cvar MULTI_ASYNC: :class:`~torchrl.collectors.MultiAsyncCollector`. As
        above, but the learner takes whichever worker finished first and the
        others keep collecting through the update. Data is stale by up to a few
        updates, which is what V-trace exists to correct (see
        ``agent.value_estimator``).
    """

    SYNC = "sync"
    MULTI_SYNC = "multi_sync"
    MULTI_ASYNC = "multi_async"


def parse_collector_kind(value: str | CollectorKind) -> CollectorKind:
    """
    Resolve a configured collector name.

    :param value: Value of ``collector.type``.
    :return: The matching :class:`CollectorKind`.
    :raises ValueError: If the name is not one of the known kinds. Raised rather
        than defaulted, because falling back to ``sync`` would silently run a
        throughput experiment against the very baseline it is measuring.
    """
    try:
        return CollectorKind(str(value))
    except ValueError:
        known = ", ".join(kind.value for kind in CollectorKind)
        raise ValueError(
            f"Unknown collector.type '{value}'; expected one of: {known}."
        ) from None


@dataclass(frozen=True)
class AsyncCollectorOptions:
    """
    Knobs that only the asynchronous collectors read.

    :param workers_per_batch: ``multi_async`` only: how many single-worker
        rollouts are stacked into one learner batch. ``None`` uses the worker
        count, which reproduces the row count of a synchronous batch.
    """

    workers_per_batch: int | None = None


class TrainingCollector(Protocol):
    def __iter__(self) -> Iterator[TensorDictBase]:
        """:return: Iterator over ``(rows, time)`` batches."""
        ...

    def shutdown(self) -> None:
        """Tear down the worker pool."""
        ...

    def update_policy_weights_(self) -> None:
        """Push the learner's current weights to wherever inference happens."""
        ...


def requires_weight_sync(kind: CollectorKind) -> bool:
    """
    Whether the learner must push its weights to the collector after an update.

    True exactly when inference runs against a *copy* of the policy in another
    process.

    :param kind: Collector kind in use.
    :return: True if :meth:`update_policy_weights_` must be called per batch.
    """
    return kind in (CollectorKind.MULTI_SYNC, CollectorKind.MULTI_ASYNC)


def is_off_policy(kind: CollectorKind) -> bool:
    """
    Whether a batch can contain actions drawn from more than one set of weights.

    :param kind: Collector kind in use.
    :return: True if collection overlaps the update.
    """
    return kind is CollectorKind.MULTI_ASYNC


class _AssemblingCollector:
    """
    A collector whose batches are reshaped into ``(rows, time)`` on the way out.

    Wraps rather than subclasses a TorchRL collector: the assembler needs to
    buffer across yields (``multi_async`` stacks several single-worker rollouts
    into one batch), which a ``postproc`` -- applied to one batch in isolation --
    cannot do.

    :param collector: The underlying TorchRL collector.
    :param assemble: Maps one raw batch to a ``(rows, time)`` batch, or to None
        when it has not accumulated enough to emit one yet.
    """

    def __init__(
        self,
        collector: BaseCollector,
        assemble: Callable[[TensorDictBase], TensorDictBase | None],
    ) -> None:
        self._collector = collector
        self._assemble = assemble

    def __iter__(self) -> Iterator[TensorDictBase]:
        """
        :return: Iterator over assembled ``(rows, time)`` batches.
        """
        for data in self._collector:
            batch = self._assemble(data)
            if batch is not None:
                yield batch

    def shutdown(self) -> None:
        """Tear down the wrapped collector; any buffered remainder is dropped."""
        self._collector.shutdown()

    def update_policy_weights_(self) -> None:
        """Forward a weight push to the wrapped collector."""
        self._collector.update_policy_weights_()


class _WorkerRolloutAssembler:
    """
    Stack consecutive single-worker rollouts into one ``(rows, time)`` batch.

    :class:`~torchrl.collectors.MultiAsyncCollector` yields one worker's rollout
    at a time -- ``frames_per_batch`` frames from a *single* environment, flat.
    Handing that to PPO directly would make every batch one environment's
    experience, so the collector is built with a per-worker budget of
    ``frames_per_batch // rows`` and this assembler stacks ``rows`` of them back
    into a batch of the requested size.

    Each row is contiguous in time within one environment, which is all GAE
    needs. Rows are not pinned to workers: a fast worker can contribute two
    rows to the same batch, and those two are consecutive slices of one stream
    treated as two truncated fragments -- the same approximation the synchronous
    collector already makes at every batch boundary.

    :param rows: Rollouts stacked per emitted batch.
    """

    def __init__(self, rows: int) -> None:
        self._rows = rows
        self._pending: list[TensorDictBase] = []

    def __call__(self, data: TensorDictBase) -> TensorDictBase | None:
        """
        :param data: One worker's flat rollout.
        :return: The stacked batch once ``rows`` have arrived, else None.
        """
        self._pending.append(data.reshape(-1))
        if len(self._pending) < self._rows:
            return None
        batch = stack_tensordicts(self._pending).to_tensordict()
        self._pending.clear()
        return batch


def build_collector(
    kind: CollectorKind,
    *,
    env_factories: list[Callable[[], EnvBase]],
    make_vec_env: Callable[[], EnvBase],
    policy: nn.Module,
    frames_per_batch: int,
    total_frames: int,
    collector_kwargs: Mapping[str, Any],
    options: AsyncCollectorOptions,
    mp_start_method: str = "fork",
) -> TrainingCollector:
    """
    Build the collector named by ``kind``, normalized to ``(rows, time)`` batches.

    :param kind: Which collector to build.
    :param env_factories: One environment factory per worker. Used directly by
        every kind except ``sync``, which needs them assembled into a single
        batched environment first.
    :param make_vec_env: Builds that batched environment; called only for
        ``sync``, so the other kinds do not pay for a pool they never use.
    :param policy: Collection policy.
    :param frames_per_batch: Frames per batch handed to the learner. For
        ``multi_async`` this is divided across the rollouts stacked into one
        batch rather than passed through, since that collector's own
        ``frames_per_batch`` is per worker.
    :param total_frames: Frames this collector should produce.
    :param collector_kwargs: Extra arguments from the trainer, e.g.
        ``policy_device`` and ``compile_policy``.
    :param options: Settings only the asynchronous kinds read.
    :param mp_start_method: Start method forced process-wide before any kind
        that starts its own workers is built. See :func:`_force_start_method`.
    :return: A collector yielding ``(rows, time)`` batches.
    """
    if kind is not CollectorKind.SYNC:
        _force_start_method(mp_start_method)

    if kind is CollectorKind.SYNC:
        # Opt out of torchrl's automatic policy-transform registration: env
        # transforms are managed explicitly by the env factories, and the
        # policies used here read "action_mask" directly without needing the
        # InitTracker transform the collector's heuristic would append.
        return Collector(
            create_env_fn=make_vec_env(),
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            auto_register_policy_transforms=False,
            **collector_kwargs,
        )

    if kind is CollectorKind.MULTI_SYNC:
        _check_per_worker_cuda(
            kind, collector_kwargs, len(env_factories), mp_start_method
        )
        return MultiSyncCollector(
            create_env_fn=env_factories,
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            auto_register_policy_transforms=False,
            **collector_kwargs,
        )

    _check_per_worker_cuda(kind, collector_kwargs, len(env_factories), mp_start_method)
    rows = options.workers_per_batch or len(env_factories)
    per_worker = max(1, frames_per_batch // rows)
    if per_worker * rows != frames_per_batch:
        logger.warning(
            "frames_per_batch %d is not divisible by %d rollouts per batch; "
            "batches will hold %d frames.",
            frames_per_batch,
            rows,
            per_worker * rows,
        )
    return _AssemblingCollector(
        MultiAsyncCollector(
            create_env_fn=env_factories,
            policy=policy,
            # Per worker for this collector, not per batch: each worker collects
            # this many frames and yields them on its own.
            frames_per_batch=per_worker,
            total_frames=total_frames,
            auto_register_policy_transforms=False,
            **collector_kwargs,
        ),
        _WorkerRolloutAssembler(rows),
    )


def _force_start_method(mp_start_method: str) -> None:
    """
    Force the process-wide multiprocessing start method before workers start.

    This lives here, rather than only in the trainer that calls this module,
    because it is a correctness requirement of the code that *starts the
    processes*, and leaving it with the caller means every future caller has to
    know to repeat it. TorchRL's multiprocess collectors resolve their context
    from the global start method and fall back to ``spawn`` when it is unset
    (``torchrl._utils._get_default_mp_start_method``), and importing torchrl is
    itself enough to leave it unset.

    Under ``spawn`` every worker re-imports the world and everything the env
    factories close over is pickled, which silently gives each worker a private
    copy of what was meant to be shared memory -- the level curriculum's
    distribution channel in particular, which then freezes at whatever was
    published before collection began. It is also what turned a 4-second startup
    into 52 seconds (``docs/training-performance.md`` section 5).

    :param mp_start_method: Start method to force, normally ``fork``.
    """
    torch_mp.set_start_method(mp_start_method, force=True)


def _check_per_worker_cuda(
    kind: CollectorKind,
    collector_kwargs: Mapping[str, Any],
    num_workers: int,
    mp_start_method: str,
) -> None:
    """
    Reject a CUDA collection policy under the multiprocess collectors.

    These kinds put a copy of the policy in every worker. Under ``fork`` that is
    not merely expensive, it cannot work at all: CUDA refuses to initialize in a
    forked child, so the run dies with ``Cannot re-initialize CUDA in forked
    subprocess`` from inside torch's own IPC machinery, several frames below
    anything that names the collector. Raising here turns that into a message
    that says which setting to change.

    Under a start method that can carry CUDA the configuration is legal but
    still unwise -- one context per worker, hundreds of megabytes each, to run
    forward passes of batch size one, which is the case a GPU is worst at -- so
    it warns instead.

    The device is neither overridden nor silently split: the trainer's optimizer
    lives on ``agent.device``, and quietly moving collection off it here would
    trade a clear failure for a confusing one.

    :param kind: Collector kind being built.
    :param collector_kwargs: Trainer-supplied collector arguments.
    :param num_workers: Worker processes about to be started.
    :param mp_start_method: Start method the workers will be created with.
    :raises ValueError: If a CUDA collection policy is combined with ``fork``.
    """
    device = collector_kwargs.get("policy_device")
    if device is None or torch.device(device).type != "cuda":
        return
    if mp_start_method == "fork":
        raise ValueError(
            f"collector.type={kind.value} runs one policy copy per worker process, "
            f"and those are forked, so a CUDA collection policy cannot work: CUDA "
            f"refuses to initialize in a forked child. Set agent.collector_device=cpu "
            f"to run collection on the CPU while the PPO update keeps the GPU. "
            f"Setting agent.device=cpu instead would drag the update onto the CPU "
            f"too, which costs far more than collection gains."
        )
    logger.warning(
        "collector.type=%s runs one policy copy per worker, so this will open %d "
        "CUDA contexts for batch-size-1 forwards. Set agent.collector_device=cpu "
        "to move collection off the GPU while the update keeps it; setting "
        "agent.device=cpu instead would drag the update onto the CPU too.",
        kind.value,
        num_workers,
    )
