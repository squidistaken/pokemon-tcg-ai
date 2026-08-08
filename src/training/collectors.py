"""
Collector construction and batch-layout normalization.

TorchRL ships four collectors that differ along two independent axes (how many
copies of the policy exist, and whether the learner waits for every environment
at every step ) and they do not agree on the shape of the batch they hand
back.

======================  ==================  ==================  ================
kind                    policy copies       learner waits?      raw batch shape
======================  ==================  ==================  ================
``sync``                1, in the parent    yes, for all envs   ``(N, T)``
``multi_sync``          one per worker      yes, for all envs   ``(N, T)``
``multi_async``         one per worker      no                  ``(T,)``
``async_batched``       1, in the parent    no                  ``(F,)``
======================  ==================  ==================  ================
"""

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import torch
import torch.multiprocessing as torch_mp
from tensordict import (
    TensorDictBase,
)
from tensordict import (
    cat as cat_tensordicts,
)
from tensordict import (
    stack as stack_tensordicts,
)
from torch import nn
from torchrl.collectors import (
    AsyncBatchedCollector,
    BaseCollector,
    Collector,
    MultiAsyncCollector,
    MultiSyncCollector,
)
from torchrl.envs import EnvBase

logger = logging.getLogger(__name__)

_ENV_INDEX_KEY = "env_index"

#: How far one environment's pending backlog may exceed the shortest before the
#: imbalance is reported.
_BACKLOG_IMBALANCE_FACTOR = 8


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
    :cvar ASYNC_BATCHED: :class:`~torchrl.collectors.AsyncBatchedCollector`.
        One policy copy, in the parent, behind an inference server that batches
        whatever observations have arrived; no barrier. Measured far slower than
        the alternatives on this environment (see
        ``docs/training-performance.md``) because it adds a second IPC hop per
        step, but wired up so that can be re-measured rather than assumed.
    """

    SYNC = "sync"
    MULTI_SYNC = "multi_sync"
    MULTI_ASYNC = "multi_async"
    ASYNC_BATCHED = "async_batched"


def parse_collector_kind(value: str | CollectorKind) -> CollectorKind:
    """
    Resolve a configured collector name.

    :param value: Value of ``collector.type``.
    :return: The matching :class:`CollectorKind`.
    :raises ValueError: If the name is not one of the four kinds. Raised rather
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

    :param max_batch_size: ``async_batched`` only: the largest number of
        observations the inference server folds into one forward pass. Below the
        environment count the server can never batch every environment at once.
    :param min_batch_size: ``async_batched`` only: observations the server
        waits for before dispatching, bounded by ``server_timeout``. ``1``
        dispatches immediately, trading GPU batching for latency.
    :param server_timeout: ``async_batched`` only: seconds the inference
        server waits for more work before running a partial batch.
    :param env_backend: ``async_batched`` only: how the
        :class:`~torchrl.envs.AsyncEnvPool` runs environments.
        ``multiprocessing`` gives each its own process, as the other kinds do;
        ``threading`` keeps them in this process, where the engine's Python-side
        encoding contends for the GIL.
    :param workers_per_batch: ``multi_async`` only: how many single-worker
        rollouts are stacked into one learner batch. ``None`` uses the worker
        count, which reproduces the row count of a synchronous batch.
    """

    max_batch_size: int = 64
    min_batch_size: int = 1
    server_timeout: float = 0.01
    env_backend: str = "multiprocessing"
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
    return kind in (CollectorKind.MULTI_ASYNC, CollectorKind.ASYNC_BATCHED)


#: Substring of the mmap failure ``async_batched`` dies with on this
#: environment. See :func:`_explain_mapping_exhaustion`.
_MAPPING_EXHAUSTION_MARKER = "unable to mmap"


def _explain_mapping_exhaustion(error: BaseException) -> str | None:
    """
    Recognize the transport failure ``async_batched`` hits on this observation.

    :class:`~torchrl.envs.AsyncEnvPool` ships every transition through a
    ``multiprocessing.Queue``, and torch moves each leaf tensor by allocating a
    fresh shared-memory segment that the receiver maps. This environment's
    observation is ~288 leaf tensors, so one transition costs several hundred
    mappings and the receiving process exhausts Linux's ``vm.max_map_count``
    (65530 by default) within a few hundred steps. It surfaces as ``Cannot
    allocate memory`` on an mmap of a few dozen bytes, with memory, shared
    memory and file descriptors all far from exhausted, which reads as anything
    but what it is.

    The batched environments the other kinds use allocate one shared tensordict
    at startup and write into it in place, so they never do per-step mapping and
    never hit this.

    :param error: Exception raised out of collection.
    :return: An explanation to attach, or None if this is a different failure.
    """
    causes: list[BaseException] = []
    seen = error
    while seen is not None and seen not in causes:
        causes.append(seen)
        seen = seen.__cause__ or seen.__context__  # pyright: ignore[reportAssignmentType]
    if not any(_MAPPING_EXHAUSTION_MARKER in str(cause) for cause in causes):
        return None
    return (
        "collector.type=async_batched sends every transition through a queue as "
        "individually shared tensors, and this environment's observation is ~288 "
        "of them per step, which exhausts Linux's vm.max_map_count (65530) in the "
        "receiving process. Use collector.type=multi_sync, which shares one "
        "preallocated tensordict per worker and does no per-step mapping. See "
        "docs/training-performance.md section 6."
    )


class _AssemblingCollector:
    """
    A collector whose batches are reshaped into ``(rows, time)`` on the way out.

    Wraps rather than subclasses a TorchRL collector: the assemblers need to
    buffer across yields (``multi_async`` stacks several rollouts into one
    batch, ``async_batched`` carries a per-environment remainder into the next),
    which a ``postproc`` -- applied to one batch in isolation -- cannot do.

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
        :raises RuntimeError: Re-raised from the wrapped collector, with the
            shared-memory mapping limit named when that is what went wrong.
        """
        try:
            for data in self._collector:
                batch = self._assemble(data)
                if batch is not None:
                    yield batch
        except RuntimeError as error:
            explanation = _explain_mapping_exhaustion(error)
            if explanation is None:
                raise
            raise RuntimeError(explanation) from error

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


class _EnvStreamAssembler:
    """
    Regroup interleaved transitions back into one row per environment.

    :class:`~torchrl.collectors.AsyncBatchedCollector` returns whatever arrived
    in its result queue, so consecutive entries are typically from *different*
    environments; only the per-environment subsequence is a trajectory. This
    splits each batch on :data:`_ENV_INDEX_KEY`, appends to a per-environment
    backlog, and emits a rectangular ``(num_envs, T)`` batch where ``T`` is what
    every environment can currently supply. The remainder stays buffered and
    leads the next batch, so a row continues exactly where the previous one
    ended, with no frame dropped and no trajectory spliced.

    ``T`` is set by the slowest environment, so an emitted batch can be smaller
    than ``frames_per_batch``; the surplus is not lost, it is carried. What is
    lost is whatever is still buffered when the collector shuts down, at most
    one batch's worth over a run.

    :param num_envs: Environments the collector is running.
    """

    def __init__(self, num_envs: int) -> None:
        self._num_envs = num_envs
        self._pending: list[list[TensorDictBase]] = [[] for _ in range(num_envs)]
        self._warned_imbalance = False

    def __call__(self, data: TensorDictBase) -> TensorDictBase | None:
        """
        :param data: One flat batch of interleaved transitions.
        :return: A ``(num_envs, T)`` batch, or None while some environment has
            contributed nothing yet.
        """
        self._append(data)
        lengths = [sum(chunk.numel() for chunk in queue) for queue in self._pending]
        width = min(lengths)
        if width == 0:
            return None
        self._check_imbalance(lengths, width)

        rows: list[TensorDictBase] = []
        for env_index, (queue, length) in enumerate(
            zip(self._pending, lengths, strict=True)
        ):
            merged = queue[0] if len(queue) == 1 else cat_tensordicts(queue, dim=0)
            rows.append(merged[:width])
            # Guarded rather than sliced unconditionally: an empty slice of a
            # lazily stacked tensordict is not an empty tensordict, it raises.
            self._pending[env_index] = [merged[width:]] if length > width else []
        return stack_tensordicts(rows).exclude(_ENV_INDEX_KEY).to_tensordict()

    def _append(self, data: TensorDictBase) -> None:
        """
        Split one raw batch by environment and queue each part in arrival order.

        :param data: One flat batch of interleaved transitions.
        :raises KeyError: If the collector stopped stamping the environment
            index, without which the batch cannot be separated into
            trajectories at all.
        """
        env_indices = data.get(_ENV_INDEX_KEY, default=None)
        if env_indices is None:
            raise KeyError(
                f"AsyncBatchedCollector batch carries no '{_ENV_INDEX_KEY}' entry, so "
                f"its interleaved transitions cannot be separated back into "
                f"per-environment trajectories."
            )
        grouped: list[list[int]] = [[] for _ in range(self._num_envs)]
        for position, env_index in enumerate(_as_int_list(env_indices)):
            grouped[env_index].append(position)
        for env_index, positions in enumerate(grouped):
            if positions:
                self._pending[env_index].append(data[positions])

    def _check_imbalance(self, lengths: Sequence[int], width: int) -> None:
        """
        Warn once if one environment is holding the whole batch back.

        :param lengths: Frames currently buffered per environment.
        :param width: Frames every environment can supply, i.e. ``min(lengths)``.
        """
        if self._warned_imbalance or max(lengths) <= width * _BACKLOG_IMBALANCE_FACTOR:
            return
        self._warned_imbalance = True
        logger.warning(
            "Async collection is badly imbalanced: the slowest environment has "
            "supplied %d frames while the fastest has %d buffered. Batches are "
            "sized by the slowest, so memory grows with the gap.",
            width,
            max(lengths),
        )


def _as_int_list(env_indices: Any) -> list[int]:
    """
    Read the per-transition environment index out of whatever it is stored as.

    The collector sets a plain Python int per transition, which stacking turns
    into a non-tensor stack rather than a tensor, so neither ``.tolist()`` nor
    tensor indexing can be assumed.

    :param env_indices: The stacked ``env_index`` entry.
    :return: One environment index per transition, in batch order.
    """
    if isinstance(env_indices, torch.Tensor):
        return [int(value) for value in env_indices.reshape(-1).tolist()]
    return [int(value) for value in env_indices.tolist()]


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
        ``policy_device`` and ``compile_policy``. Silently ignored by
        ``async_batched``, which takes neither.
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

    if kind is CollectorKind.MULTI_ASYNC:
        _check_per_worker_cuda(
            kind, collector_kwargs, len(env_factories), mp_start_method
        )
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
                # Per worker for this collector, not per batch: each worker
                # collects this many frames and yields them on its own.
                frames_per_batch=per_worker,
                total_frames=total_frames,
                auto_register_policy_transforms=False,
                **collector_kwargs,
            ),
            _WorkerRolloutAssembler(rows),
        )

    # AsyncBatchedCollector runs the learner's own policy module inside an
    # in-process inference server, so it takes neither policy_device (the server
    # has its own `device`) nor the collector-level compile flag.
    device = collector_kwargs.get("policy_device")
    return _AssemblingCollector(
        AsyncBatchedCollector(
            create_env_fn=env_factories,
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            max_batch_size=options.max_batch_size,
            min_batch_size=options.min_batch_size,
            server_timeout=options.server_timeout,
            env_backend=options.env_backend,  # pyright: ignore[reportArgumentType]
            device=device,
        ),
        _EnvStreamAssembler(len(env_factories)),
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
