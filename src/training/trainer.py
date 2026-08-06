import logging
import signal
import time
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import torch.multiprocessing as torch_mp
from tensordict import TensorDict
from torch import Tensor, nn
from torchrl.collectors import Collector
from torchrl.envs import EnvBase, ParallelEnv, SerialEnv
from tqdm import tqdm

from src.training.base_trainer import BaseTrainer
from src.training.callbacks import CallbackList, TrainingCallback
from src.training.evaluator import Evaluator
from src.training.multi_evaluator import MultiEvaluator

logger = logging.getLogger(__name__)

# Metrics every iteration reports.
# Anything else in a metrics mapping comes from the algorithm's _update and is
# formatted generically.
_CORE_METRICS = ("frames", "episodes", "win_rate", "draw_rate", "fps")

#: Lowercase substrings identifying torchrl's report that a ParallelEnv worker
#: process is gone. The engine aborts the process outright rather than raising
#: (a C++ exception thrown inside the engine unwinds through cabt's `Select`
#: entry point, which has no handler, into ctypes frames that carry no unwind
#: tables, so it reaches std::terminate), so the parent only ever learns about
#: it as a dead pipe. That surfaces as a bare RuntimeError, hence the string
#: match: torchrl raises no dedicated exception type for it.
_WORKER_DEATH_MARKERS = (
    "at least one process failed",
    "cannot proceed, worker",
)

#: Consecutive restarts that collect nothing before dying again, after which
#: the loop stops respawning. A pool that cannot complete a single batch is
#: failing for a reason a restart will not fix (a bad checkpoint, an
#: unsatisfiable config), and retrying it forever would burn the wall clock
#: while looking like a live run.
_MAX_BARREN_RESTARTS = 3


def _is_worker_death(error: BaseException) -> bool:
    """
    Whether an exception raised out of collection means a worker process died.

    :param error: Exception raised while iterating the collector.
    :return: True if this is a dead worker rather than a fault in the update.
    """
    if isinstance(error, EOFError | BrokenPipeError | ConnectionResetError):
        # The parent hit the far end of a worker's pipe directly, before
        # torchrl's own liveness check ran.
        return True
    message = str(error).lower()
    return any(marker in message for marker in _WORKER_DEATH_MARKERS)


@dataclass
class _RunTotals:
    """
    Running totals for one training run, carried across collector restarts.

    Held in the parent process, so a dead worker pool costs the run only its
    in-flight batch: the counters, the policy, the optimizer and the curriculum
    all survive untouched.

    :param frames: Frames collected so far.
    :param episodes: Episodes finished so far.
    :param wins: Episodes won so far.
    :param draws: Episodes drawn so far.
    :param last_eval_frames: Frame count at the most recent evaluation.
    """

    frames: int = 0
    episodes: int = 0
    wins: int = 0
    draws: int = 0
    last_eval_frames: int = 0


@contextmanager
def _deferred_interrupt() -> Generator[None, None, None]:
    """
    Ignore SIGINT for the duration of the block, restoring the handler after.

    Shutdown must not itself be interruptible. A second Ctrl-C arriving while
    the collector is tearing down leaves the ParallelEnv workers orphaned and
    their semaphores unreleased, which is precisely the mess the first Ctrl-C
    was trying to avoid.

    :return: Context manager yielding nothing.
    """
    try:
        previous_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    except ValueError:
        # Signal handlers can only be installed from the main thread; off it,
        # cleanup simply runs unprotected rather than failing.
        yield
        return
    try:
        yield
    finally:
        signal.signal(
            signal.SIGINT,
            previous_handler if previous_handler is not None else signal.default_int_handler,
        )


class Trainer(BaseTrainer):
    """
    Training scaffold around torchrl's Collector.

    Owns the vectorized environment, the collector lifecycle and progress
    statistics. The learning algorithm lives in :meth:`_update`, which is a
    no-op in this base class, so running it directly gives a pure collection
    loop (e.g. the random-policy baseline). The  PPO trainer
    subclasses this, passes the actor as ``policy`` and implements
    :meth:`_update` with the advantage/loss/optimizer step.
    """

    def __init__(
            self,
            env_factories: list[Callable[[], EnvBase]],
            policy: nn.Module,
            frames_per_batch: int,
            total_frames: int,
            use_parallel_env: bool = True,
            mp_start_method: str = "fork",
            serial_for_single: bool = True,
            callbacks: Iterable[TrainingCallback] | None = None,
            run_config: Mapping[str, Any] | None = None,
            evaluator: Evaluator | MultiEvaluator | None = None,
            eval_interval: int = 0,
            max_collector_restarts: int = 0,
            rebuild_env_factories: Callable[[int], list[Callable[[], EnvBase]]] | None = None,
    ) -> None:
        """
        :param env_factories: One environment factory per worker.
        :param policy: Collection policy; any tensordict module writing "action".
        :param frames_per_batch: Frames collected per collector iteration.
        :param total_frames: Total frames (e.g. timesteps) to collect over the run.
        :param use_parallel_env: Use multiprocess ParallelEnv instead of SerialEnv.
        :param mp_start_method: Multiprocessing start method for ParallelEnv workers.
        :param serial_for_single: Fall back to a single-process env when there is
            only one worker, instead of paying ParallelEnv's process overhead.
        :param callbacks: Observers notified of run start, every rollout and run
            end. None attaches nothing, leaving console logging as the only sink.
        :param run_config: Opaque run metadata (in practice the resolved Hydra
            config) forwarded verbatim to ``on_train_start``; never read here.
        :param evaluator: Scores the policy against one fixed reference opponent
            (:class:`~src.training.evaluator.Evaluator`) or several
            (:class:`~src.training.multi_evaluator.MultiEvaluator`) every
            ``eval_interval`` frames, reporting through ``on_eval_end``. None
            skips evaluation entirely. Needed under self-play, where the
            collected win-rate is pinned near 0.5 by construction.
        :param eval_interval: Frames between evaluations; ``0`` disables them
            even when an evaluator is supplied.
        :param max_collector_restarts: How many times a dead worker pool may be
            rebuilt and collection resumed, rather than ending the run. ``0``
            (the default) propagates the failure as before. See :meth:`train`
            for what a restart costs.
        :param rebuild_env_factories: Builds a replacement set of factories for
            restart *n* (1-based), so a restarted pool does not replay the
            per-worker seed stream the dead one started from. None reuses the
            original factories, which is fine for a stochastic policy but
            re-deals the same deck/seat sequence. Unused when
            ``max_collector_restarts`` is 0.
        """
        self._env_factories = env_factories
        self._policy = policy
        self._frames_per_batch = frames_per_batch
        self._total_frames = total_frames
        self._use_parallel_env = use_parallel_env
        self._mp_start_method = mp_start_method
        self._serial_for_single = serial_for_single
        self._callbacks = CallbackList(callbacks or ())
        self._run_config = run_config if run_config is not None else {}
        self._evaluator = evaluator
        self._eval_interval = eval_interval
        self._max_collector_restarts = max_collector_restarts
        self._rebuild_env_factories = rebuild_env_factories
        # The Collector rounds its budget up to a whole number of batches, so
        # anchoring the loop to the same rounded figure keeps the remaining
        # frames handed to a restarted Collector exactly divisible. Passing the
        # raw remainder instead would re-trigger torchrl's not-divisible warning
        # on every restart and drift the run's total frame count.
        batches = -(-total_frames // frames_per_batch)
        self._budget = batches * frames_per_batch

    def train(self) -> dict[str, float]:
        """
        Run collection until ``total_frames``, updating after every rollout.

        When ``max_collector_restarts`` allows it, a worker process that dies
        mid-rollout is treated as a recoverable fault rather than the end of the
        run: the pool is torn down, a fresh one is built and collection resumes
        from the frame count reached so far. This is what it takes to finish a
        long run unattended against an engine that aborts its process on rare
        game states instead of raising (see :data:`_WORKER_DEATH_MARKERS`).

        A restart is cheap but not free. Everything the learner owns lives in
        this process and survives untouched -- weights, optimizer state, the
        curriculum's level buffer, snapshot cadence, the frame counter. What is
        lost is per-worker: the batch in flight, every battle in progress, and
        the workers' PFSP win-rate tallies, which are estimated per worker and
        so restart from their prior.

        :return: Aggregate statistics: frames, episodes, win/draw rate and fps,
            covering the frames collected before any interruption.
        """
        totals = _RunTotals()
        start_time = time.time()
        restarts = 0
        barren_restarts = 0
        collector: Collector | None = None
        try:
            self._callbacks.on_train_start(self._run_config)
            with tqdm(total=self._total_frames, unit="frame") as progress_bar:
                while totals.frames < self._budget:
                    if restarts:
                        self._prepare_restart(restarts)
                    # Only children this collector starts are ours to reap if it
                    # dies; anything already running (e.g. the W&B service) is not.
                    preexisting = {child.pid for child in torch_mp.active_children()}
                    collector = self._make_collector(self._budget - totals.frames)
                    frames_before = totals.frames
                    try:
                        self._collect(collector, progress_bar, totals, start_time)
                    except (RuntimeError, OSError, EOFError) as error:
                        if not _is_worker_death(error):
                            raise
                        if restarts >= self._max_collector_restarts:
                            logger.error(
                                "Worker pool died at %d frames after %d restart(s); "
                                "the restart budget (collector.max_restarts) is spent.",
                                totals.frames,
                                restarts,
                            )
                            raise
                        barren_restarts = (
                            barren_restarts + 1 if totals.frames == frames_before else 0
                        )
                        if barren_restarts >= _MAX_BARREN_RESTARTS:
                            logger.error(
                                "Worker pool died %d times in a row without collecting "
                                "a batch; not restarting again.",
                                barren_restarts,
                            )
                            raise
                        restarts += 1
                        logger.warning(
                            "Worker pool died at %d frames (%s). Restarting collection "
                            "(%d of %d); the in-flight batch and all battles in "
                            "progress are discarded.",
                            totals.frames,
                            error,
                            restarts,
                            self._max_collector_restarts,
                        )
                    else:
                        break
                    finally:
                        self._shutdown_collector(collector, preexisting)
                        collector = None
        except KeyboardInterrupt:
            logger.warning(
                "Interrupted at %d frames; shutting down and reporting partial results. "
                "Press Ctrl-C again only if shutdown hangs.",
                totals.frames,
            )
        except BaseException as error:
            # Recorded before the teardown below so metric backends can mark the
            # run failed. Without this a crashed run closes as cleanly as a
            # finished one and is indistinguishable from a short successful run.
            self._callbacks.on_train_error(error)
            raise
        finally:
            with _deferred_interrupt():
                if self._evaluator is not None:
                    self._evaluator.close()
            if collector is not None:
                self._shutdown_collector(collector)
            summary = self._metrics(
                totals.frames,
                totals.episodes,
                totals.wins,
                totals.draws,
                time.time() - start_time,
            )
            if restarts:
                logger.warning(
                    "Run completed across %d collector restart(s).", restarts
                )
            self._callbacks.on_train_end(summary)
        return summary

    def _collect(
            self,
            collector: Collector,
            progress_bar: tqdm,
            totals: _RunTotals,
            start_time: float,
    ) -> None:
        """
        Drain one collector, updating and reporting after every rollout.

        Returns when the collector's own budget is exhausted; raises whatever
        collection or the update raised, which :meth:`train` classifies into
        recoverable worker death and everything else.

        :param collector: Collector to iterate; owns one worker pool.
        :param progress_bar: Bar tracking frames across the whole run, not just
            this collector.
        :param totals: Run totals, advanced in place so they survive a restart.
        :param start_time: Wall-clock start of the run, for the fps figure.
        """
        for data in collector:
            self._callbacks.on_rollout_start(totals.frames)
            assert isinstance(data, TensorDict)
            batch_frames = data.numel()
            totals.frames += batch_frames
            done = cast(Tensor, data["next", "done"]).reshape(-1)
            final_rewards = cast(Tensor, data["next", "reward"]).reshape(-1)[done]
            totals.episodes += int(done.sum())
            totals.wins += int((final_rewards > 0).sum())
            totals.draws += int((final_rewards == 0).sum())

            # Perform update step (return surrgate loss)
            losses = self._update(data)

            metrics = self._metrics(
                totals.frames,
                totals.episodes,
                totals.wins,
                totals.draws,
                time.time() - start_time,
                losses,
            )
            progress_bar.update(batch_frames)
            self._log_progress(progress_bar, metrics)
            self._callbacks.on_rollout_end(totals.frames, metrics)
            if self._should_evaluate(totals.frames, totals.last_eval_frames):
                totals.last_eval_frames = totals.frames
                assert self._evaluator is not None
                self._callbacks.on_eval_end(
                    totals.frames, self._evaluator.evaluate(self._policy)
                )

    def _make_collector(self, remaining_frames: int) -> Collector:
        """
        Build a collector over a fresh worker pool for the frames still owed.

        :param remaining_frames: Frames this collector should produce, i.e. the
            run budget less what previous collectors already delivered.
        :return: A Collector wrapping a newly built vectorized environment.
        """
        # Opt out of torchrl's automatic policy-transform registration: env
        # transforms are managed explicitly by the env factories, and the
        # policies used here read "action_mask" directly without needing the
        # InitTracker transform the collector's heuristic would append.
        return Collector(
            create_env_fn=self._make_vec_env(),
            policy=self._policy,
            frames_per_batch=self._frames_per_batch,
            total_frames=remaining_frames,
            auto_register_policy_transforms=False,
            **self._collector_kwargs(),
        )

    def _prepare_restart(self, restart_index: int) -> None:
        """
        Re-establish per-worker state before a replacement pool is built.

        The base class only re-deals the environment factories. Subclasses
        extend this for state that is keyed to collector rows and would
        otherwise be misattributed to whichever episode lands in that row next.

        :param restart_index: 1-based index of the restart about to happen.
        """
        if self._rebuild_env_factories is not None:
            self._env_factories = self._rebuild_env_factories(restart_index)

    @staticmethod
    def _shutdown_collector(
            collector: Collector,
            preexisting_pids: set[int | None] | None = None,
    ) -> None:
        """
        Tear a collector down without letting cleanup mask the original failure.

        ``shutdown`` re-runs torchrl's liveness check, so tearing down a pool
        whose worker has already died raises the very condition that brought us
        here. Running from a ``finally``, that replacement exception would
        discard the specific one ("worker 6 dead") in favour of the generic one,
        which is exactly how the failure this guards against reports itself.

        Because that shutdown cannot be relied on to finish, it is also not
        guaranteed to reap the pool's surviving workers. Any it leaves behind
        are terminated here when ``preexisting_pids`` says which children the
        collector owns, or the next collector's workers would contend with
        orphans still holding the previous pool's shared memory.

        :param collector: Collector to shut down; may already be broken.
        :param preexisting_pids: PIDs of child processes that predate this
            collector and must be left alone (the W&B service, say). None skips
            the reaping entirely, for callers that never started a pool of
            their own.
        """
        with _deferred_interrupt():
            try:
                collector.shutdown()
            except Exception:
                logger.warning(
                    "Collector shutdown failed; continuing.", exc_info=True
                )
            if preexisting_pids is None:
                return
            for child in torch_mp.active_children():
                if child.pid in preexisting_pids:
                    continue
                child.terminate()
                child.join(timeout=10.0)
                if child.is_alive():
                    child.kill()
                    child.join(timeout=10.0)

    def _should_evaluate(self, frames: int, last_eval_frames: int) -> bool:
        """
        Whether an evaluation round is due after the rollout just finished.

        :param frames: Total frames collected so far.
        :param last_eval_frames: Frame count at the previous evaluation.
        :return: True if an evaluator is attached and the interval has elapsed.
        """
        return (
            self._evaluator is not None
            and self._eval_interval > 0
            and frames - last_eval_frames >= self._eval_interval
        )

    @staticmethod
    def _metrics(
            frames: int,
            episodes: int,
            wins: int,
            draws: int,
            elapsed: float,
            losses: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Build the metrics mapping for the run so far.

        :param frames: Total frames collected so far.
        :param episodes: Total episodes finished so far.
        :param wins: Total wins so far.
        :param draws: Total draws so far.
        :param elapsed: Wall-clock seconds since training started.
        :param losses: Loss values from the last update, merged in if present.
        :return: Metrics keyed by :data:`_CORE_METRICS` plus any loss keys.
        """
        metrics: dict[str, float] = {
            "frames": frames,
            "episodes": episodes,
            "win_rate": wins / max(episodes, 1),
            "draw_rate": draws / max(episodes, 1),
            # Guarded because a fast first batch can land inside the clock's
            # resolution, making elapsed 0.
            "fps": frames / max(elapsed, 1e-9),
        }
        if losses:
            metrics.update(losses)
        return metrics

    # Instance method (not static) so subclasses can override with instance
    # state; the base returns no extra kwargs, leaving collection unchanged.
    # noinspection PyMethodMayBeStatic
    def _collector_kwargs(self) -> dict:  # noqa: PLR6301
        """
        Extra keyword arguments to pass to the :class:`Collector`.

        Empty in the base class, so the random-collection baseline builds the
        collector exactly as before. Subclasses override this to enable
        collector-level features (e.g. PPO sets ``compile_policy``).

        :return: Mapping splatted into the ``Collector(...)`` construction.
        """
        return {}

    # Deliberately an instance method with an unused `data` argument: this is the
    # no-op base-class hook that subclasses (PPO) override with this exact signature,
    # so "could be static" and "unused argument" do not apply.
    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def _update(self, data: TensorDict) -> dict[str, float] | None:  # noqa: ARG002, PLR6301
        """
        Run the algorithm-specific update on a collected batch.

        :param data: One batch of ``frames_per_batch`` transitions from the
            Collector, as a TensorDict shaped ``(B, T)`` where ``B`` is
            ``num_workers`` and ``T`` is ``frames_per_batch // num_workers``
            (2D: one row per vectorized worker, not flattened). Every entry
            is one agent-side step of :class:`~src.env.tcg_env.TCGEnv` (a
            single ``Categorical`` pick); opponent moves are played inside
            the environment and never appear as separate entries. Layout,
            with ``n_actions = max_options + 1``::

                data                                    (state the policy acted on)
                |-- "observation"   (B, T, ...)        nested   structured state, see
                |                                               docs/torchrl_environment.md
                |-- "action_mask"   (B, T, n_actions) bool     legal actions there
                |-- "action"        (B, T)            int64    index the policy picked
                |-- "done"          (B, T, 1)          bool     incoming done/terminated/
                |-- "terminated"    (B, T, 1)          bool     truncated, carried over from
                |-- "truncated"     (B, T, 1)          bool     the previous step's "next"
                `-- "next"                                     (state after the action)
                    |-- "observation"  (B, T, ...)        nested
                    |-- "action_mask"  (B, T, n_actions) bool
                    |-- "reward"       (B, T, 1)          float32  terminal-only, see docs
                    |-- "done"         (B, T, 1)          bool     this step's own done/
                    |-- "terminated"   (B, T, 1)          bool     terminated/truncated
                    `-- "truncated"    (B, T, 1)          bool

            A PPO ``_update`` would read ``data["action"]``, ``data["next",
            "reward"]`` and the done flags to compute advantages, and would
            additionally need ``"sample_log_prob"`` from the actor (not
            produced by the current random policy).
        :return: Loss values to log, or None when no update was performed.
        """
        return None

    def _make_vec_env(self) -> EnvBase:
        """
        Build the vectorized environment from the injected factories.

        The process-wide start method is forced to match before construction.
        Importing torchrl sets it to ``spawn``, and ``ParallelEnv`` starts its
        workers lazily on first use, so the constructor argument alone does not
        decide how they are ultimately created. Under ``spawn`` anything the
        factories close over is pickled, which silently gives every worker a
        private copy of what was meant to be shared memory -- the level
        curriculum's distribution channel in particular, which then freezes at
        whatever was published before collection began.

        :return: ParallelEnv (fork workers) or SerialEnv over the factories.
        """
        if self._use_parallel_env:
            torch_mp.set_start_method(self._mp_start_method, force=True)
            return ParallelEnv(
                num_workers=len(self._env_factories),
                create_env_fn=self._env_factories,
                mp_start_method=self._mp_start_method,
                serial_for_single=self._serial_for_single,
            )
        return SerialEnv(num_workers=len(self._env_factories),
                         create_env_fn=self._env_factories)

    @staticmethod
    def _log_progress(progress_bar: tqdm, metrics: Mapping[str, float]) -> None:
        """
        Show the running training metrics in the progress bar's postfix.

        All routine per-iteration data lives in the bar itself rather than in
        printed lines, so the terminal shows a single live bar instead of one
        metrics line per iteration. Frames and fps are omitted from the
        postfix because the bar's counter and rate already display them.
        Keys outside :data:`_CORE_METRICS` (the loss terms) are appended
        generically, so a new loss term needs no change here.

        :param progress_bar: tqdm bar tracking collected frames.
        :param metrics: Mapping from :meth:`_metrics`.
        """
        postfix: dict[str, str] = {
            "episodes": str(int(metrics["episodes"])),
            "win_rate": f"{metrics['win_rate']:.3f}",
            "draw_rate": f"{metrics['draw_rate']:.3f}",
        }
        postfix.update(
            {
                key: f"{value:.4f}"
                for key, value in metrics.items()
                if key not in _CORE_METRICS
            }
        )
        progress_bar.set_postfix(postfix)
