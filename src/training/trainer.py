import logging
import signal
import time
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
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

# Substrings torchrl uses when a ParallelEnv worker process has died. The
# engine aborts the process outright on some card interactions (a C++
# std::runtime_error crossing the extern "C" boundary calls std::terminate), so
# this is not an exception the environment can catch -- the worker is simply
# gone, and the collector reports it as one of these.
_WORKER_DEATH_SIGNATURES = (
    "worker",
    "At least one process failed",
)


def _is_worker_death(error: BaseException) -> bool:
    """
    Whether a collector error is a dead environment worker rather than a bug.

    :param error: Exception raised while iterating the collector.
    :return: True if the message matches torchrl's worker-death reports.
    """
    message = str(error)
    return any(signature in message for signature in _WORKER_DEATH_SIGNATURES)


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
            max_collector_restarts: int = 10,
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
        :param max_collector_restarts: How many times a dead environment worker
            may be recovered from by rebuilding the collector and carrying on.
            The learner (model, optimizer, curriculum, frame counter) lives in
            this process and is untouched by a worker dying, so the only cost
            is the in-flight batch and a few seconds of fork time. This is a
            budget for the whole run, not per occurrence, so a permanently
            broken environment still fails instead of restarting forever.
            ``0`` restores the previous behaviour of dying on the first one.
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

    def train(self) -> dict[str, float]:
        """
        Run collection until ``total_frames``, updating after every rollout.

        :return: Aggregate statistics: frames, episodes, win/draw rate and fps,
            covering the frames collected before any interruption.
        """
        frames = 0
        episodes = 0
        wins = 0
        draws = 0
        last_eval_frames = 0
        start_time = time.time()
        collector: Collector | None = None
        restarts_left = self._max_collector_restarts
        try:
            self._callbacks.on_train_start(self._run_config)
            with tqdm(total=self._total_frames, unit="frame") as progress_bar:
                while frames < self._total_frames:
                    # Only children this collector starts are ours to reap if it
                    # dies; anything already running (e.g. the W&B service) is not.
                    preexisting = {child.pid for child in torch_mp.active_children()}
                    collector = self._make_collector(self._total_frames - frames)
                    try:
                        for data in collector:
                            self._callbacks.on_rollout_start(frames)
                            assert isinstance(data, TensorDict)
                            batch_frames = data.numel()
                            frames += batch_frames
                            done = cast(Tensor, data["next", "done"]).reshape(-1)
                            final_rewards = cast(Tensor, data["next", "reward"]).reshape(-1)[done]
                            episodes += int(done.sum())
                            wins += int((final_rewards > 0).sum())
                            draws += int((final_rewards == 0).sum())

                            losses = self._update(data)

                            metrics = self._metrics(
                                frames, episodes, wins, draws, time.time() - start_time, losses
                            )
                            progress_bar.update(batch_frames)
                            self._log_progress(progress_bar, metrics)
                            self._callbacks.on_rollout_end(frames, metrics)
                            if self._should_evaluate(frames, last_eval_frames):
                                last_eval_frames = frames
                                assert self._evaluator is not None
                                self._callbacks.on_eval_end(
                                    frames, self._evaluator.evaluate(self._policy)
                                )
                    except RuntimeError as error:
                        if restarts_left <= 0 or not _is_worker_death(error):
                            raise
                        restarts_left -= 1
                        logger.warning(
                            "Environment worker died at %d frames (%s). Discarding the "
                            "in-flight batch and rebuilding the collector; %d restart(s) "
                            "left. The learner's weights, optimizer and frame count are "
                            "unaffected.",
                            frames,
                            error,
                            restarts_left,
                        )
                        self._abandon_collector(collector, preexisting)
                        collector = None
                        continue
                    self._shutdown_collector(collector)
                    collector = None
                    break
        except KeyboardInterrupt:
            logger.warning(
                "Interrupted at %d frames; shutting down and reporting partial results. "
                "Press Ctrl-C again only if shutdown hangs.",
                frames,
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
            summary = self._metrics(frames, episodes, wins, draws, time.time() - start_time)
            self._callbacks.on_train_end(summary)
        return summary

    def _make_collector(self, total_frames: int) -> Collector:
        """
        Build a collector over a freshly constructed vectorized environment.

        Opts out of torchrl's automatic policy-transform registration: env
        transforms are managed explicitly by the env factories, and the policies
        used here read ``action_mask`` directly without needing the InitTracker
        transform the collector's heuristic would append.

        :param total_frames: Frames this collector should still collect, i.e.
            the run's remaining budget rather than its total.
        :return: A collector ready to iterate.
        """
        return Collector(
            create_env_fn=self._make_vec_env(),
            policy=self._policy,
            frames_per_batch=self._frames_per_batch,
            total_frames=total_frames,
            auto_register_policy_transforms=False,
            **self._collector_kwargs(),
        )

    @staticmethod
    def _shutdown_collector(collector: Collector) -> None:
        """
        Shut a collector down, tolerating an already-broken worker pool.

        :param collector: Collector to tear down.
        """
        try:
            collector.shutdown()
        except Exception:
            logger.debug("Collector shutdown raised; continuing.", exc_info=True)

    @staticmethod
    def _abandon_collector(collector: Collector, preexisting_pids: set[int | None]) -> None:
        """
        Tear down a collector whose worker pool has already partly died.

        torchrl's own shutdown path re-checks worker health and raises when it
        finds the dead one, so it cannot be relied on to clean up after exactly
        the failure being recovered from. Any worker it leaves behind is
        terminated here, or the next collector's workers would contend with
        orphans holding the previous run's shared memory.

        :param collector: Collector to abandon.
        :param preexisting_pids: PIDs of child processes that predate this
            collector and must be left alone.
        """
        with _deferred_interrupt():
            Trainer._shutdown_collector(collector)
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
