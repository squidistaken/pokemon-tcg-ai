import logging
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast

from tensordict import TensorDict
from torch import Tensor, nn
from torchrl.collectors import Collector
from torchrl.envs import EnvBase, ParallelEnv, SerialEnv
from tqdm import tqdm

from src.training.base_trainer import BaseTrainer
from src.training.callbacks import CallbackList, TrainingCallback

logger = logging.getLogger(__name__)

# Metrics every iteration reports.
# Anything else in a metrics mapping comes from the algorithm's _update and is
# formatted generically.
_CORE_METRICS = ("frames", "episodes", "win_rate", "draw_rate", "fps")


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

    def train(self) -> dict[str, float]:
        """
        Run collection until ``total_frames``, updating after every rollout.

        :return: Aggregate statistics: frames, episodes, win/draw rate and fps.
        """
        # Opt out of torchrl's automatic policy-transform registration: env
        # transforms are managed explicitly by the env factories, and the
        # policies used here read "action_mask" directly without needing the
        # InitTracker transform the collector's heuristic would append.
        collector = Collector(   # Maybe move to data member
            create_env_fn=self._make_vec_env(),
            policy=self._policy,
            frames_per_batch=self._frames_per_batch,
            total_frames=self._total_frames,
            auto_register_policy_transforms=False,
            **self._collector_kwargs(),
        )
        frames = 0
        episodes = 0
        wins = 0
        draws = 0
        start_time = time.time()
        self._callbacks.on_train_start(self._run_config)
        try:
            with tqdm(total=self._total_frames, unit="frame") as progress_bar:
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
                    self._log_progress(progress_bar, episodes, wins, losses)
                    self._log_metrics(metrics)
                    self._callbacks.on_rollout_end(frames, metrics)
        finally:
            collector.shutdown()
            summary = self._metrics(frames, episodes, wins, draws, time.time() - start_time)
            self._callbacks.on_train_end(summary)
        return summary

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

        TODO: No-op in the base class; PPO overrides this with the
        advantage/minibatch/optimizer loop.

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

        :return: ParallelEnv (fork workers) or SerialEnv over the factories.
        """
        if self._use_parallel_env:
            return ParallelEnv(
                num_workers=len(self._env_factories),
                create_env_fn=self._env_factories,
                mp_start_method=self._mp_start_method,
                serial_for_single=self._serial_for_single,
            )
        return SerialEnv(num_workers=len(self._env_factories),
                         create_env_fn=self._env_factories)

    @staticmethod
    def _log_progress(
            progress_bar: tqdm,
            episodes: int,
            wins: int,
            losses: dict[str, float] | None,
    ) -> None:
        """
        Update the progress bar's postfix with episode and loss statistics.

        :param progress_bar: tqdm bar tracking collected frames.
        :param episodes: Total episodes finished so far.
        :param wins: Total wins so far.
        :param losses: Loss values from the last update, if any.
        """
        postfix = {"episodes": episodes, "win_rate": f"{wins / max(episodes, 1):.3f}"}
        if losses:
            postfix.update({key: f"{value:.4f}" for key, value in losses.items()})
        progress_bar.set_postfix(postfix)

    def _log_metrics(self, metrics: Mapping[str, float]) -> None:
        """
        Emit one timestamped log line per collector iteration with the
        running training metrics.

        The tqdm progress bar (:meth:`_log_progress`) only overwrites a
        single terminal line in place, so it leaves no persistent record of
        metrics over time; this writes through the standard ``logging``
        module instead (picked up by Hydra's default handler, so every line
        carries a timestamp), independent of whether a progress bar is
        attached to a terminal.

        :param metrics: Mapping from :meth:`_metrics`. Keys outside
            :data:`_CORE_METRICS` are appended generically, so a new loss
            term needs no change here.
        """
        parts = [
            f"frames={int(metrics['frames'])}/{self._total_frames}",
            f"episodes={int(metrics['episodes'])}",
            f"win_rate={metrics['win_rate']:.3f}",
            f"draw_rate={metrics['draw_rate']:.3f}",
            f"fps={metrics['fps']:.1f}",
        ]
        parts.extend(
            f"{key}={value:.4f}"
            for key, value in metrics.items()
            if key not in _CORE_METRICS
        )
        logger.info(" ".join(parts))
