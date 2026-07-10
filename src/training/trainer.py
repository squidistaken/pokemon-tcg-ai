import time
from typing import Callable

import torch.nn as nn
from tensordict import TensorDict
from torchrl.collectors import Collector
from torchrl.envs import EnvBase, ParallelEnv, SerialEnv
from tqdm import tqdm

from src.training.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Training scaffold around torchrl's Collector.

    Owns the vectorized environment, the collector lifecycle and progress
    statistics. The learning algorithm lives in :meth:`_update`, which is a
    no-op in this base class, so running it directly gives a pure collection
    loop (e.g. the random-policy baseline). The upcoming PPO trainer
    could subclass this, passes the actor as ``policy`` and implements
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
        """
        self._env_factories = env_factories
        self._policy = policy
        self._frames_per_batch = frames_per_batch
        self._total_frames = total_frames
        self._use_parallel_env = use_parallel_env
        self._mp_start_method = mp_start_method
        self._serial_for_single = serial_for_single

    def train(self) -> dict[str, float]:
        """
        Run collection until ``total_frames``, updating after every batch.

        :return: Aggregate statistics: frames, episodes, win/draw rate and fps.
        """
        collector = Collector(   # Maybe move to data member
            create_env_fn=self._make_vec_env(),
            policy=self._policy,
            frames_per_batch=self._frames_per_batch,
            total_frames=self._total_frames,
        )
        frames = 0
        episodes = 0
        wins = 0
        draws = 0
        start_time = time.time()
        try:
            with tqdm(total=self._total_frames, unit="frame") as progress_bar:
                for data in collector:
                    batch_frames = data.numel()
                    frames += batch_frames
                    done = data["next", "done"].reshape(-1)
                    final_rewards = data["next", "reward"].reshape(-1)[done]
                    episodes += int(done.sum())
                    wins += int((final_rewards > 0).sum())
                    draws += int((final_rewards == 0).sum())
                    losses = self._update(data)
                    progress_bar.update(batch_frames)
                    self._log_progress(progress_bar, episodes, wins, losses)
        finally:
            collector.shutdown()
        elapsed = time.time() - start_time
        return {
            "frames": frames,
            "episodes": episodes,
            "win_rate": wins / max(episodes, 1),
            "draw_rate": draws / max(episodes, 1),
            "fps": frames / elapsed,
        }

    def _update(self, data: TensorDict) -> dict[str, float] | None:
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

    def _log_progress(
            self,
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
