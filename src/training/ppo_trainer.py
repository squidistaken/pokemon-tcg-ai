from collections.abc import Callable

import torch
from tensordict import TensorDict
from torch import nn
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorSpec
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import EnvBase
from torchrl.modules import ActorValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from src.models.actor_critic import ActorCritic
from src.policies.ppo_actor import build_ppo_operator
from src.training.trainer import Trainer


class PPOTrainer(Trainer):
    """
    Clipped-PPO trainer with invalid-action masking over the TCG environment.

    Extends :class:`~src.training.trainer.Trainer` by implementing the
    :meth:`_update` hook with the canonical on-policy PPO loop
    (``docs/torchrl/05-ppo-recipe.md``): estimate advantages with
    :class:`~torchrl.objectives.value.GAE`, refill a
    :class:`~torchrl.data.ReplayBuffer` sampled without replacement, and run a
    minibatch loop over :class:`~torchrl.objectives.ClipPPOLoss`
    (``loss_objective + loss_critic + loss_entropy``) with gradient clipping.
    GAE is recomputed at the start of every epoch because the value estimates
    change as the critic learns.

    The collection policy and both loss networks are views of a single
    shared-trunk :class:`~torchrl.modules.ActorValueOperator`, so the backbone
    runs once per step and its parameters are optimized once despite feeding
    both heads. The wrapped :class:`~src.models.actor_critic.ActorCritic` is
    kept for snapshotting into a self-play pool (see
    :class:`~src.policies.greedy_policy_opponent.GreedyPolicyOpponent`).
    """

    def __init__(
            self,
            env_factories: list[Callable[[], EnvBase]],
            actor_critic: ActorCritic,
            action_spec: TensorSpec,
            frames_per_batch: int,
            total_frames: int,
            clip_epsilon: float = 0.2,
            entropy_coeff: float = 0.01,
            gamma: float = 0.99,
            lmbda: float = 0.95,
            lr: float = 3.0e-4,
            num_epochs: int = 4,
            sub_batch_size: int = 256,
            max_grad_norm: float = 1.0,
            device: torch.device | str = "cpu",
            use_parallel_env: bool = True,
            mp_start_method: str = "fork",
            serial_for_single: bool = True,
    ) -> None:
        """
        :param env_factories: One environment factory per worker.
        :param actor_critic: Shared-trunk actor-critic to train; its modules
            are reused by the collection policy and both loss networks.
        :param action_spec: Action spec used to wire the masked policy.
        :param frames_per_batch: Frames collected per collector iteration; also
            the replay-buffer capacity for the on-policy reuse.
        :param total_frames: Total frames to collect over the run.
        :param clip_epsilon: PPO surrogate clipping range.
        :param entropy_coeff: Entropy-bonus weight.
        :param gamma: Discount factor for GAE.
        :param lmbda: GAE trace-decay factor.
        :param lr: Adam learning rate.
        :param num_epochs: Optimization epochs over each collected batch.
        :param sub_batch_size: Minibatch size drawn from the buffer per step.
        :param max_grad_norm: Gradient-norm clipping threshold.
        :param device: Device for optimization tensors.
        :param use_parallel_env: Use ParallelEnv instead of SerialEnv.
        :param mp_start_method: Multiprocessing start method for ParallelEnv.
        :param serial_for_single: Fall back to a single-process env for one worker.
        """
        self._actor_critic = actor_critic
        self._operator: ActorValueOperator = build_ppo_operator(actor_critic, action_spec).to(device)
        super().__init__(
            env_factories=env_factories,
            policy=self._operator.get_policy_operator(),
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            use_parallel_env=use_parallel_env,
            mp_start_method=mp_start_method,
            serial_for_single=serial_for_single,
        )
        self._device = torch.device(device)
        self._num_epochs = num_epochs
        self._sub_batch_size = min(sub_batch_size, frames_per_batch)
        self._max_grad_norm = max_grad_norm

        self._advantage = GAE(
            gamma=gamma,
            lmbda=lmbda,
            value_network=self._operator.get_value_operator(),
            average_gae=True,
        )
        self._loss = ClipPPOLoss(
            actor_network=self._operator.get_policy_operator(),
            critic_network=self._operator.get_value_operator(),
            clip_epsilon=clip_epsilon,
            entropy_bonus=True,
            entropy_coeff=entropy_coeff,
        )
        self._optim = torch.optim.Adam(self._loss.parameters(), lr=lr)
        self._replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(frames_per_batch, device=self._device),
            sampler=SamplerWithoutReplacement(),
        )

    @property
    def actor_critic(self) -> ActorCritic:
        """
        The actor-critic being trained (for checkpointing / snapshots).

        :return: The wrapped :class:`ActorCritic`.
        """
        return self._actor_critic

    def _update(self, data: TensorDict) -> dict[str, float]:
        """
        Run the PPO advantage/minibatch/optimizer loop over a collected batch.

        :param data: One ``(B, T)`` batch from the collector (see
            :meth:`Trainer._update` for the tensordict layout). The stored
            ``action_log_prob`` from collection is the old policy's, as PPO
            requires.
        :return: Mean losses and gradient norm over the update, for logging.
        """
        data = data.to(self._device)
        totals = {"loss_objective": 0.0, "loss_critic": 0.0, "loss_entropy": 0.0, "grad_norm": 0.0}
        steps = 0
        n_minibatches = max(1, data.numel() // self._sub_batch_size)
        for _ in range(self._num_epochs):
            with torch.no_grad():
                self._advantage(data)
            self._replay_buffer.empty()
            self._replay_buffer.extend(data.reshape(-1))
            for _ in range(n_minibatches):
                minibatch = self._replay_buffer.sample(self._sub_batch_size)
                loss_values = self._loss(minibatch)
                loss = (
                    loss_values["loss_objective"]
                    + loss_values["loss_critic"]
                    + loss_values["loss_entropy"]
                )
                self._optim.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self._loss.parameters(), self._max_grad_norm)
                self._optim.step()
                totals["loss_objective"] += float(loss_values["loss_objective"].detach())
                totals["loss_critic"] += float(loss_values["loss_critic"].detach())
                totals["loss_entropy"] += float(loss_values["loss_entropy"].detach())
                totals["grad_norm"] += float(grad_norm)
                steps += 1
        return {key: value / max(steps, 1) for key, value in totals.items()}
