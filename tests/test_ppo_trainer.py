import math

import torch
from torchrl.collectors import Collector
from torchrl.envs import SerialEnv

from src.env.opponent_pool import OpponentPool
from src.env.random_opponent import RandomOpponent
from src.policies.ppo_actor import build_actor_critic
from src.training.env_factory import make_env_factories
from src.training.ppo_trainer import PPOTrainer
from tests.conftest import flat_env_cfg


def make_random_pool() -> OpponentPool:
    """
    Build a two-member random opponent pool (module-level for picklability).

    :return: OpponentPool with two seeded random opponents.
    """
    return OpponentPool([RandomOpponent(seed=0), RandomOpponent(seed=1)], seed=2)


def _make_trainer(actor_critic, action_spec, opponent_factory=None) -> PPOTrainer:
    """
    Build a small PPO trainer over a serial flat-observation env.

    :param actor_critic: Actor-critic to train.
    :param action_spec: Environment action spec.
    :param opponent_factory: Optional self-play opponent factory.
    :return: A configured PPOTrainer with tiny budgets.
    """
    return PPOTrainer(
        env_factories=make_env_factories(flat_env_cfg(), opponent_factory=opponent_factory),
        actor_critic=actor_critic,
        action_spec=action_spec,
        frames_per_batch=64,
        total_frames=128,
        num_epochs=2,
        sub_batch_size=32,
        use_parallel_env=False,
    )


def test_ppo_trainer_trains_without_nans(model_cfg, flat_obs_spec, action_spec) -> None:
    """
    A short PPO run collects the requested frames, finishes episodes and leaves
    every parameter finite (no NaNs from the optimizer).
    """
    actor_critic = build_actor_critic(model_cfg, flat_obs_spec, action_spec)
    trainer = _make_trainer(actor_critic, action_spec)
    stats = trainer.train()
    assert stats["frames"] == 128
    assert stats["episodes"] > 0
    assert all(torch.isfinite(p).all() for p in actor_critic.parameters())


def test_ppo_update_returns_finite_losses(model_cfg, flat_obs_spec, action_spec) -> None:
    """
    ``_update`` on a real collected batch returns the three PPO loss terms and
    a gradient norm, all finite.
    """
    actor_critic = build_actor_critic(model_cfg, flat_obs_spec, action_spec)
    trainer = _make_trainer(actor_critic, action_spec)
    env = SerialEnv(2, make_env_factories(flat_env_cfg()))
    collector = Collector(env, trainer.policy, frames_per_batch=64, total_frames=64)
    try:
        losses = trainer.update(next(iter(collector)))
    finally:
        collector.shutdown()
    assert set(losses) >= {"loss_objective", "loss_critic", "loss_entropy", "grad_norm"}
    assert all(math.isfinite(value) for value in losses.values())


def test_ppo_trainer_self_play_pool(model_cfg, flat_obs_spec, action_spec) -> None:
    """
    Training against an opponent pool (the self-play scaffold) collects frames.
    """
    actor_critic = build_actor_critic(model_cfg, flat_obs_spec, action_spec)
    trainer = _make_trainer(actor_critic, action_spec, opponent_factory=make_random_pool)
    stats = trainer.train()
    assert stats["frames"] == 128
