import math

import pytest
import torch
from torch import nn
from torchrl.collectors import Collector
from torchrl.envs import SerialEnv

from src.env.opponent_pool import OpponentPool
from src.env.random_opponent import RandomOpponent
from src.models.masked_rpo_categorical import MaskedRPOCategorical
from src.policies.ppo_actor import build_actor_critic
from src.training.env_factory import make_env_factories
from tests.conftest import PPOTrainerForTests, flat_env_cfg


def make_random_pool() -> OpponentPool:
    """
    Build a two-member random opponent pool (module-level for picklability).

    :return: OpponentPool with two seeded random opponents.
    """
    return OpponentPool([RandomOpponent(seed=0), RandomOpponent(seed=1)], seed=2)


def _make_trainer(actor_critic, action_spec, opponent_factory=None, **kwargs) -> PPOTrainerForTests:
    """
    Build a small PPO trainer over a serial flat-observation env.

    :param actor_critic: Actor-critic to train.
    :param action_spec: Environment action spec.
    :param opponent_factory: Optional self-play opponent factory.
    :param kwargs: Overrides forwarded to :class:`PPOTrainerForTests` (e.g.
        ``rpo_alpha``, ``target_kl``, ``use_amp``, annealing flags).
    :return: A configured PPO trainer with tiny budgets.
    """
    params = {
        "env_factories": make_env_factories(flat_env_cfg(), opponent_factory=opponent_factory),
        "actor_critic": actor_critic,
        "action_spec": action_spec,
        "frames_per_batch": 64,
        "total_frames": 128,
        "num_epochs": 2,
        "sub_batch_size": 32,
        "use_parallel_env": False,
    }
    params.update(kwargs)
    return PPOTrainerForTests(**params)


def _collect_one_batch(trainer: PPOTrainerForTests) -> object:
    """
    Collect a single 64-frame batch with a trainer's own collection policy.

    :param trainer: Trainer whose policy drives collection.
    :return: One collected batch (cloned so the collector can shut down).
    """
    env = SerialEnv(2, make_env_factories(flat_env_cfg()))
    collector = Collector(
        env,
        trainer.policy_for_test,
        frames_per_batch=64,
        total_frames=64,
        auto_register_policy_transforms=False,
    )
    try:
        return next(iter(collector)).clone()
    finally:
        collector.shutdown()


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
    collector = Collector(
        env,
        trainer.policy_for_test,
        frames_per_batch=64,
        total_frames=64,
        auto_register_policy_transforms=False,
    )
    try:
        losses = trainer.update_for_test(next(iter(collector)))
    finally:
        collector.shutdown()
    # _update returns None only if every minibatch was NaN/Inf; a healthy batch
    # must produce real losses.
    assert losses is not None
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


def test_ppo_trainer_rpo_runs_and_leaves_flag_clear(model_cfg, flat_obs_spec, action_spec) -> None:
    """
    RPO training runs NaN-free and the process-global perturbation flag is
    cleared after the update, so rollout collection is never perturbed.
    """
    actor_critic = build_actor_critic(model_cfg, flat_obs_spec, action_spec)
    trainer = _make_trainer(actor_critic, action_spec, rpo_alpha=0.5)
    stats = trainer.train()
    assert stats["frames"] == 128
    assert all(torch.isfinite(p).all() for p in actor_critic.parameters())
    assert MaskedRPOCategorical.rpo_enabled is False


def test_ncl_model_is_guarded_stub(model_cfg, flat_obs_spec, action_spec) -> None:
    """
    Passing a non-None ``ncl_model`` raises ``NotImplementedError`` (guarded stub).
    """
    actor_critic = build_actor_critic(model_cfg, flat_obs_spec, action_spec)
    with pytest.raises(NotImplementedError):
        _make_trainer(actor_critic, action_spec, ncl_model=nn.Linear(2, 2))


def test_target_kl_early_stops_epoch_loop(model_cfg, flat_obs_spec, action_spec) -> None:
    """
    ``target_kl`` breaks the epoch loop early: far fewer optimizer steps than
    ``num_epochs * minibatches`` (10 epochs * 2 minibatches = 20).

    A large ``lr`` makes the policy diverge fast so ``kl_approx`` grows
    decisively positive (~2e-2 after the first epoch), well above the
    ``1e-3`` budget — so the early stop fires after epoch 1 regardless of
    initialization. (A tiny ``target_kl`` would be unreliable: torchrl's
    ``kl_approx`` oscillates around zero, even negative, when the policy
    barely moves.)
    """
    actor_critic = build_actor_critic(model_cfg, flat_obs_spec, action_spec)
    trainer = _make_trainer(actor_critic, action_spec, num_epochs=10, lr=0.1, target_kl=1e-3)
    steps = trainer.count_optimizer_steps_for_update(_collect_one_batch(trainer))
    assert 0 < steps < 20


def test_amp_update_is_finite_on_cpu(model_cfg, flat_obs_spec, action_spec) -> None:
    """
    AMP (bfloat16 autocast, no scaler on CPU) runs the update NaN-free.
    """
    actor_critic = build_actor_critic(model_cfg, flat_obs_spec, action_spec)
    trainer = _make_trainer(actor_critic, action_spec, use_amp=True)
    stats = trainer.train()
    assert stats["frames"] == 128
    assert all(torch.isfinite(p).all() for p in actor_critic.parameters())


def test_lr_and_entropy_anneal_decrease(model_cfg, flat_obs_spec, action_spec) -> None:
    """
    With annealing enabled, the LR and entropy coefficient fall below their
    initial values over the course of training.
    """
    actor_critic = build_actor_critic(model_cfg, flat_obs_spec, action_spec)
    trainer = _make_trainer(
        actor_critic,
        action_spec,
        total_frames=192,  # 3 collector iterations at frames_per_batch=64
        lr=1.0e-3,
        entropy_coeff=0.05,
        lr_anneal=True,
        ent_anneal=True,
        ent_warm_frac=0.0,
    )
    trainer.train()
    assert trainer.current_lr_for_test < 1.0e-3
    assert trainer.current_entropy_coeff_for_test < 0.05
