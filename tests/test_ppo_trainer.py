import math
from typing import cast

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torchrl.collectors import Collector
from torchrl.envs import SerialEnv

from src.env.opponent_pool import OpponentPool
from src.env.random_opponent import RandomOpponent
from src.policies.ppo_actor import build_actor_critic
from src.training.env_factory import make_env_factories
from tests.conftest import N_ACTIONS, PPOTrainerForTests, structured_env_cfg
from tests.test_curriculum import STEPS, WORKERS, make_curriculum
from tests.test_curriculum import batch as curriculum_batch


def make_random_pool() -> OpponentPool:
    """
    Build a two-member random opponent pool (module-level for picklability).

    :return: OpponentPool with two seeded random opponents.
    """
    return OpponentPool([RandomOpponent(seed=0), RandomOpponent(seed=1)], seed=2)


def _make_trainer(actor_critic, action_spec, opponent_factory=None, **kwargs) -> PPOTrainerForTests:
    """
    Build a small PPO trainer over a serial structured-observation env.

    :param actor_critic: Actor-critic to train.
    :param action_spec: Environment action spec.
    :param opponent_factory: Optional self-play opponent factory.
    :param kwargs: Overrides forwarded to :class:`PPOTrainerForTests` (e.g.
        ``target_kl``, ``use_amp``, annealing flags).
    :return: A configured PPO trainer with tiny budgets.
    """
    params = {
        "env_factories": make_env_factories(structured_env_cfg(), opponent_factory=opponent_factory),
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
    env = SerialEnv(2, make_env_factories(structured_env_cfg()))
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


def _pointer_head_cfg(transformer_model_cfg: DictConfig) -> DictConfig:
    """
    Transformer + pointer-head config: the only pairing whose collection
    policy actually produces ``option_repr`` (``requires_option_repr`` on the
    head and ``produces_option_repr`` on the backbone both true), so it is
    the one that would show B1's fix doing nothing if it were broken.

    :param transformer_model_cfg: The transformer fixture config.
    :return: A merged copy with ``option_tokens=True`` and the pointer head;
        the fixture itself is left untouched.
    """
    return cast(
        DictConfig,
        OmegaConf.merge(
            transformer_model_cfg,
            {
                "model": {
                    "backbone": {"option_tokens": True},
                    "head": {"_target_": "src.models.heads.PointerPolicyHead"},
                }
            },
        ),
    )


def test_ppo_trainer_trains_without_nans(structured_model_cfg, structured_obs_spec, action_spec) -> None:
    """
    A short PPO run over the default pairing (structured obs + ``MLPBackbone``
    with its adapter) collects the requested frames, finishes episodes and
    leaves every parameter finite (no NaNs from the optimizer).

    Seeds torch explicitly: whether an episode finishes within the small
    frame budget below depends on the (otherwise unseeded) initial policy's
    action samples, which would otherwise make ``episodes > 0`` flaky
    depending on how much of the global RNG stream earlier tests consumed.
    """
    torch.manual_seed(0)
    actor_critic = build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec)
    trainer = _make_trainer(actor_critic, action_spec)
    stats = trainer.train()
    assert stats["frames"] == 128
    assert stats["episodes"] > 0
    assert all(torch.isfinite(p).all() for p in actor_critic.parameters())


def test_ppo_update_returns_finite_losses(structured_model_cfg, structured_obs_spec, action_spec) -> None:
    """
    ``_update`` on a real collected batch returns the three PPO loss terms, a
    gradient norm and the policy entropy, all finite.
    """
    actor_critic = build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec)
    trainer = _make_trainer(actor_critic, action_spec)
    env = SerialEnv(2, make_env_factories(structured_env_cfg()))
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
    assert set(losses) >= {"loss_objective", "loss_critic", "loss_entropy", "grad_norm", "entropy"}
    assert all(math.isfinite(value) for value in losses.values())


def test_ppo_update_reports_unweighted_policy_entropy(
    structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    The reported ``entropy`` is the policy's own entropy, not the entropy
    already scaled by the bonus coefficient.

    Both properties are checked, because a metric that merely exists is not
    yet the right one: the value must sit in ``(0, log N_ACTIONS]``, the range
    of a categorical over the action space, and must relate to
    ``loss_entropy`` by exactly the configured coefficient. Logging
    ``loss_entropy`` in its place would pass the first check and fail the
    second, and under entropy annealing it would fall over the run purely
    because the coefficient does, with a flat policy entropy behind it.
    """
    entropy_coeff = 0.02
    actor_critic = build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec)
    trainer = _make_trainer(actor_critic, action_spec, entropy_coeff=entropy_coeff)
    losses = trainer.update_for_test(_collect_one_batch(trainer))
    assert losses is not None
    assert 0.0 < losses["entropy"] <= math.log(N_ACTIONS)
    assert losses["loss_entropy"] == pytest.approx(-entropy_coeff * losses["entropy"], rel=1e-5)


def test_collected_batch_excludes_option_repr_and_hidden(
    transformer_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    A real collected batch must carry only ``action``/``action_log_prob``,
    never ``option_repr``/``hidden``/``logits`` (B1, finding 5).

    Uses the pointer head, the one pairing whose collection-policy forward
    actually produces ``option_repr`` -- with the flat head this assertion
    would pass trivially even if the fix did nothing, since the flat policy
    never emits ``option_repr`` in the first place.
    """
    cfg = _pointer_head_cfg(transformer_model_cfg)
    actor_critic = build_actor_critic(cfg, structured_obs_spec, action_spec)
    trainer = _make_trainer(actor_critic, action_spec)
    data = _collect_one_batch(trainer)
    keys = set(data.keys())
    assert {"action", "action_log_prob"} <= keys
    assert not {"option_repr", "hidden", "logits"} & keys


#: `pytest.mark.filterwarnings` splits its argument on ":" (action:message:
#: category:module:lineno), so the message fragment below stops short of the
#: warning's own "aten::..." op name -- a `re.match` prefix is enough to
#: select it without breaking the split.
_VMAP_ATTENTION_FALLBACK_WARNING = "ignore:There is a performance drop:UserWarning"


@pytest.mark.filterwarnings(_VMAP_ATTENTION_FALLBACK_WARNING)
def test_ppo_update_finite_on_pointer_head_batch(
    transformer_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    ``_update`` still returns finite PPO losses on a batch collected with the
    pointer head, even though that batch never carried ``option_repr`` (B1).

    The loss's ``actor_network``/``critic_network`` are separate
    ``get_*_operator()`` wrappers over the shared trunk, so they recompute
    ``option_repr`` themselves from ``observation`` rather than reading it
    back from the collected data; trimming the collected keys must not starve
    them.

    This is the first test in the suite to run a real ``_update`` (GAE +
    ``ClipPPOLoss``) over a ``TransformerBackbone``: GAE's value network call
    is wrapped in ``vmap`` (torchrl's default), which forces the encoder's
    attention through functorch's CPU fallback for
    ``scaled_dot_product_attention`` -- correct, just unoptimized -- and
    ``torch`` reports that with a ``UserWarning`` this repo's
    ``filterwarnings = ["error", ...]`` would otherwise turn into a failure
    having nothing to do with what this test checks. Real (non-pytest)
    training never sees this filter, so the warning is silent there.
    """
    cfg = _pointer_head_cfg(transformer_model_cfg)
    actor_critic = build_actor_critic(cfg, structured_obs_spec, action_spec)
    trainer = _make_trainer(actor_critic, action_spec)
    losses = trainer.update_for_test(_collect_one_batch(trainer))
    assert losses is not None
    assert set(losses) >= {"loss_objective", "loss_critic", "loss_entropy", "grad_norm", "entropy"}
    assert all(math.isfinite(value) for value in losses.values())


def test_value_head_activation_from_config_reaches_module(
    structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    ``value_head.activation`` from config must reach the built ``ValueHead``
    (B2, finding 7); ``build_actor_critic`` previously forwarded only
    ``num_cells``, silently dropping this setting in favour of ``ValueHead``'s
    ``"tanh"`` default.
    """
    cfg = cast(
        DictConfig,
        OmegaConf.merge(
            structured_model_cfg, {"model": {"value_head": {"activation": "relu"}}}
        ),
    )
    actor_critic = build_actor_critic(cfg, structured_obs_spec, action_spec)
    mlp_modules = list(actor_critic.value_head.mlp.modules())
    assert any(isinstance(module, torch.nn.ReLU) for module in mlp_modules)
    assert not any(isinstance(module, torch.nn.Tanh) for module in mlp_modules)


def test_ppo_trainer_self_play_pool(structured_model_cfg, structured_obs_spec, action_spec) -> None:
    """
    Training against an opponent pool (the self-play scaffold) collects frames.
    """
    actor_critic = build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec)
    trainer = _make_trainer(actor_critic, action_spec, opponent_factory=make_random_pool)
    stats = trainer.train()
    assert stats["frames"] == 128


def test_target_kl_early_stops_epoch_loop(structured_model_cfg, structured_obs_spec, action_spec) -> None:
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
    actor_critic = build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec)
    trainer = _make_trainer(actor_critic, action_spec, num_epochs=10, lr=0.1, target_kl=1e-3)
    steps = trainer.count_optimizer_steps_for_update(_collect_one_batch(trainer))
    assert 0 < steps < 20


def test_amp_update_is_finite_on_cpu(structured_model_cfg, structured_obs_spec, action_spec) -> None:
    """
    AMP (bfloat16 autocast, no scaler on CPU) runs the update NaN-free.
    """
    actor_critic = build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec)
    trainer = _make_trainer(actor_critic, action_spec, use_amp=True)
    stats = trainer.train()
    assert stats["frames"] == 128
    assert all(torch.isfinite(p).all() for p in actor_critic.parameters())


def test_lr_and_entropy_anneal_decrease(structured_model_cfg, structured_obs_spec, action_spec) -> None:
    """
    With annealing enabled, the LR and entropy coefficient fall below their
    initial values over the course of training.
    """
    actor_critic = build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec)
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


def test_restart_abandons_open_curriculum_episodes(
    structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    Rebuilding a dead worker pool must clear the curriculum's row accumulators.

    The trainer owns the wiring; without it the residuals banked against the
    dead pool's rows are committed under whatever matchup the replacement pool
    deals into that row next.
    """
    curriculum = make_curriculum()
    trainer = _make_trainer(
        build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec),
        action_spec,
        curriculum=curriculum,
    )

    curriculum.observe(
        curriculum_batch(
            levels=[[0] * STEPS] * WORKERS,
            residuals=[[3.0] * STEPS] * WORKERS,
            done=[[False] * STEPS] * WORKERS,
        )
    )
    trainer.prepare_restart_for_test(1)
    curriculum.observe(
        curriculum_batch(
            levels=[[1] * STEPS] * WORKERS,
            residuals=[[1.0] * STEPS] * WORKERS,
            done=[[False] * (STEPS - 1) + [True]] * WORKERS,
        )
    )

    assert curriculum.buffer.entries[0].visits == 0
    assert curriculum.buffer.entries[1].mean_residual == pytest.approx(1.0)
