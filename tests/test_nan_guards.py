"""Contracts for the guards that keep a non-finite gradient from ending a run."""

import logging
from typing import cast

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torchrl.data import Composite, TensorSpec

from src.models.actor_critic import ActorCritic
from src.policies.ppo_actor import build_actor_critic
from src.trainer_builder import _load_warm_start_weights
from src.training.callbacks.finite_check import non_finite_entries
from src.training.callbacks.train_state_callback import TrainStateCallback


def _preln(cfg: DictConfig, final_norm: bool) -> DictConfig:
    """
    Copy a transformer model config with the pre-LN flags set.

    :param cfg: Model config carrying a transformer backbone.
    :param final_norm: Whether to add the LayerNorm on the encoder output.
    :return: A copy with ``norm_first`` on and ``final_norm`` as given.
    """
    switched = cfg.copy()
    switched.model.backbone = OmegaConf.merge(
        cfg.model.backbone, {"norm_first": True, "final_norm": final_norm}
    )
    return switched


def test_clip_grad_norm_poisons_the_model_without_the_guard() -> None:
    """
    Pin the behaviour the guard exists for: one NaN gradient makes every
    parameter NaN, because the clip coefficient itself becomes NaN.
    """
    healthy = nn.Parameter(torch.ones(3))
    poisoned = nn.Parameter(torch.ones(3))
    healthy.grad = torch.full((3,), 0.5)
    poisoned.grad = torch.full((3,), float("nan"))

    total_norm = nn.utils.clip_grad_norm_([healthy, poisoned], 1.0)

    assert not torch.isfinite(total_norm)
    assert healthy.grad is not None
    assert torch.isnan(healthy.grad).all(), (
        "clip_grad_norm_ scales every gradient by max_norm / total_norm, so a "
        "NaN norm spreads to gradients that were finite"
    )


def test_overflowing_gradient_norm_reports_non_finite() -> None:
    """
    A large but finite gradient still overflows the fp32 norm, which the guard
    must treat the same as an outright NaN.
    """
    param = nn.Parameter(torch.ones(3))
    param.grad = torch.full((3,), 1e25)

    assert not torch.isfinite(nn.utils.clip_grad_norm_([param], 1.0))


def test_non_finite_entries_names_only_the_bad_tensors() -> None:
    """
    The state-dict scan reports non-finite float tensors and nothing else.
    """
    state = {
        "good": torch.ones(2),
        "bad": torch.tensor([1.0, float("nan")]),
        "also_bad": torch.tensor([float("inf")]),
        "integer": torch.tensor([1, 2]),
    }

    assert sorted(non_finite_entries(state)) == ["also_bad", "bad"]


def test_train_state_write_refuses_a_non_finite_model(
    tmp_path,
    structured_model_cfg: DictConfig,
    structured_obs_spec: Composite,
    action_spec: TensorSpec,
) -> None:
    """
    A poisoned model must not overwrite the state a restart resumes from.
    """
    actor_critic = build_actor_critic(
        structured_model_cfg, structured_obs_spec, action_spec
    )
    optimizer = torch.optim.AdamW(actor_critic.parameters(), lr=1e-4)
    path = tmp_path / "train_state.pt"
    callback = TrainStateCallback(
        actor_critic=actor_critic, optimizer=optimizer, path=path, interval=1
    )

    callback.on_rollout_end(1000, {})
    assert path.is_file(), "a finite model should write normally"
    good_frames = torch.load(path, map_location="cpu", weights_only=False)["frames"]

    with torch.no_grad():
        next(iter(actor_critic.parameters())).fill_(float("nan"))
    callback.on_rollout_end(2000, {})

    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    assert reloaded["frames"] == good_frames, (
        "the NaN write must leave the previous resumable state in place"
    )
    assert not non_finite_entries(reloaded["state_dict"])


def test_warm_start_accepts_a_newly_added_final_layer_norm(
    transformer_model_cfg: DictConfig,
    structured_obs_spec: Composite,
    action_spec: TensorSpec,
    caplog,
) -> None:
    """
    Switching to pre-LN with ``final_norm`` on adds one LayerNorm the saved
    weights predate, and the resume has to survive that.
    """
    saved = build_actor_critic(
        _preln(transformer_model_cfg, final_norm=False),
        structured_obs_spec,
        action_spec,
    ).state_dict()
    target = build_actor_critic(
        _preln(transformer_model_cfg, final_norm=True), structured_obs_spec, action_spec
    )

    with caplog.at_level(logging.WARNING):
        _load_warm_start_weights(target, saved, "train.init_checkpoint")

    assert "identity initialization" in caplog.text
    encoder_norm = cast(nn.LayerNorm, target.backbone.encoder.norm)
    assert torch.equal(encoder_norm.weight, torch.ones_like(encoder_norm.weight))
    assert torch.equal(encoder_norm.bias, torch.zeros_like(encoder_norm.bias))


def test_warm_start_still_rejects_a_real_architecture_mismatch(
    transformer_model_cfg: DictConfig,
    structured_model_cfg: DictConfig,
    structured_obs_spec: Composite,
    action_spec: TensorSpec,
) -> None:
    """
    The LayerNorm tolerance must not turn into a blanket non-strict load.
    """
    saved = build_actor_critic(
        structured_model_cfg, structured_obs_spec, action_spec
    ).state_dict()
    target = build_actor_critic(transformer_model_cfg, structured_obs_spec, action_spec)

    with pytest.raises(ValueError, match="does not fit this model"):
        _load_warm_start_weights(target, saved, "train.resume_state")


def test_resume_reapplies_configured_optimizer_hyperparameters() -> None:
    """
    ``load_state_dict`` restores the saved param groups wholesale, so turning
    weight decay on for a resumed run is a silent no-op without the re-apply.
    """
    param = nn.Parameter(torch.ones(3))
    previous = torch.optim.AdamW([param], lr=2e-4, weight_decay=0.0)
    param.grad = torch.ones(3)
    previous.step()

    resumed = torch.optim.AdamW([param], lr=1e-4, weight_decay=0.01)
    resumed.load_state_dict(previous.state_dict())
    assert resumed.param_groups[0]["weight_decay"] == 0.0, (
        "documents the trap PPOTrainer works around"
    )

    for group in resumed.param_groups:
        group["lr"] = 1e-4
        group["weight_decay"] = 0.01
    assert resumed.param_groups[0]["weight_decay"] == 0.01


def test_actor_critic_is_the_type_the_guards_assume(
    structured_model_cfg: DictConfig,
    structured_obs_spec: Composite,
    action_spec: TensorSpec,
) -> None:
    """
    Both guards read ``state_dict`` off an :class:`ActorCritic`.
    """
    built = build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec)
    assert isinstance(built, ActorCritic)
    assert not non_finite_entries(built.state_dict())
