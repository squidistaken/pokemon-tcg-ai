import pytest
import torch
from tensordict import TensorDict

from src.models import ActorCritic, LinearPolicyHead, MLPBackbone, ValueHead
from src.models.backbone import activation_class
from src.policies.ppo_actor import build_actor_critic

from tests.conftest import FLAT_DIM, N_ACTIONS


def _dummy_obs(batch: int) -> TensorDict:
    """
    Build a flat observation tensordict with a partly-legal action mask.

    :param batch: Batch size.
    :return: TensorDict with ``observation`` and ``action_mask``.
    """
    mask = torch.zeros(batch, N_ACTIONS, dtype=torch.bool)
    mask[:, :4] = True
    return TensorDict(
        {
            "observation": {"observation": torch.randn(batch, FLAT_DIM)},
            "action_mask": mask,
        },
        batch_size=[batch],
    )


def test_activation_class_unknown_raises() -> None:
    """
    An unknown activation name raises a clear error.
    """
    assert activation_class("tanh") is torch.nn.Tanh
    with pytest.raises(ValueError, match="Unknown activation"):
        activation_class("sigmoidz")


def test_actor_critic_forward_shapes(model_cfg, flat_obs_spec, action_spec) -> None:
    """
    A forward pass produces logits ``(B, 97)`` and a scalar value ``(B, 1)``.
    """
    actor_critic = build_actor_critic(model_cfg, flat_obs_spec, action_spec)
    out = actor_critic(_dummy_obs(5))
    assert out["logits"].shape == (5, N_ACTIONS)
    assert out["state_value"].shape == (5, 1)


def test_gradients_reach_trunk_and_both_heads(model_cfg, flat_obs_spec, action_spec) -> None:
    """
    Backprop from both outputs reaches the shared trunk and both heads.
    """
    actor_critic = build_actor_critic(model_cfg, flat_obs_spec, action_spec)
    out = actor_critic(_dummy_obs(4))
    (out["logits"].sum() + out["state_value"].sum()).backward()

    trunk = next(actor_critic.backbone.parameters())
    policy = next(actor_critic.policy_head.parameters())
    value = next(actor_critic.value_head.parameters())
    assert trunk.grad is not None and torch.any(trunk.grad != 0)
    assert policy.grad is not None and torch.any(policy.grad != 0)
    assert value.grad is not None and torch.any(value.grad != 0)


def test_mlp_backbone_rejects_wrong_input_count() -> None:
    """
    The MLP backbone validates that it receives one tensor per input key.
    """
    backbone = MLPBackbone(input_dim=FLAT_DIM, out_features=8, num_cells=[8], in_keys=["observation"])
    with pytest.raises(ValueError, match="expected 1 inputs"):
        backbone(torch.randn(2, FLAT_DIM), torch.randn(2, FLAT_DIM))


def test_linear_head_ignores_option_repr() -> None:
    """
    The linear head reads only ``state_repr`` and tolerates ``option_repr``.
    """
    head = LinearPolicyHead(in_features=8, n_actions=N_ACTIONS)
    state = torch.randn(3, 8)
    logits = head(state, option_repr=torch.randn(3, 5, 8))
    assert logits.shape == (3, N_ACTIONS)
    assert torch.equal(logits, head(state))


def test_value_head_scalar_output() -> None:
    """
    The value head maps ``state_repr`` to a scalar per sample.
    """
    head = ValueHead(in_features=8, num_cells=[8])
    assert head(torch.randn(6, 8)).shape == (6, 1)


def test_incompatible_head_backbone_raises(monkeypatch, model_cfg, flat_obs_spec, action_spec) -> None:
    """
    Pairing an option-requiring head with an option-less backbone is rejected.
    """
    monkeypatch.setattr(LinearPolicyHead, "requires_option_repr", True, raising=False)
    with pytest.raises(ValueError, match="per-option tokens"):
        build_actor_critic(model_cfg, flat_obs_spec, action_spec)
