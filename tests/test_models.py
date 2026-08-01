from typing import cast

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Composite

from src.models import LinearPolicyHead, MLPBackbone, TransformerBackbone, ValueHead
from src.models.backbone import activation_class
from src.models.structured_obs_adapter import StructuredObsAdapter
from src.policies.ppo_actor import build_actor_critic
from tests.conftest import N_ACTIONS


def _dummy_obs(structured_obs_spec: Composite, batch: int) -> TensorDict:
    """
    Build a small structured observation batch with a partly-legal mask.

    A few card IDs, option rows and global scalars are filled in so the
    encoding is non-degenerate (all-zero inputs would leave first-layer
    weight gradients zero).

    :param structured_obs_spec: Env spec (observation + action mask).
    :param batch: Batch size.
    :return: TensorDict with the ``observation`` groups and ``action_mask``.
    """
    obs = structured_obs_spec.zero((batch,))
    obs[("observation", "globals")] += torch.randn(batch, obs[("observation", "globals")].shape[-1])
    obs[("observation", "my", "hand_ids")][:, :3] = torch.tensor([5, 9, 14])
    obs[("observation", "my", "hand_mask")][:, :3] = True
    obs[("observation", "options", "card_id")][:, :4] = 7
    obs["action_mask"][:, :4] = True
    return obs


def test_activation_class_unknown_raises() -> None:
    """
    An unknown activation name raises a clear error.
    """
    assert activation_class("tanh") is torch.nn.Tanh
    with pytest.raises(ValueError, match="Unknown activation"):
        activation_class("sigmoidz")


def test_actor_critic_forward_shapes(structured_model_cfg, structured_obs_spec, action_spec) -> None:
    """
    A forward pass produces logits ``(B, 97)`` and a scalar value ``(B, 1)``.
    """
    actor_critic = build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec)
    out = actor_critic(_dummy_obs(structured_obs_spec, 5))
    assert out["logits"].shape == (5, N_ACTIONS)
    assert out["state_value"].shape == (5, 1)


def test_gradients_reach_trunk_and_both_heads(
        structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    Backprop from both outputs reaches the shared trunk (including the
    adapter's card embedding) and both heads.
    """
    actor_critic = build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec)
    out = actor_critic(_dummy_obs(structured_obs_spec, 4))
    (out["logits"].sum() + out["state_value"].sum()).backward()

    # nn.Module.__getattr__ is typed as returning Tensor | Module, so walking
    # into the submodules needs the concrete type restated for the checker.
    backbone = cast(MLPBackbone, actor_critic.backbone)
    adapter = cast(StructuredObsAdapter, backbone.adapter)
    trunk = next(backbone.mlp.parameters())
    embedding = adapter._card_embedding.weight  # noqa: SLF001
    policy = next(actor_critic.policy_head.parameters())
    value = next(actor_critic.value_head.parameters())
    assert trunk.grad is not None and torch.any(trunk.grad != 0)
    assert embedding.grad is not None and torch.any(embedding.grad != 0)
    assert policy.grad is not None and torch.any(policy.grad != 0)
    assert value.grad is not None and torch.any(value.grad != 0)


def test_mlp_backbone_rejects_wrong_input_count() -> None:
    """
    The MLP backbone validates that it receives one tensor per input key.
    """
    backbone = MLPBackbone(input_dim=16, out_features=8, num_cells=[8], in_keys=["observation"])
    with pytest.raises(ValueError, match="expected 1 inputs"):
        backbone(torch.randn(2, 16), torch.randn(2, 16))


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


def test_transformer_actor_critic_forward_shapes(
        transformer_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    The transformer backbone slots into the same actor-critic assembly as the
    MLP backbone and produces the same output shapes.
    """
    actor_critic = build_actor_critic(transformer_model_cfg, structured_obs_spec, action_spec)
    out = actor_critic(_dummy_obs(structured_obs_spec, 5))
    assert out["logits"].shape == (5, N_ACTIONS)
    assert out["state_value"].shape == (5, 1)


def test_transformer_gradients_reach_trunk_and_both_heads(
        transformer_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    Backprop from both outputs reaches the transformer trunk (including the
    adapter's card embedding) and both heads.
    """
    actor_critic = build_actor_critic(transformer_model_cfg, structured_obs_spec, action_spec)
    out = actor_critic(_dummy_obs(structured_obs_spec, 4))
    (out["logits"].sum() + out["state_value"].sum()).backward()

    backbone = cast(TransformerBackbone, actor_critic.backbone)
    adapter = cast(StructuredObsAdapter, backbone.adapter)
    trunk = next(backbone.encoder.parameters())
    embedding = adapter._card_embedding.weight  # noqa: SLF001
    policy = next(actor_critic.policy_head.parameters())
    value = next(actor_critic.value_head.parameters())
    assert trunk.grad is not None and torch.any(trunk.grad != 0)
    assert embedding.grad is not None and torch.any(embedding.grad != 0)
    assert policy.grad is not None and torch.any(policy.grad != 0)
    assert value.grad is not None and torch.any(value.grad != 0)


def test_transformer_backbone_requires_adapter() -> None:
    """
    The transformer backbone has no naive-flatten fallback: it needs the
    adapter's per-group widths and encodings.
    """
    with pytest.raises(ValueError, match="requires a StructuredObsAdapter"):
        TransformerBackbone(input_dim=16, out_features=8, adapter=None, num_heads=2)


def test_transformer_backbone_rejects_indivisible_heads(structured_obs_spec) -> None:
    """
    ``out_features`` must be divisible by ``num_heads`` (attention head width).
    """
    adapter = StructuredObsAdapter(
        obs_spec=structured_obs_spec,
        in_keys=[("observation", "globals")],
    )
    with pytest.raises(ValueError, match="divisible by num_heads"):
        TransformerBackbone(
            input_dim=adapter.out_features,
            out_features=8,
            adapter=adapter,
            num_heads=3,
            in_keys=[("observation", "globals")],
        )


def test_incompatible_head_backbone_raises(
        monkeypatch, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    Pairing an option-requiring head with an option-less backbone is rejected.
    """
    monkeypatch.setattr(LinearPolicyHead, "requires_option_repr", True, raising=False)
    with pytest.raises(ValueError, match="per-option tokens"):
        build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec)
