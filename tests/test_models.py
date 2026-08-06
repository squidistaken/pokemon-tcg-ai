from typing import cast

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Composite

from src.models import LinearPolicyHead, MLPBackbone, PointerHead, ValueHead
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


def test_pointer_head_without_structured_groups_raises(
        pointer_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    A pointer head over a non-structured observation is rejected.

    Token emission is derived from the chosen head, so a config *can no longer*
    pair the pointer head with an adapter that withholds tokens. The one
    remaining mismatch is a backbone whose in_keys name no structured group at
    all, leaving no adapter to produce them.
    """
    pointer_model_cfg.model.backbone.in_keys = [["observation", "globals"]]
    with pytest.raises(ValueError, match="per-option tokens"):
        build_actor_critic(pointer_model_cfg, structured_obs_spec, action_spec)


def test_pointer_head_is_permutation_equivariant(
        pointer_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    Reordering the option rows reorders the logits to match.

    This is the property the flat head lacks: it reads the option table only
    through a permutation-invariant mean, so its logits are unchanged by a
    permutation while the correct action moves, leaving slot index the only
    thing it can learn.
    """
    actor_critic = build_actor_critic(pointer_model_cfg, structured_obs_spec, action_spec)
    n_options = 5
    obs = _dummy_obs(structured_obs_spec, batch=1)
    obs[("observation", "options", "card_id")][:, :n_options] = torch.arange(1, n_options + 1)

    original = actor_critic.policy_logits(obs)[0]
    permutation = torch.tensor([4, 1, 0, 3, 2])
    reordered = obs.clone()
    options = cast(TensorDict, reordered[("observation", "options")])
    index = torch.arange(options["card_id"].shape[-1])
    index[:n_options] = permutation
    for leaf in list(options.keys(include_nested=True, leaves_only=True)):
        options.set(leaf, options.get(leaf)[:, index])
    permuted = actor_critic.policy_logits(reordered)[0]

    assert torch.allclose(original[:n_options][permutation], permuted[:n_options], atol=1e-5)
    assert torch.allclose(original[-1], permuted[-1], atol=1e-5)


def test_flat_head_is_permutation_invariant(
        structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    The flat baseline's logits do *not* follow a permutation of the options.

    Pinned deliberately: this is the defect the pointer head exists to fix, and
    a regression here would mean the two heads had silently converged.
    """
    actor_critic = build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec)
    n_options = 5
    obs = _dummy_obs(structured_obs_spec, batch=1)
    obs[("observation", "options", "card_id")][:, :n_options] = torch.arange(1, n_options + 1)

    original = actor_critic.policy_logits(obs)[0]
    reordered = obs.clone()
    options = cast(TensorDict, reordered[("observation", "options")])
    index = torch.arange(options["card_id"].shape[-1])
    index[:n_options] = torch.tensor([4, 1, 0, 3, 2])
    for leaf in list(options.keys(include_nested=True, leaves_only=True)):
        options.set(leaf, options.get(leaf)[:, index])

    assert torch.allclose(original, actor_critic.policy_logits(reordered)[0], atol=1e-6)


def test_pointer_head_requires_option_tokens() -> None:
    """
    Calling the pointer head without option tokens fails loudly.
    """
    head = PointerHead(in_features=8, n_actions=5, option_dim=6, num_cells=[8])
    with pytest.raises(ValueError, match="requires per-option tokens"):
        head(torch.randn(2, 8), None)
