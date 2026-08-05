from typing import cast

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torchrl.data import Composite

from src.models import LinearPolicyHead, MLPBackbone, TransformerBackbone, ValueHead
from src.models.backbone import activation_class
from src.models.heads import PointerPolicyHead
from src.models.structured_obs_adapter import StructuredObsAdapter
from src.policies.ppo_actor import build_actor_critic
from tests.conftest import N_ACTIONS

#: The in_keys every structured-observation backbone consumes, as the
#: transformer fixture declares them.
_STRUCTURED_IN_KEYS = [
    ("observation", "globals"),
    ("observation", "select_cats"),
    ("observation", "context_card_ids"),
    ("observation", "stadium_id"),
    ("observation", "options"),
    ("observation", "pokemon"),
    ("observation", "my"),
    ("observation", "opp"),
    ("observation", "select_deck"),
    ("observation", "looking"),
]


def _transformer_cfg(base: DictConfig, **backbone: object) -> DictConfig:
    """
    Copy the transformer fixture config with backbone keys overridden.

    Mirrors what ``conf/experiment/tf_*.yaml`` do to
    ``conf/model/backbone/transformer.yaml``, so these tests exercise the same
    compositions the diagnosis sweep launches.

    :param base: The ``transformer_model_cfg`` fixture.
    :param backbone: Keys to set under ``model.backbone``.
    :return: A merged copy; the fixture is left untouched.
    """
    return cast(
        DictConfig, OmegaConf.merge(base, {"model": {"backbone": backbone}})
    )


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


def test_transformer_rejects_unknown_pooling(structured_obs_spec) -> None:
    """
    The readout mode is validated at construction, not at the first forward.
    """
    adapter = StructuredObsAdapter(
        obs_spec=structured_obs_spec, in_keys=[("observation", "globals")]
    )
    with pytest.raises(ValueError, match="Unknown pooling"):
        TransformerBackbone(
            input_dim=adapter.out_features,
            out_features=8,
            adapter=adapter,
            num_heads=2,
            pooling="sum",
            in_keys=[("observation", "globals")],
        )


@pytest.mark.parametrize("pooling", ["mean", "cls", "attention"])
def test_transformer_pooling_modes_produce_state_repr(
        pooling, transformer_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    Every readout mode yields the same downstream shapes (``+experiment=tf_pooling``).
    """
    cfg = _transformer_cfg(transformer_model_cfg, pooling=pooling)
    actor_critic = build_actor_critic(cfg, structured_obs_spec, action_spec)
    out = actor_critic(_dummy_obs(structured_obs_spec, 5))
    assert out["logits"].shape == (5, N_ACTIONS)
    assert out["state_value"].shape == (5, 1)


def test_transformer_pre_ln_forward(
        transformer_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    Pre-LN plus a final LayerNorm composes and stays finite
    (``+experiment=tf_preln``).
    """
    cfg = _transformer_cfg(transformer_model_cfg, norm_first=True, final_norm=True)
    actor_critic = build_actor_critic(cfg, structured_obs_spec, action_spec)
    out = actor_critic(_dummy_obs(structured_obs_spec, 4))
    assert out["logits"].shape == (4, N_ACTIONS)
    assert torch.isfinite(out["logits"]).all()


def test_transformer_entity_tokens_reach_gradients(
        transformer_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    Expanding a group into per-entity tokens keeps output shapes and trains
    the new projection (``+experiment=tf_tokens``).
    """
    cfg = _transformer_cfg(transformer_model_cfg, token_groups=["pokemon"])
    actor_critic = build_actor_critic(cfg, structured_obs_spec, action_spec)

    # _dummy_obs leaves the board empty, and a fully-padded group correctly
    # contributes no gradient (see test_transformer_ignores_padded_entities),
    # so put a Pokemon on it before asking whether the path trains.
    obs = _dummy_obs(structured_obs_spec, 4)
    obs[("observation", "pokemon", "card_id")][:, 0] = 7
    obs[("observation", "pokemon", "mask")][:, 0] = True

    out = actor_critic(obs)
    assert out["logits"].shape == (4, N_ACTIONS)

    out["logits"].sum().backward()
    backbone = cast(TransformerBackbone, actor_critic.backbone)
    projection = next(backbone.entity_projections["pokemon"].parameters())
    assert projection.grad is not None and torch.any(projection.grad != 0)


def test_transformer_ignores_padded_entities(
        transformer_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    Content in a masked-out entity slot cannot influence ``state_repr``.

    The guard that matters for the token path: padded slots are excluded both
    as attention keys (``src_key_padding_mask``) and from the pooled readout.
    Miss either and the model reads padding as if it were board state.
    """
    cfg = _transformer_cfg(transformer_model_cfg, token_groups=["pokemon"])
    actor_critic = build_actor_critic(cfg, structured_obs_spec, action_spec)
    actor_critic.eval()

    obs = _dummy_obs(structured_obs_spec, 3)
    baseline = actor_critic(obs.clone())["state_value"]

    # Fill a slot the mask marks absent; a correct model is blind to it.
    polluted = obs.clone()
    assert not polluted[("observation", "pokemon", "mask")][:, 2].any()
    polluted[("observation", "pokemon", "card_id")][:, 2] = 11
    polluted[("observation", "pokemon", "features")][:, 2] = 5.0

    assert torch.allclose(baseline, actor_critic(polluted)["state_value"], atol=1e-6)


def test_pointer_head_scores_option_tokens(
        transformer_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    The pointer head consumes the backbone's per-option tokens and emits one
    logit per action slot (``+experiment=tf_pointer``).
    """
    cfg = _transformer_cfg(transformer_model_cfg, option_tokens=True)
    cfg = cast(
        DictConfig,
        OmegaConf.merge(cfg, {"model": {"head": {"_target_": "src.models.heads.PointerPolicyHead"}}}),
    )
    actor_critic = build_actor_critic(cfg, structured_obs_spec, action_spec)
    assert actor_critic.backbone.produces_option_repr

    out = actor_critic(_dummy_obs(structured_obs_spec, 5))
    assert out["logits"].shape == (5, N_ACTIONS)

    out["logits"].sum().backward()
    query = next(actor_critic.policy_head.parameters())
    assert query.grad is not None and torch.any(query.grad != 0)


def test_pointer_head_distinguishes_options() -> None:
    """
    Two option slots with different representations get different logits.

    This is the property ``LinearPolicyHead`` lacks: it reads only the pooled
    state, so it can express a preference for a *slot index* but never for the
    option occupying it.
    """
    head = PointerPolicyHead(in_features=8, n_actions=3)
    option_repr = torch.zeros(1, 3, 8)
    option_repr[0, 0] = torch.ones(8)
    option_repr[0, 1] = -torch.ones(8)
    logits = head(torch.randn(1, 8), option_repr)
    assert logits.shape == (1, 3)
    assert not torch.isclose(logits[0, 0], logits[0, 1])


def test_pointer_head_requires_option_repr() -> None:
    """
    Without per-option tokens the head cannot function, and says so.
    """
    head = PointerPolicyHead(in_features=8, n_actions=N_ACTIONS)
    with pytest.raises(ValueError, match="needs per-option tokens"):
        head(torch.randn(2, 8))


def test_pointer_head_rejects_slot_mismatch() -> None:
    """
    The option table must carry exactly one row per action slot, or the logits
    would silently misalign with the action mask.
    """
    head = PointerPolicyHead(in_features=8, n_actions=N_ACTIONS)
    with pytest.raises(ValueError, match="slots but the action space"):
        head(torch.randn(2, 8), torch.randn(2, N_ACTIONS - 1, 8))


def test_transformer_combined_arm(
        transformer_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    Every diagnosis change stacked, as ``+experiment=tf_combined`` runs it.
    """
    cfg = _transformer_cfg(
        transformer_model_cfg,
        num_layers=2,
        ff_dim=64,
        norm_first=True,
        final_norm=True,
        pooling="cls",
        token_groups=["pokemon"],
        option_tokens=True,
    )
    cfg = cast(
        DictConfig,
        OmegaConf.merge(cfg, {"model": {"head": {"_target_": "src.models.heads.PointerPolicyHead"}}}),
    )
    actor_critic = build_actor_critic(cfg, structured_obs_spec, action_spec)
    out = actor_critic(_dummy_obs(structured_obs_spec, 4))
    assert out["logits"].shape == (4, N_ACTIONS)
    assert out["state_value"].shape == (4, 1)
    assert torch.isfinite(out["logits"]).all()


def test_encode_entity_tokens_shapes_and_masks(structured_obs_spec) -> None:
    """
    Entity tokens keep the padded slot count and report validity per slot.
    """
    adapter = StructuredObsAdapter(obs_spec=structured_obs_spec, in_keys=_STRUCTURED_IN_KEYS)
    obs = _dummy_obs(structured_obs_spec, 2)
    inputs = [obs.get(key) for key in _STRUCTURED_IN_KEYS]

    tokens = adapter.encode_entity_tokens(*inputs, groups=["options", "pokemon", "my"])
    for name, (values, mask) in tokens.items():
        assert values.shape == (2, adapter.group_slot_counts[name], 64)
        assert mask.shape == (2, adapter.group_slot_counts[name])

    # _dummy_obs fills four option rows and three hand cards; nothing else.
    assert tokens["options"][1].sum(dim=-1).tolist() == [4, 4]
    assert tokens["my"][1].sum(dim=-1).tolist() == [3, 3]
    assert not tokens["pokemon"][1].any()


def test_encode_entity_tokens_rejects_scalar_groups(structured_obs_spec) -> None:
    """
    ``globals`` has no entity axis; it is already one token via encode_groups.
    """
    adapter = StructuredObsAdapter(obs_spec=structured_obs_spec, in_keys=_STRUCTURED_IN_KEYS)
    obs = _dummy_obs(structured_obs_spec, 2)
    inputs = [obs.get(key) for key in _STRUCTURED_IN_KEYS]

    with pytest.raises(ValueError, match="no entity axis"):
        adapter.encode_entity_tokens(*inputs, groups=["globals"])
    with pytest.raises(ValueError, match="not among the adapter"):
        adapter.encode_entity_tokens(*inputs, groups=["nonexistent"])


def test_incompatible_head_backbone_raises(
        monkeypatch, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    Pairing an option-requiring head with an option-less backbone is rejected.
    """
    monkeypatch.setattr(LinearPolicyHead, "requires_option_repr", True, raising=False)
    with pytest.raises(ValueError, match="per-option tokens"):
        build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec)
