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
    weight gradients zero). ``cats[..., 0]`` (not just ``card_id``) is set on
    the real option rows: the encoder always writes a shifted option type
    there, and ``card_id`` alone (e.g. a YES/NO option) is not a row the real
    encoder could ever emit as padding-only (finding 8).

    :param structured_obs_spec: Env spec (observation + action mask).
    :param batch: Batch size.
    :return: TensorDict with the ``observation`` groups and ``action_mask``.
    """
    obs = structured_obs_spec.zero((batch,))
    obs[("observation", "globals")] += torch.randn(batch, obs[("observation", "globals")].shape[-1])
    obs[("observation", "my", "hand_ids")][:, :3] = torch.tensor([5, 9, 14])
    obs[("observation", "my", "hand_mask")][:, :3] = True
    obs[("observation", "options", "card_id")][:, :4] = 7
    obs[("observation", "options", "cats")][:, :4, 0] = 1
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
    # options also counts the always-valid synthetic stop slot (finding 8).
    assert tokens["options"][1].sum(dim=-1).tolist() == [5, 5]
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


# ── Entity identity (finding 1 / A2) ────────────────────────────────────────


def test_transformer_pokemon_seat_swap_changes_state_repr(
        transformer_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    A Pokemon's segment id encodes its seat; moving the same card from the
    agent's active slot to the opponent's active slot changes ``state_repr``
    (finding 1: entity tokens previously carried no seat identity, so a full
    seat swap was bit-identical).
    """
    cfg = _transformer_cfg(transformer_model_cfg, token_groups=["pokemon"])
    actor_critic = build_actor_critic(cfg, structured_obs_spec, action_spec)
    actor_critic.eval()

    agent_side = _dummy_obs(structured_obs_spec, 2)
    rows = agent_side[("observation", "pokemon", "card_id")].shape[-1]
    half = rows // 2
    agent_side[("observation", "pokemon", "card_id")][:, 0] = 11
    agent_side[("observation", "pokemon", "mask")][:, 0] = True

    opp_side = agent_side.clone()
    opp_side[("observation", "pokemon", "card_id")][:, 0] = 0
    opp_side[("observation", "pokemon", "mask")][:, 0] = False
    opp_side[("observation", "pokemon", "card_id")][:, half] = 11
    opp_side[("observation", "pokemon", "mask")][:, half] = True

    value_agent = actor_critic(agent_side)["state_value"]
    value_opp = actor_critic(opp_side)["state_value"]
    assert not torch.allclose(value_agent, value_opp)


def test_transformer_pokemon_bench_swap_within_seat_is_invariant(
        transformer_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    Two bench slots on the same seat share one segment id, so swapping them
    leaves ``state_repr`` unchanged -- segment ids must preserve permutation
    invariance *within* a seat even though they break it *across* seats.
    """
    cfg = _transformer_cfg(transformer_model_cfg, token_groups=["pokemon"])
    actor_critic = build_actor_critic(cfg, structured_obs_spec, action_spec)
    actor_critic.eval()

    obs = _dummy_obs(structured_obs_spec, 2)
    obs[("observation", "pokemon", "card_id")][:, 1] = 11
    obs[("observation", "pokemon", "mask")][:, 1] = True
    obs[("observation", "pokemon", "card_id")][:, 2] = 13
    obs[("observation", "pokemon", "mask")][:, 2] = True

    swapped = obs.clone()
    swapped[("observation", "pokemon", "card_id")][:, 1] = 13
    swapped[("observation", "pokemon", "card_id")][:, 2] = 11

    value_a = actor_critic(obs)["state_value"]
    value_b = actor_critic(swapped)["state_value"]
    assert torch.allclose(value_a, value_b, atol=1e-6)


def test_transformer_zone_identity_distinguishes_hand_from_discard(
        transformer_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    The same card in ``my.hand`` vs. ``my.discard`` gets a different zone
    segment id, so it produces a different ``state_repr`` (finding 1: zones
    otherwise share one card_repr + projection with no zone identity).
    """
    cfg = _transformer_cfg(transformer_model_cfg, token_groups=["my"])
    actor_critic = build_actor_critic(cfg, structured_obs_spec, action_spec)
    actor_critic.eval()

    in_hand = _dummy_obs(structured_obs_spec, 2)
    in_hand[("observation", "my", "hand_ids")][:, 0] = 11
    in_hand[("observation", "my", "hand_mask")][:, 0] = True

    in_discard = in_hand.clone()
    in_discard[("observation", "my", "hand_ids")][:, 0] = 0
    in_discard[("observation", "my", "hand_mask")][:, 0] = False
    in_discard[("observation", "my", "discard_ids")][:, 0] = 11
    in_discard[("observation", "my", "discard_mask")][:, 0] = True

    value_hand = actor_critic(in_hand)["state_value"]
    value_discard = actor_critic(in_discard)["state_value"]
    assert not torch.allclose(value_hand, value_discard)


def test_stop_slot_pointer_logit_differs_from_padded_slot(
        transformer_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    The stop slot's own segment embedding gives it a representation distinct
    from an ordinary padded slot, so the pointer head does not score them
    identically (finding 2: the two used to get the exact same logit).
    """
    cfg = _transformer_cfg(transformer_model_cfg, option_tokens=True)
    cfg = cast(
        DictConfig,
        OmegaConf.merge(cfg, {"model": {"head": {"_target_": "src.models.heads.PointerPolicyHead"}}}),
    )
    actor_critic = build_actor_critic(cfg, structured_obs_spec, action_spec)
    actor_critic.eval()

    # All-zero options: no real option is set, so an ordinary padded slot and
    # the stop slot share identical (zero) content -- only the segment
    # embedding can still tell the stop slot apart.
    obs = structured_obs_spec.zero((2,))
    out = actor_critic(obs)
    assert not torch.allclose(out["logits"][:, 0], out["logits"][:, -1])


# ── encoded_option_repr (finding 3 / A3) ────────────────────────────────────


def test_option_repr_defaults_to_pre_attention_projection(
        transformer_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    ``encoded_option_repr`` defaults to False: ``option_repr`` is exactly the
    cheap per-entity projection (plus type and segment identity), not
    anything attention has touched.
    """
    cfg = _transformer_cfg(transformer_model_cfg, option_tokens=True)
    actor_critic = build_actor_critic(cfg, structured_obs_spec, action_spec)
    backbone = cast(TransformerBackbone, actor_critic.backbone)
    obs = _dummy_obs(structured_obs_spec, 3)
    inputs = [obs.get(key) for key in _STRUCTURED_IN_KEYS]

    _, option_repr = backbone(*inputs)
    option_rows, _ = backbone.adapter.encode_entity_tokens(*inputs, groups=["options"])["options"]
    segment_ids = backbone.adapter.group_segment_ids["options"]
    expected = (
        backbone.entity_projections["options"](option_rows)
        + backbone.entity_type_embedding["options"]
        + backbone.entity_segment_embedding["options"][segment_ids]
    )
    assert torch.allclose(option_repr, expected)


def test_encoded_option_repr_reads_attended_tokens(
        transformer_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    ``encoded_option_repr=True`` scores option tokens after they pass through
    the encoder; with identical weights it still disagrees with the cheap
    projection, because attention actually changes the option rows.
    """
    cheap_cfg = _transformer_cfg(transformer_model_cfg, option_tokens=True)
    encoded_cfg = _transformer_cfg(
        transformer_model_cfg, option_tokens=True, token_groups=["options"], encoded_option_repr=True
    )
    cheap = build_actor_critic(cheap_cfg, structured_obs_spec, action_spec)
    encoded = build_actor_critic(encoded_cfg, structured_obs_spec, action_spec)
    # option_tokens=True needs 'options' among _needs_entity_tokens either
    # way, so both configs have identical parameter shapes/keys; loading
    # cheap's weights into encoded isolates the flag as the only difference.
    encoded.load_state_dict(cheap.state_dict())

    obs = _dummy_obs(structured_obs_spec, 3)
    inputs = [obs.get(key) for key in _STRUCTURED_IN_KEYS]
    _, option_repr_cheap = cheap.backbone(*inputs)
    _, option_repr_encoded = encoded.backbone(*inputs)
    assert not torch.allclose(option_repr_cheap, option_repr_encoded)


# ── replace_pooled (A4) ──────────────────────────────────────────────────────


def test_replace_pooled_drops_token_and_parameters(structured_obs_spec) -> None:
    """
    ``replace_pooled=True`` removes the pooled group's ``token_projections``
    entry and ``token_type_embedding`` row for every name in ``token_groups``,
    keeping the parameter count honest rather than duplicating capacity.
    """
    adapter = StructuredObsAdapter(obs_spec=structured_obs_spec, in_keys=_STRUCTURED_IN_KEYS)
    kept = TransformerBackbone(
        input_dim=adapter.out_features, out_features=32, adapter=adapter, num_heads=4,
        token_groups=["pokemon"], in_keys=_STRUCTURED_IN_KEYS,
    )
    dropped = TransformerBackbone(
        input_dim=adapter.out_features, out_features=32, adapter=adapter, num_heads=4,
        token_groups=["pokemon"], replace_pooled=True, in_keys=_STRUCTURED_IN_KEYS,
    )
    assert len(dropped.token_projections) == len(kept.token_projections) - 1
    assert dropped.token_type_embedding.shape[0] == kept.token_type_embedding.shape[0] - 1
    assert "pokemon" not in dropped.pooled_group_names
    assert "pokemon" in kept.pooled_group_names


def test_replace_pooled_rejects_config_that_can_be_fully_padded(structured_obs_spec) -> None:
    """
    Dropping every pooled token via ``replace_pooled``, with no ``options``
    stop slot to backstop it, can leave a row with zero valid tokens; this is
    rejected at construction rather than emitting NaN from the encoder later.
    """
    single_group_keys = [("observation", "pokemon")]
    adapter = StructuredObsAdapter(obs_spec=structured_obs_spec, in_keys=single_group_keys)
    with pytest.raises(ValueError, match="zero valid tokens"):
        TransformerBackbone(
            input_dim=adapter.out_features,
            out_features=32,
            adapter=adapter,
            num_heads=4,
            token_groups=["pokemon"],
            replace_pooled=True,
            in_keys=single_group_keys,
        )


@pytest.mark.parametrize(
    ("token_groups", "extra_kwargs"),
    [
        (["pokemon"], {}),
        (["my"], {}),
        (["options"], {"option_tokens": True, "encoded_option_repr": True}),
    ],
    ids=["pokemon", "my", "options"],
)
def test_replace_pooled_finite_on_empty_observation(
        token_groups, extra_kwargs, transformer_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    Every accepted ``replace_pooled`` configuration stays finite on a fully
    empty observation -- the case that could reach the all-padded-row NaN
    mechanism the construction-time check exists to prevent.
    """
    cfg = _transformer_cfg(
        transformer_model_cfg, token_groups=token_groups, replace_pooled=True, **extra_kwargs
    )
    actor_critic = build_actor_critic(cfg, structured_obs_spec, action_spec)
    out = actor_critic(structured_obs_spec.zero((2,)))
    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["state_value"]).all()


@pytest.mark.parametrize("pooling", ["mean", "cls", "attention"])
@pytest.mark.parametrize("replace_pooled", [False, True])
@pytest.mark.parametrize("encoded_option_repr", [False, True])
def test_transformer_stays_finite_across_toggle_combinations(
        pooling, replace_pooled, encoded_option_repr, transformer_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    Every combination of ``pooling`` x ``replace_pooled`` x
    ``encoded_option_repr`` produces finite logits and value, including when
    every requested entity group is fully padded (an empty observation).
    """
    # 'options' only joins token_groups when encoded_option_repr needs it
    # there (A3); with it False, 'options' + option_tokens there would hit
    # the wasteful-attention rejection instead of the combination under test.
    token_groups = ["pokemon", "options"] if encoded_option_repr else ["pokemon"]
    cfg = _transformer_cfg(
        transformer_model_cfg,
        pooling=pooling,
        token_groups=token_groups,
        option_tokens=True,
        replace_pooled=replace_pooled,
        encoded_option_repr=encoded_option_repr,
    )
    actor_critic = build_actor_critic(cfg, structured_obs_spec, action_spec)
    out = actor_critic(structured_obs_spec.zero((2,)))
    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["state_value"]).all()


# ── Construction-time validation (finding 4 / A5) ───────────────────────────


def test_transformer_rejects_unknown_token_group(structured_obs_spec) -> None:
    """
    A ``token_groups`` name absent from the adapter's registered groups
    raises at construction, not on first forward (finding 4).
    """
    adapter = StructuredObsAdapter(obs_spec=structured_obs_spec, in_keys=_STRUCTURED_IN_KEYS)
    with pytest.raises(ValueError, match="not among the adapter's registered groups"):
        TransformerBackbone(
            input_dim=adapter.out_features, out_features=32, adapter=adapter, num_heads=4,
            token_groups=["not_a_group"], in_keys=_STRUCTURED_IN_KEYS,
        )


def test_transformer_rejects_scalar_token_group(structured_obs_spec) -> None:
    """
    ``globals``/``select_cats`` have no entity axis and are rejected from
    ``token_groups`` at construction.
    """
    adapter = StructuredObsAdapter(obs_spec=structured_obs_spec, in_keys=_STRUCTURED_IN_KEYS)
    with pytest.raises(ValueError, match="no entity axis"):
        TransformerBackbone(
            input_dim=adapter.out_features, out_features=32, adapter=adapter, num_heads=4,
            token_groups=["globals"], in_keys=_STRUCTURED_IN_KEYS,
        )


def test_transformer_rejects_duplicate_token_group(structured_obs_spec) -> None:
    """
    A repeated ``token_groups`` name is rejected at construction.

    Left through, it would concatenate that group's entity block into the
    sequence twice — doubling its weight in the readout — and the recorded
    option offset would point at the last occurrence only. Both failures are
    silent, which is what makes rejecting cheaper than allowing.
    """
    adapter = StructuredObsAdapter(obs_spec=structured_obs_spec, in_keys=_STRUCTURED_IN_KEYS)
    with pytest.raises(ValueError, match=r"token_groups repeats \['pokemon'\]"):
        TransformerBackbone(
            input_dim=adapter.out_features, out_features=32, adapter=adapter, num_heads=4,
            token_groups=["pokemon", "pokemon"], in_keys=_STRUCTURED_IN_KEYS,
        )


def test_transformer_rejects_wasteful_option_attention_combo(structured_obs_spec) -> None:
    """
    ``options`` in ``token_groups`` with ``option_tokens=True`` and
    ``encoded_option_repr=False`` pays for the option tokens' attention pass
    and then discards the result; rejected at construction.
    """
    adapter = StructuredObsAdapter(obs_spec=structured_obs_spec, in_keys=_STRUCTURED_IN_KEYS)
    with pytest.raises(ValueError, match="encoded_option_repr"):
        TransformerBackbone(
            input_dim=adapter.out_features, out_features=32, adapter=adapter, num_heads=4,
            token_groups=["options"], option_tokens=True, in_keys=_STRUCTURED_IN_KEYS,
        )


# ── MLPBackbone.option_tokens (A6) ──────────────────────────────────────────


def test_mlp_option_tokens_trains_option_projection(
        structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    ``MLPBackbone(option_tokens=True)`` pairs with the pointer head and
    backprop reaches the option projection (``ptr_mlp_pointer``).
    """
    cfg = cast(
        DictConfig,
        OmegaConf.merge(
            structured_model_cfg,
            {
                "model": {
                    "backbone": {"option_tokens": True},
                    "head": {"_target_": "src.models.heads.PointerPolicyHead"},
                }
            },
        ),
    )
    actor_critic = build_actor_critic(cfg, structured_obs_spec, action_spec)
    assert actor_critic.backbone.produces_option_repr

    out = actor_critic(_dummy_obs(structured_obs_spec, 4))
    assert out["logits"].shape == (4, N_ACTIONS)
    out["logits"].sum().backward()

    backbone = cast(MLPBackbone, actor_critic.backbone)
    projection = next(backbone.option_projection.parameters())
    assert projection.grad is not None and torch.any(projection.grad != 0)


# ── Checkpoint compatibility (A4 hard requirement) ──────────────────────────


def test_reference_arm_state_dict_loads_strictly(
        transformer_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    A reference-arm config (``token_groups=[]``, ``option_tokens=false``)
    still loads strictly: A2-A4's new modules only appear once their flags
    are set, so the parameter set is unchanged from before this work.
    """
    reference = build_actor_critic(transformer_model_cfg, structured_obs_spec, action_spec)
    fresh = build_actor_critic(transformer_model_cfg, structured_obs_spec, action_spec)
    fresh.load_state_dict(reference.state_dict(), strict=True)

    backbone = cast(TransformerBackbone, reference.backbone)
    assert len(backbone.token_projections) == len(_STRUCTURED_IN_KEYS)
    assert len(backbone.entity_projections) == 0
    assert len(backbone.entity_segment_embedding) == 0
