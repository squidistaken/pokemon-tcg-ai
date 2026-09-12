import pytest
import torch
from tensordict import TensorDict

from src.models.structured_obs_adapter import StructuredObsAdapter
from src.policies.ppo_actor import DEFAULT_IN_KEYS, build_actor_critic
from tests.conftest import N_ACTIONS


@pytest.fixture
def adapter(structured_obs_spec) -> StructuredObsAdapter:
    """
    Adapter over the full structured observation.

    :param structured_obs_spec: Env spec fixture (observation + action mask).
    :return: Adapter with default embedding widths.
    """
    return StructuredObsAdapter(obs_spec=structured_obs_spec, in_keys=DEFAULT_IN_KEYS)


def _zero_obs(structured_obs_spec, batch: int) -> TensorDict:
    """
    Build an all-zero structured observation batch.

    :param structured_obs_spec: Env spec fixture (observation + action mask).
    :param batch: Batch size.
    :return: Zeroed TensorDict matching the spec.
    """
    return structured_obs_spec.zero((batch,))


def _adapter_inputs(observation: TensorDict) -> list:
    """
    Pull the adapter's positional inputs out of an observation tensordict.

    :param observation: TensorDict holding the nested ``observation`` groups.
    :return: One value per ``DEFAULT_IN_KEYS``, in order.
    """
    return [observation.get(key) for key in DEFAULT_IN_KEYS]


def test_forward_width_matches_out_features(adapter, structured_obs_spec) -> None:
    """
    The forward output width equals the advertised ``out_features``.
    """
    obs = _zero_obs(structured_obs_spec, batch=3)
    encoded = adapter(*_adapter_inputs(obs))
    assert encoded.shape == (3, adapter.out_features)
    assert encoded.dtype == torch.float32


def test_forward_handles_extra_batch_dims(adapter, structured_obs_spec) -> None:
    """
    Leading (time, batch) dimensions pass through unchanged.
    """
    obs = structured_obs_spec.zero((2, 5))
    encoded = adapter(*_adapter_inputs(obs))
    assert encoded.shape == (2, 5, adapter.out_features)


def test_globals_are_scaled(adapter, structured_obs_spec) -> None:
    """
    Raw global counts come out divided by their fixed scale constants.
    """
    obs = _zero_obs(structured_obs_spec, batch=1)
    obs[("observation", "globals")][0, 0] = 50.0
    encoded = adapter(*_adapter_inputs(obs))
    assert encoded[0, 0].item() == pytest.approx(1.0)


def test_zone_pooling_is_permutation_invariant(adapter, structured_obs_spec) -> None:
    """
    Reordering a discard pile leaves the encoding unchanged, while changing
    its contents does not.
    """
    obs_a = _zero_obs(structured_obs_spec, batch=1)
    obs_b = _zero_obs(structured_obs_spec, batch=1)
    obs_c = _zero_obs(structured_obs_spec, batch=1)
    for obs, ids in ((obs_a, [3, 7, 11]), (obs_b, [11, 3, 7]), (obs_c, [3, 7, 12])):
        obs[("observation", "my", "discard_ids")][0, : len(ids)] = torch.tensor(ids)
        obs[("observation", "my", "discard_mask")][0, : len(ids)] = True
    encoded_a = adapter(*_adapter_inputs(obs_a))
    encoded_b = adapter(*_adapter_inputs(obs_b))
    encoded_c = adapter(*_adapter_inputs(obs_c))
    assert torch.allclose(encoded_a, encoded_b)
    assert not torch.allclose(encoded_a, encoded_c)


def test_card_categories_reach_the_representation(adapter) -> None:
    """
    Regression test: card type/energy type/weakness/resistance were computed
    by ``CardDatabase`` but never read, so mutating them used to have no
    effect on ``_card_repr``. It must now.
    """
    card_id = torch.tensor([5])
    before = adapter._card_repr(card_id).clone()  # noqa: SLF001
    adapter._card_categories[5] = (adapter._card_categories[5] + 1) % 60  # noqa: SLF001
    after = adapter._card_repr(card_id)  # noqa: SLF001
    assert not torch.allclose(before, after)


def test_card_attacks_reach_the_representation(adapter) -> None:
    """
    Regression test: a card's usable attacks were computed by
    ``CardDatabase`` but never pooled into its representation, so mutating
    them used to have no effect on ``_card_repr``. It must now.
    """
    card_id = torch.tensor([5])
    before = adapter._card_repr(card_id).clone()  # noqa: SLF001
    current = adapter._card_attack_ids[5, 0].item()  # noqa: SLF001
    adapter._card_attack_ids[5, 0] = 1 if current != 1 else 2  # noqa: SLF001
    after = adapter._card_repr(card_id)  # noqa: SLF001
    assert not torch.allclose(before, after)


def test_padding_card_id_encodes_to_constant(adapter, structured_obs_spec) -> None:
    """
    ID 0 (none/padding) contributes the same fixed representation on every
    call, block by block: zero learned embedding, zero static features, and
    zero pooled attack representation (all three reserve row 0 for "absent",
    and card 0 has no attacks to pool), except the categorical block, whose
    "absent" card type/energy type/weakness/resistance is a constant but not
    necessarily zero vector, same as ``select_cats``/``options.cats``.
    Padding slots still drop out of masked pooling via the validity mask,
    not via this being zero.
    """
    obs = _zero_obs(structured_obs_spec, batch=1)
    encoded = adapter(*_adapter_inputs(obs))
    assert torch.all(torch.isfinite(encoded))
    card_zero_a = adapter._card_repr(torch.zeros(1, dtype=torch.int64))  # noqa: SLF001
    card_zero_b = adapter._card_repr(torch.zeros(1, dtype=torch.int64))  # noqa: SLF001
    assert torch.equal(card_zero_a, card_zero_b)

    embed_dim = adapter._card_embedding.embedding_dim  # noqa: SLF001
    static_dim = adapter._card_static.shape[-1]  # noqa: SLF001
    cat_dim = adapter.CARD_CATEGORY_FIELD_COUNT * adapter._category_embed_dim  # noqa: SLF001
    embed_block = card_zero_a[..., :embed_dim]
    static_block = card_zero_a[..., embed_dim : embed_dim + static_dim]
    attack_block = card_zero_a[..., embed_dim + static_dim + cat_dim :]
    assert torch.all(embed_block == 0.0)
    assert torch.all(static_block == 0.0)
    assert torch.all(attack_block == 0.0)
    assert attack_block.shape[-1] == adapter._attack_repr_dim  # noqa: SLF001


def test_gradients_reach_card_embedding(adapter, structured_obs_spec) -> None:
    """
    Backprop through the encoding reaches the learned card embedding.
    """
    obs = _zero_obs(structured_obs_spec, batch=2)
    obs[("observation", "my", "hand_ids")][:, 0] = 5
    obs[("observation", "my", "hand_mask")][:, 0] = True
    adapter.zero_grad()
    adapter(*_adapter_inputs(obs)).sum().backward()
    grad = adapter._card_embedding.weight.grad  # noqa: SLF001
    assert grad is not None and torch.any(grad[5] != 0)


def test_unknown_group_rejected(structured_obs_spec) -> None:
    """
    An in-key naming something other than an encoder group is rejected.
    """
    with pytest.raises(ValueError, match="Unknown structured observation group"):
        StructuredObsAdapter(obs_spec=structured_obs_spec, in_keys=["action_mask"])


def test_build_actor_critic_attaches_adapter(
    structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    Building against the structured spec wires an adapter into the backbone
    and sizes the MLP from its output width, and a forward pass produces
    logits and a value of the right shapes.
    """
    actor_critic = build_actor_critic(
        structured_model_cfg, structured_obs_spec, action_spec
    )
    backbone = actor_critic.backbone
    assert isinstance(backbone.adapter, StructuredObsAdapter)
    assert backbone.input_dim == backbone.adapter.out_features

    obs = _zero_obs(structured_obs_spec, batch=4)
    out = actor_critic(obs)
    assert out["logits"].shape == (4, N_ACTIONS)
    assert out["state_value"].shape == (4, 1)


# ── Option validity (finding 8 / A1) ────────────────────────────────────────


def test_yes_no_option_without_card_id_is_valid(adapter, structured_obs_spec) -> None:
    """
    A yes/no/end-turn option has ``cats[..., 0]`` set but ``card_id`` stays 0;
    the pooled ``options`` vector must not be all-zero and the entity token
    must be marked valid (finding 8: ``card_id``-based validity treated this
    as padding, so a yes/no/end-turn-only selection pooled to exactly zero).
    """
    obs = _zero_obs(structured_obs_spec, batch=1)
    obs[("observation", "options", "cats")][0, 0, 0] = 1
    inputs = _adapter_inputs(obs)

    options_index = DEFAULT_IN_KEYS.index(("observation", "options"))
    pooled_options = adapter.encode_groups(*inputs)[options_index]
    assert torch.any(pooled_options != 0.0)

    tokens = adapter.encode_entity_tokens(*inputs, groups=["options"])
    _, validity = tokens["options"]
    assert bool(validity[0, 0])


def test_stop_slot_is_always_valid_in_option_tokens(
    adapter, structured_obs_spec
) -> None:
    """
    The synthetic stop slot (the option table's last row) is valid even with
    zero real options, so a pointer head always has at least one legal token
    to attend to.
    """
    obs = _zero_obs(structured_obs_spec, batch=2)
    tokens = adapter.encode_entity_tokens(*_adapter_inputs(obs), groups=["options"])
    _, validity = tokens["options"]
    assert torch.all(validity[:, -1])
    assert not torch.any(validity[:, :-1])  # no real option was set


# ── Segment identity (finding 1 / A2) ───────────────────────────────────────


def test_group_segment_ids_seat_zone_and_stop_slot(
    adapter, structured_obs_spec
) -> None:
    """
    Pins the exact segment id layout the class docstring promises: seat for
    ``pokemon``, one id per zone for ``my``, and real-vs-stop for ``options``.
    """
    segment_ids = adapter.group_segment_ids

    rows = structured_obs_spec[("observation", "pokemon", "card_id")].shape[-1]
    pokemon_segments = segment_ids["pokemon"]
    assert pokemon_segments.shape == (rows,)
    assert torch.all(pokemon_segments[: rows // 2] == 0)
    assert torch.all(pokemon_segments[rows // 2 :] == 1)

    hand_cap = structured_obs_spec[("observation", "my", "hand_ids")].shape[-1]
    discard_cap = structured_obs_spec[("observation", "my", "discard_ids")].shape[-1]
    my_segments = segment_ids["my"]
    assert torch.all(my_segments[:hand_cap] == 0)
    assert torch.all(my_segments[hand_cap : hand_cap + discard_cap] == 1)
    assert torch.all(my_segments[hand_cap + discard_cap :] == 2)

    option_segments = segment_ids["options"]
    assert torch.all(option_segments[:-1] == 0)
    assert option_segments[-1] == 1

    # globals/select_cats have no entity axis and are absent from the dict,
    # unlike group_slot_counts, which records 1 for them.
    assert "globals" not in segment_ids
    assert "select_cats" not in segment_ids


def _seat_swapped(observation: TensorDict) -> TensorDict:
    """
    Swap the two seats' halves of the ``pokemon`` table.

    The row layout is agent active + bench then opponent active + bench, so
    exchanging the halves is exactly "the same board, other way round".

    :param observation: Observation tensordict to copy and swap.
    :return: A copy whose two seats' board rows are exchanged.
    """
    swapped = observation.clone()
    pokemon = swapped.get(("observation", "pokemon"))
    half = pokemon["card_id"].shape[-1] // 2
    for leaf in (
        "card_id",
        "tool_id",
        "energy_card_ids",
        "pre_evolution_ids",
        "features",
        "mask",
    ):
        rows = pokemon[leaf]
        pokemon[leaf] = torch.cat([rows[:, half:], rows[:, :half]], dim=1)
    return swapped


def _board_observation(structured_obs_spec) -> TensorDict:
    """
    An observation with a different Pokemon on each seat's active slot.

    :param structured_obs_spec: Env spec fixture (observation + action mask).
    :return: Zeroed observation with both actives occupied by distinct cards.
    """
    obs = _zero_obs(structured_obs_spec, batch=1)
    pokemon = obs.get(("observation", "pokemon"))
    half = pokemon["card_id"].shape[-1] // 2
    pokemon["card_id"][0, 0] = 5
    pokemon["mask"][0, 0] = True
    pokemon["card_id"][0, half] = 9
    pokemon["mask"][0, half] = True
    return obs


def test_pooled_pokemon_is_seat_blind_without_the_split(structured_obs_spec) -> None:
    """
    The bug the seat split exists for: with one pool over all board rows,
    swapping the two seats leaves the encoding bit-identical, so no backbone
    reading the pooled path can tell its own board from the opponent's.
    """
    adapter = StructuredObsAdapter(
        obs_spec=structured_obs_spec, in_keys=DEFAULT_IN_KEYS, pokemon_seat_split=False
    )
    obs = _board_observation(structured_obs_spec)
    encoded = adapter(*_adapter_inputs(obs))
    swapped = adapter(*_adapter_inputs(_seat_swapped(obs)))
    assert torch.equal(encoded, swapped)


def test_seat_split_distinguishes_the_two_boards(structured_obs_spec) -> None:
    """
    With the split on, the same swap moves the encoding — and the two halves
    of the ``pokemon`` block are exchanged rather than arbitrarily different,
    which is what "pooled per seat" means.
    """
    adapter = StructuredObsAdapter(
        obs_spec=structured_obs_spec, in_keys=DEFAULT_IN_KEYS, pokemon_seat_split=True
    )
    obs = _board_observation(structured_obs_spec)
    encoded = adapter.encode_groups(*_adapter_inputs(obs))
    swapped = adapter.encode_groups(*_adapter_inputs(_seat_swapped(obs)))
    index = adapter.group_names.index("pokemon")
    board, swapped_board = encoded[index], swapped[index]
    assert not torch.allclose(board, swapped_board)
    half = board.shape[-1] // 2
    assert torch.allclose(board[..., :half], swapped_board[..., half:])
    assert torch.allclose(board[..., half:], swapped_board[..., :half])


def test_seat_split_widens_only_the_pokemon_group(structured_obs_spec) -> None:
    """
    The split doubles the ``pokemon`` group's width and leaves every other
    group — and so every other backbone projection — untouched.
    """
    pooled = StructuredObsAdapter(
        obs_spec=structured_obs_spec, in_keys=DEFAULT_IN_KEYS, pokemon_seat_split=False
    )
    split = StructuredObsAdapter(
        obs_spec=structured_obs_spec, in_keys=DEFAULT_IN_KEYS, pokemon_seat_split=True
    )
    index = pooled.group_names.index("pokemon")
    for position, (narrow, wide) in enumerate(
        zip(pooled.group_feature_widths, split.group_feature_widths, strict=True)
    ):
        assert wide == (2 * narrow if position == index else narrow)
    assert (
        split.out_features == pooled.out_features + pooled.group_feature_widths[index]
    )
    # The per-entity token path is unaffected: same slots, same segment ids.
    assert split.group_slot_counts == pooled.group_slot_counts
    assert torch.equal(
        split.group_segment_ids["pokemon"], pooled.group_segment_ids["pokemon"]
    )


def test_seat_split_preserves_bench_permutation_invariance(structured_obs_spec) -> None:
    """
    Pooling still happens *within* a seat, so bench order stays irrelevant —
    the property the single pool had and the split must not cost.
    """
    adapter = StructuredObsAdapter(
        obs_spec=structured_obs_spec, in_keys=DEFAULT_IN_KEYS, pokemon_seat_split=True
    )
    obs = _zero_obs(structured_obs_spec, batch=1)
    pokemon = obs.get(("observation", "pokemon"))
    pokemon["card_id"][0, 1:4] = torch.tensor([3, 7, 11])
    pokemon["mask"][0, 1:4] = True
    reordered = obs.clone()
    reordered_pokemon = reordered.get(("observation", "pokemon"))
    reordered_pokemon["card_id"][0, 1:4] = torch.tensor([11, 3, 7])
    assert torch.allclose(
        adapter(*_adapter_inputs(obs)), adapter(*_adapter_inputs(reordered))
    )
