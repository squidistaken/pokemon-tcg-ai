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


def test_padding_card_id_encodes_to_zero(adapter, structured_obs_spec) -> None:
    """
    ID 0 (none/padding) contributes a zero card representation, so an
    all-empty observation encodes zone summaries as zeros.
    """
    obs = _zero_obs(structured_obs_spec, batch=1)
    encoded = adapter(*_adapter_inputs(obs))
    assert torch.all(torch.isfinite(encoded))
    card_zero = adapter._card_repr(torch.zeros(1, dtype=torch.int64))  # noqa: SLF001
    assert torch.all(card_zero == 0.0)


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
    actor_critic = build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec)
    backbone = actor_critic.backbone
    assert isinstance(backbone.adapter, StructuredObsAdapter)
    assert backbone.input_dim == backbone.adapter.out_features

    obs = _zero_obs(structured_obs_spec, batch=4)
    out = actor_critic(obs)
    assert out["logits"].shape == (4, N_ACTIONS)
    assert out["state_value"].shape == (4, 1)
