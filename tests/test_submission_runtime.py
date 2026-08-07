from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from cg.api import Observation
from src.env.battle_handle import BattleHandle
from src.env.deck import load_deck
from src.env.random_opponent import RandomOpponent
from src.env.structured_observation_encoder import (
    StructuredObservationEncoder as TrainingEncoder,
)
from src.models.transformer import TransformerBackbone as TrainingTransformerBackbone
from src.policies.ppo_actor import build_actor_critic
from submission.cg_api import to_observation_class
from submission.runtime import Policy, StructuredObservationEncoder
from submission.runtime import TransformerBackbone as PortableTransformerBackbone
from tests.conftest import DECK_PATH, MAX_OPTIONS

_LINEAR_HEAD_TARGET = "src.models.heads.LinearPolicyHead"
_POINTER_HEAD_TARGET = "src.models.heads.PointerPolicyHead"
_FIXTURES_PATH = Path(__file__).parent / "fixtures" / "observations.pt"


def _assert_nested_equal(expected: Any, actual: Any) -> None:
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_nested_equal(expected[key], actual[key])
        return
    assert isinstance(expected, torch.Tensor)
    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def _portable_config(
    model_config: Any,
    action_selection: str = "sample",
) -> dict[str, Any]:
    return {
        "model": OmegaConf.to_container(model_config.model, resolve=True),
        "env": {"encoder": "structured", "max_options": MAX_OPTIONS},
        "inference": {"action_selection": action_selection},
    }


def _transformer_cfg(
    base: DictConfig,
    head_target: str = _LINEAR_HEAD_TARGET,
    **backbone: object,
) -> DictConfig:
    """
    Copy the ``transformer_model_cfg`` fixture with backbone/head overridden.

    Mirrors ``tests/test_models.py``'s helper of the same name, so these cases
    track the same config space the training-side transformer tests exercise.

    :param base: The ``transformer_model_cfg`` fixture.
    :param head_target: Policy head ``_target_`` to select.
    :param backbone: Keys to set under ``model.backbone``.
    :return: A merged copy; the fixture is left untouched.
    """
    cfg = cast(DictConfig, OmegaConf.merge(base, {"model": {"backbone": backbone}}))
    return cast(
        DictConfig,
        OmegaConf.merge(cfg, {"model": {"head": {"_target_": head_target}}}),
    )


def test_torch_only_encoder_matches_training_encoder() -> None:
    """Raw engine observations produce exactly the training tensors."""
    deck = load_deck(DECK_PATH)
    handle = BattleHandle()
    opponent = RandomOpponent(seed=19)
    training_encoder = TrainingEncoder(max_options=MAX_OPTIONS)
    portable_encoder = StructuredObservationEncoder(max_options=MAX_OPTIONS)
    observation = handle.start(deck, deck)
    compared = 0
    try:
        for _ in range(100):
            state = observation.current
            if state is None or state.result != -1:
                break
            expected = training_encoder.encode(
                observation, state.yourIndex, 0
            ).to_dict()
            portable_observation = to_observation_class(asdict(observation))
            actual = portable_encoder.encode(portable_observation, state.yourIndex, 0)
            _assert_nested_equal(expected, actual)
            compared += 1
            observation = handle.select(opponent(observation))
    finally:
        handle.finish()
    assert compared > 0


def test_torch_only_model_matches_training_logits(
    structured_model_cfg,
    structured_obs_spec,
    action_spec,
) -> None:
    """The portable modules strictly load and preserve checkpoint logits."""
    torch.manual_seed(23)
    actor_critic = build_actor_critic(
        structured_model_cfg, structured_obs_spec, action_spec
    ).eval()
    payload = {"state_dict": actor_critic.state_dict()}
    policy = Policy(payload, _portable_config(structured_model_cfg))
    fixtures = torch.load(
        Path(__file__).parent / "fixtures" / "observations.pt",
        weights_only=False,
    )

    for case_name in fixtures.keys():  # noqa: SIM118 - TensorDict iteration differs.
        case = fixtures[case_name]
        expected = actor_critic.policy_logits(case)
        actual = policy.model.policy_logits(case["observation"].to_dict())
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


#: One entry per covered transformer axis, plus a couple of realistic
#: combinations. ``legacy_defaults_mean_pooling`` overrides nothing at all:
#: the ``transformer_model_cfg`` fixture already omits ``pooling``,
#: ``norm_first``, ``final_norm``, ``token_groups``, ``option_tokens``,
#: ``replace_pooled`` and ``encoded_option_repr`` entirely, exactly like an
#: older checkpoint's embedded config -- this case doubles as coverage that
#: the portable runtime's defaults for those keys equal the training
#: constructor's defaults, since that is the only way it could match here.
#: ``zone_entity_tokens`` uses ``my`` (a multi-zone group) rather than
#: ``pokemon``, since only ``pokemon`` is exercised by any real registered
#: checkpoint.
_TRANSFORMER_LOGIT_CASES: dict[str, tuple[dict[str, object], str]] = {
    "legacy_defaults_mean_pooling": ({}, _LINEAR_HEAD_TARGET),
    "cls_pooling": ({"pooling": "cls"}, _LINEAR_HEAD_TARGET),
    "attention_pooling": ({"pooling": "attention"}, _LINEAR_HEAD_TARGET),
    "pokemon_entity_tokens": ({"token_groups": ["pokemon"]}, _LINEAR_HEAD_TARGET),
    "zone_entity_tokens": ({"token_groups": ["my"]}, _LINEAR_HEAD_TARGET),
    "option_tokens_cheap_projection": (
        {"option_tokens": True},
        _POINTER_HEAD_TARGET,
    ),
    "option_tokens_encoded_attended": (
        {
            "option_tokens": True,
            "token_groups": ["options"],
            "encoded_option_repr": True,
        },
        _POINTER_HEAD_TARGET,
    ),
    "replace_pooled_pokemon": (
        {"token_groups": ["pokemon"], "replace_pooled": True},
        _LINEAR_HEAD_TARGET,
    ),
    "pre_ln_final_norm": (
        {"norm_first": True, "final_norm": True},
        _LINEAR_HEAD_TARGET,
    ),
    "two_stacked_layers": ({"num_layers": 2}, _LINEAR_HEAD_TARGET),
    "tf_combined": (
        {
            "pooling": "cls",
            "token_groups": ["pokemon"],
            "option_tokens": True,
            "norm_first": True,
            "final_norm": True,
            "num_layers": 2,
        },
        _POINTER_HEAD_TARGET,
    ),
}


@pytest.mark.parametrize(
    ("backbone_overrides", "head_target"),
    _TRANSFORMER_LOGIT_CASES.values(),
    ids=list(_TRANSFORMER_LOGIT_CASES),
)
def test_torch_only_transformer_matches_training_logits(
    backbone_overrides: dict[str, object],
    head_target: str,
    transformer_model_cfg: DictConfig,
    structured_obs_spec,
    action_spec,
) -> None:
    """The portable transformer backbone strictly loads and preserves logits
    across pooling modes, per-entity token groups, the pointer head, and
    normalization/depth variants -- the config axes real checkpoints do not
    all individually exercise.
    """
    torch.manual_seed(23)
    cfg = _transformer_cfg(
        transformer_model_cfg, head_target=head_target, **backbone_overrides
    )
    actor_critic = build_actor_critic(cfg, structured_obs_spec, action_spec).eval()
    payload = {"state_dict": actor_critic.state_dict()}
    policy = Policy(payload, _portable_config(cfg))
    fixtures = torch.load(_FIXTURES_PATH, weights_only=False)

    for case_name in fixtures.keys():  # noqa: SIM118 - TensorDict iteration differs.
        case = fixtures[case_name]
        expected = actor_critic.policy_logits(case)
        actual = policy.model.policy_logits(case["observation"].to_dict())
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("backbone", ["mlp", "transformer"])
def test_seat_split_adapter_matches_training_logits(
    backbone: str,
    structured_model_cfg: DictConfig,
    transformer_model_cfg: DictConfig,
    structured_obs_spec,
    action_spec,
) -> None:
    """
    ``adapter.pokemon_seat_split`` widens the ``pokemon`` group, which resizes
    the first projection of *both* trunks, so the portable runtime has to read
    the flag out of the embedded config to rebuild the same shapes. Covered on
    each backbone because they consume that width through different modules
    (``MLPBackbone``'s input layer, the transformer's ``token_projections``).
    """
    torch.manual_seed(23)
    base = structured_model_cfg if backbone == "mlp" else transformer_model_cfg
    cfg = cast(
        DictConfig,
        OmegaConf.merge(base, {"model": {"adapter": {"pokemon_seat_split": True}}}),
    )
    actor_critic = build_actor_critic(cfg, structured_obs_spec, action_spec).eval()
    policy = Policy({"state_dict": actor_critic.state_dict()}, _portable_config(cfg))
    fixtures = torch.load(_FIXTURES_PATH, weights_only=False)

    for case_name in fixtures.keys():  # noqa: SIM118 - TensorDict iteration differs.
        case = fixtures[case_name]
        expected = actor_critic.policy_logits(case)
        actual = policy.model.policy_logits(case["observation"].to_dict())
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_transformer_strict_loads_checkpoint_predating_segment_embeddings(
    transformer_model_cfg: DictConfig,
    structured_obs_spec,
    action_spec,
) -> None:
    """A checkpoint saved before ``entity_segment_embedding`` existed has no
    ``backbone.entity_segment_embedding.*`` keys at all (real registered
    checkpoints ``01f06a832eba``/``3ecc959f3759`` are exactly this case). The
    portable model must still strict-load it and add no segment identity --
    i.e. match a training model with those embeddings pinned to zero.
    """
    torch.manual_seed(31)
    cfg = _transformer_cfg(transformer_model_cfg, token_groups=["pokemon"])
    actor_critic = build_actor_critic(cfg, structured_obs_spec, action_spec).eval()
    backbone = cast(TrainingTransformerBackbone, actor_critic.backbone)
    with torch.no_grad():
        for parameter in backbone.entity_segment_embedding.values():
            parameter.zero_()
    state_dict = dict(actor_critic.state_dict())
    segment_keys = [
        key
        for key in state_dict
        if key.startswith("backbone.entity_segment_embedding.")
    ]
    assert segment_keys  # sanity: this checkpoint actually has some to drop.
    for key in segment_keys:
        del state_dict[key]

    policy = Policy({"state_dict": state_dict}, _portable_config(cfg))
    portable_backbone = cast(PortableTransformerBackbone, policy.model.backbone)
    assert len(portable_backbone.entity_segment_embedding) == 0

    fixtures = torch.load(_FIXTURES_PATH, weights_only=False)
    for case_name in fixtures.keys():  # noqa: SIM118 - TensorDict iteration differs.
        case = fixtures[case_name]
        expected = actor_critic.policy_logits(case)
        actual = policy.model.policy_logits(case["observation"].to_dict())
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_pointer_head_without_option_tokens_raises_clear_error(
    transformer_model_cfg: DictConfig,
    structured_obs_spec,
    action_spec,
) -> None:
    """Mirrors ``build_actor_critic``'s own check: the portable runtime
    rejects pairing a pointer head with a trunk that emits no per-option
    tokens, rather than failing obscurely inside the head's forward.
    """
    torch.manual_seed(23)
    actor_critic = build_actor_critic(
        transformer_model_cfg, structured_obs_spec, action_spec
    ).eval()
    config = _portable_config(transformer_model_cfg)
    config["model"]["head"]["_target_"] = _POINTER_HEAD_TARGET

    with pytest.raises(ValueError, match="needs per-option tokens"):
        Policy({"state_dict": actor_critic.state_dict()}, config)


def test_policy_defaults_to_sampling_without_inference_config(
    structured_model_cfg,
    structured_obs_spec,
    action_spec,
) -> None:
    """Older model configs sample by default when no serving mode is embedded."""
    actor_critic = build_actor_critic(
        structured_model_cfg, structured_obs_spec, action_spec
    )
    config = _portable_config(structured_model_cfg)
    del config["inference"]

    policy = Policy({"state_dict": actor_critic.state_dict()}, config)

    assert policy.action_selection == "sample"


def _run_serving_loop(
    policy: Policy, monkeypatch: pytest.MonkeyPatch
) -> tuple[Observation, list[int], int]:
    """Run one engine game through ``policy``, counting sampling calls.

    :return: The final raw observation, the picks made, and how many times
        ``torch.multinomial`` was called (0 in greedy mode).
    """
    sample_calls = 0
    original_multinomial = torch.multinomial

    def recorded_multinomial(
        probabilities: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        nonlocal sample_calls
        sample_calls += 1
        return original_multinomial(probabilities, num_samples)

    monkeypatch.setattr(torch, "multinomial", recorded_multinomial)
    deck = load_deck(DECK_PATH)
    handle = BattleHandle()
    try:
        observation = handle.start(deck, deck)
        portable_observation = to_observation_class(asdict(observation))
        picks = policy(portable_observation)
    finally:
        handle.finish()
    return observation, picks, sample_calls


def _assert_legal_sequential_selection(
    observation: Observation,
    picks: list[int],
    sample_calls: int,
    action_selection: str,
) -> None:
    """Assert ``picks`` respects the selection's bounds, masks, and mode."""
    assert observation.select is not None
    assert observation.select.minCount <= len(picks) <= observation.select.maxCount
    assert len(picks) == len(set(picks))
    assert all(0 <= pick < len(observation.select.option) for pick in picks)
    if action_selection == "greedy":
        assert sample_calls == 0
    else:
        assert sample_calls > 0


@pytest.mark.parametrize("action_selection", ["greedy", "sample"])
def test_inference_modes_produce_legal_sequential_selection(
    action_selection: str,
    structured_model_cfg,
    structured_obs_spec,
    action_spec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both serving modes respect bounds, masks, and sampling implementation."""
    torch.manual_seed(29)
    actor_critic = build_actor_critic(
        structured_model_cfg, structured_obs_spec, action_spec
    ).eval()
    policy = Policy(
        {"state_dict": actor_critic.state_dict()},
        _portable_config(structured_model_cfg, action_selection),
    )
    observation, picks, sample_calls = _run_serving_loop(policy, monkeypatch)
    _assert_legal_sequential_selection(
        observation, picks, sample_calls, action_selection
    )


@pytest.mark.parametrize("action_selection", ["greedy", "sample"])
def test_transformer_pointer_inference_modes_produce_legal_sequential_selection(
    action_selection: str,
    transformer_model_cfg: DictConfig,
    structured_obs_spec,
    action_spec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pointer head's serving loop re-encodes per pick, so ``option_repr``
    is rebuilt from scratch before every choice rather than produced once by
    a single ``policy_logits`` call -- unlike the equivalence tests above,
    this exercises that per-pick re-encoding end to end through the engine.
    """
    torch.manual_seed(29)
    cfg = _transformer_cfg(
        transformer_model_cfg, option_tokens=True, head_target=_POINTER_HEAD_TARGET
    )
    actor_critic = build_actor_critic(cfg, structured_obs_spec, action_spec).eval()
    policy = Policy(
        {"state_dict": actor_critic.state_dict()},
        _portable_config(cfg, action_selection),
    )
    observation, picks, sample_calls = _run_serving_loop(policy, monkeypatch)
    _assert_legal_sequential_selection(
        observation, picks, sample_calls, action_selection
    )


def test_policy_rejects_unknown_action_selection(
    structured_model_cfg,
    structured_obs_spec,
    action_spec,
) -> None:
    """Malformed bundled selection modes fail while loading, before a match."""
    actor_critic = build_actor_critic(
        structured_model_cfg, structured_obs_spec, action_spec
    )
    config = _portable_config(structured_model_cfg, "unknown")

    with pytest.raises(ValueError, match="action_selection"):
        Policy({"state_dict": actor_critic.state_dict()}, config)
