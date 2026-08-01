from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from src.env.battle_handle import BattleHandle
from src.env.deck import load_deck
from src.env.random_opponent import RandomOpponent
from src.env.structured_observation_encoder import (
    StructuredObservationEncoder as TrainingEncoder,
)
from src.policies.ppo_actor import build_actor_critic
from submission.cg_api import to_observation_class
from submission.runtime import Policy, StructuredObservationEncoder
from tests.conftest import DECK_PATH, MAX_OPTIONS


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

    assert observation.select is not None
    assert observation.select.minCount <= len(picks) <= observation.select.maxCount
    assert len(picks) == len(set(picks))
    assert all(0 <= pick < len(observation.select.option) for pick in picks)
    if action_selection == "greedy":
        assert sample_calls == 0
    else:
        assert sample_calls > 0


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
