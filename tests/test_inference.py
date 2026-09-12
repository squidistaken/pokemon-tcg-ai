from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch import nn
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import ActionMask

from src.env.battle_handle import BattleHandle
from src.env.decks.deck import load_deck
from src.env.observation.structured_observation_encoder import (
    StructuredObservationEncoder,
)
from src.env.tcg_env import TCGEnv
from src.policies.greedy_policy_opponent import save_actor_critic
from src.policies.inference import (
    InferenceAgent,
    build_inference_specs,
    load_inference_agent,
)
from src.policies.ppo_actor import build_actor_critic
from src.policies.random_masked_policy import RandomMaskedPolicy
from tests.conftest import DECK_PATH, MAX_OPTIONS, N_ACTIONS

DECK = load_deck(DECK_PATH)


def _logits(option_logits: list[float], stop: float) -> torch.Tensor:
    """
    Build an action-logit vector from option scores and a stop score.

    :param option_logits: Logit per option, placed at indices ``0..k-1``.
    :param stop: Logit for the synthetic stop action (last index).
    :return: Tensor of shape ``(N_ACTIONS,)``.
    """
    logits = torch.full((N_ACTIONS,), -1e9)
    logits[: len(option_logits)] = torch.tensor(option_logits)
    logits[N_ACTIONS - 1] = stop
    return logits


def test_sampling_agent_picks_only_extreme_logit() -> None:
    """
    With one logit overwhelming the rest, sampling converges to the argmax
    (a sanity check that probability mass concentrates as expected).
    """
    torch.manual_seed(0)

    class EmptyEncoder:
        @staticmethod
        def encode(*_args) -> TensorDict:
            return TensorDict({}, batch_size=torch.Size(()))

    class ExtremeActor(nn.Module):
        @staticmethod
        def policy_logits(encoded: TensorDict) -> torch.Tensor:
            del encoded
            return _logits([0.0, 0.0, 50.0, 0.0], stop=-50.0)

    agent = InferenceAgent(ExtremeActor(), EmptyEncoder(), max_options=MAX_OPTIONS)
    observation = SimpleNamespace(
        select=SimpleNamespace(option=[object()] * 4, minCount=1, maxCount=1),
        current=SimpleNamespace(yourIndex=0),
    )

    assert agent(observation) == [2]


def test_sampling_agent_recomputes_logits_for_partial_selects() -> None:
    """
    A multi-pick is served like training: re-encode and re-run the policy after
    each pick so the updated already-chosen count can change the next action.
    """

    class RecordingEncoder:
        def __init__(self) -> None:
            self.counts: list[int] = []

        def encode(
            self, observation, seat: int, already_chosen_option_count: int
        ) -> TensorDict:
            del observation, seat
            self.counts.append(already_chosen_option_count)
            return TensorDict({}, batch_size=torch.Size(()))

    class PickThenStopActor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def policy_logits(self, encoded: TensorDict) -> torch.Tensor:
            del encoded
            self.calls += 1
            logits = torch.full((N_ACTIONS,), -1e9)
            logits[2 if self.calls == 1 else N_ACTIONS - 1] = 1e9
            return logits

    encoder = RecordingEncoder()
    actor = PickThenStopActor()
    agent = InferenceAgent(actor, encoder, max_options=MAX_OPTIONS)
    observation = SimpleNamespace(
        select=SimpleNamespace(option=[object()] * 4, minCount=1, maxCount=3),
        current=SimpleNamespace(yourIndex=0),
    )

    assert agent(observation) == [2]
    assert encoder.counts == [0, 1]
    assert actor.calls == 2


def test_sampling_agent_is_stochastic() -> None:
    """
    Repeated draws from a close-scoring distribution don't all return the
    same single pick.
    """
    torch.manual_seed(2)

    class EmptyEncoder:
        @staticmethod
        def encode(*_args) -> TensorDict:
            return TensorDict({}, batch_size=torch.Size(()))

    class CloseActor(nn.Module):
        @staticmethod
        def policy_logits(encoded: TensorDict) -> torch.Tensor:
            del encoded
            return _logits([1.0, 0.9], stop=-50.0)

    agent = InferenceAgent(CloseActor(), EmptyEncoder(), max_options=MAX_OPTIONS)
    observation = SimpleNamespace(
        select=SimpleNamespace(option=[object()] * 2, minCount=1, maxCount=1),
        current=SimpleNamespace(yourIndex=0),
    )
    picks = {agent(observation)[0] for _ in range(30)}
    assert len(picks) > 1


def test_sampling_agent_rejects_option_overflow() -> None:
    """Inference fails clearly rather than truncating the engine option list."""

    class EmptyEncoder:
        @staticmethod
        def encode(
            observation, seat: int, already_chosen_option_count: int
        ) -> TensorDict:
            del observation, seat, already_chosen_option_count
            return TensorDict({}, batch_size=torch.Size(()))

    class SmallActor(nn.Module):
        @staticmethod
        def policy_logits(encoded: TensorDict) -> torch.Tensor:
            del encoded
            return torch.zeros(3)  # two options plus stop

    agent = InferenceAgent(SmallActor(), EmptyEncoder(), max_options=2)
    observation = SimpleNamespace(
        select=SimpleNamespace(option=[object()] * 3, minCount=1, maxCount=1),
        current=SimpleNamespace(yourIndex=0),
    )

    with pytest.raises(ValueError, match="offers 3 options"):
        agent(observation)


def test_structured_encoder_updates_cached_pick_count_only() -> None:
    """The multi-pick fast path mutates only the count-dependent global field."""
    handle = BattleHandle()
    try:
        observation = handle.start(DECK, DECK)
    finally:
        handle.finish()
    assert observation.current is not None
    encoder = StructuredObservationEncoder(max_options=MAX_OPTIONS)
    encoded = encoder.encode(observation, observation.current.yourIndex, 0)
    globals_before = encoded["globals"].clone()

    assert encoder.update_already_chosen_option_count(encoded, 2)

    expected = globals_before.clone()
    expected[encoder.ALREADY_CHOSEN_OPTION_COUNT_INDEX] = 2.0
    assert torch.equal(encoded["globals"], expected)


def _run_episode(opponent) -> torch.Tensor:
    """
    Play one episode with a random agent against the given opponent.

    :param opponent: Opponent callable for the non-agent seat.
    :return: The rollout's ``("next", "done")`` flags.
    """
    env = TransformedEnv(
        TCGEnv(
            DECK,
            DECK,
            seed=3,
            opponent=opponent,
            encoder=StructuredObservationEncoder(max_options=MAX_OPTIONS),
        ),
        ActionMask(),
    )
    try:
        rollout = env.rollout(
            400, policy=RandomMaskedPolicy(), break_when_any_done=True
        )
    finally:
        env.close()
    return rollout["next", "done"]


def test_sampling_agent_plays_legal_episode(
    structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    The submission agent plays a full episode without the engine rejecting a
    selection (an illegal pick would raise inside the engine).
    """
    torch.manual_seed(5)
    actor_critic = build_actor_critic(
        structured_model_cfg, structured_obs_spec, action_spec
    )
    agent = InferenceAgent(
        actor_critic,
        StructuredObservationEncoder(max_options=MAX_OPTIONS),
        max_options=MAX_OPTIONS,
    )
    done = _run_episode(agent)
    assert bool(done.any())


def test_load_inference_agent_roundtrip(tmp_path, structured_model_cfg) -> None:
    """
    A checkpoint + model_config.yaml pair (the artifact
    ``scripts/export_inference_checkpoint.py`` produces) reloads via
    ``load_inference_agent`` into an agent that plays a full episode legally.
    """
    torch.manual_seed(7)
    obs_spec, _, action_spec = build_inference_specs(MAX_OPTIONS)
    actor_critic = build_actor_critic(structured_model_cfg, obs_spec, action_spec)
    checkpoint_path = save_actor_critic(actor_critic, tmp_path / "model.pt")
    model_config = OmegaConf.create(
        {
            "model": OmegaConf.to_container(structured_model_cfg.model, resolve=True),
            "max_options": MAX_OPTIONS,
            "encoder": "structured",
        }
    )
    model_config_path = tmp_path / "model_config.yaml"
    OmegaConf.save(model_config, model_config_path)

    agent = load_inference_agent(checkpoint_path, model_config_path)
    done = _run_episode(agent)
    assert bool(done.any())
