import torch
from omegaconf import OmegaConf
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import ActionMask

from src.env.deck import load_deck
from src.env.structured_observation_encoder import StructuredObservationEncoder
from src.env.tcg_env import TCGEnv
from src.policies.greedy_policy_opponent import save_actor_critic
from src.policies.inference import (
    SamplingPolicyAgent,
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


def test_sample_select_picks_only_extreme_logit() -> None:
    """
    With one logit overwhelming the rest, sampling converges to the argmax
    (a sanity check that probability mass concentrates as expected).
    """
    torch.manual_seed(0)
    picks = SamplingPolicyAgent.sample_select(
        _logits([0.0, 0.0, 50.0, 0.0], stop=-50.0),
        n_options=4,
        min_count=1,
        max_count=1,
    )
    assert picks == [2]


def test_sample_select_respects_min_and_max_counts() -> None:
    """
    Sampling never returns fewer than minCount or more than maxCount picks,
    and never repeats or picks an out-of-range option, across many draws.
    """
    torch.manual_seed(1)
    logits = _logits([1.0, 0.9, 0.8, 0.7], stop=0.5)
    for _ in range(50):
        picks = SamplingPolicyAgent.sample_select(logits, n_options=4, min_count=1, max_count=3)
        assert 1 <= len(picks) <= 3
        assert len(set(picks)) == len(picks)
        assert all(0 <= pick < 4 for pick in picks)


def test_sample_select_is_stochastic() -> None:
    """
    Repeated draws from a close-scoring distribution don't all return the
    same single pick.
    """
    torch.manual_seed(2)
    logits = _logits([1.0, 0.9], stop=-50.0)
    picks = {
        tuple(SamplingPolicyAgent.sample_select(logits, n_options=2, min_count=1, max_count=1))
        for _ in range(30)
    }
    assert len(picks) > 1


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
        rollout = env.rollout(400, policy=RandomMaskedPolicy(), break_when_any_done=True)
    finally:
        env.close()
    return rollout["next", "done"]


def test_sampling_agent_plays_legal_episode(structured_model_cfg, structured_obs_spec, action_spec) -> None:
    """
    The submission agent plays a full episode without the engine rejecting a
    selection (an illegal pick would raise inside the engine).
    """
    torch.manual_seed(5)
    actor_critic = build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec)
    agent = SamplingPolicyAgent(actor_critic, StructuredObservationEncoder(max_options=MAX_OPTIONS))
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
