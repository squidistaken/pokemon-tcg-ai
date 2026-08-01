import torch
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import ActionMask

from src.env.deck import load_deck
from src.env.structured_observation_encoder import StructuredObservationEncoder
from src.env.tcg_env import TCGEnv
from src.policies.greedy_policy_opponent import (
    GreedyPolicyOpponent,
    load_greedy_opponent,
    save_actor_critic,
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


def test_greedy_select_single_pick() -> None:
    """
    A single-pick selection returns the argmax option.
    """
    picks = GreedyPolicyOpponent.greedy_select(
        _logits([0.1, 0.2, 5.0, 0.3, 0.4], stop=-1.0), n_options=5, min_count=1, max_count=1
    )
    assert picks == [2]


def test_greedy_select_stops_between_min_and_max() -> None:
    """
    Multi-select takes options above the stop logit, honoring minCount.
    """
    picks = GreedyPolicyOpponent.greedy_select(
        _logits([3.0, 2.0, 1.0, 0.5], stop=1.5), n_options=4, min_count=1, max_count=3
    )
    assert picks == [0, 1]


def test_greedy_select_respects_min_and_max_counts() -> None:
    """
    minCount forces picks below stop; maxCount caps picks above it.
    """
    empty = GreedyPolicyOpponent.greedy_select(
        _logits([0.1, 0.2], stop=5.0), n_options=2, min_count=0, max_count=2
    )
    assert empty == []
    capped = GreedyPolicyOpponent.greedy_select(
        _logits([5.0, 4.0, 3.0, 2.0], stop=-10.0), n_options=4, min_count=1, max_count=2
    )
    assert capped == [0, 1]


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


def test_greedy_opponent_plays_legal_episode(structured_model_cfg, structured_obs_spec, action_spec) -> None:
    """
    A greedy opponent plays a full episode without the engine rejecting a
    selection (an illegal pick would raise inside the engine).
    """
    actor_critic = build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec)
    opponent = GreedyPolicyOpponent(actor_critic, StructuredObservationEncoder(max_options=MAX_OPTIONS))
    done = _run_episode(opponent)
    assert bool(done.any())


def test_snapshot_roundtrip_plays(tmp_path, structured_model_cfg, structured_obs_spec, action_spec) -> None:
    """
    A snapshot saved to disk reloads into a greedy opponent that plays legally
    (the documented self-play checkpoint cycle).
    """
    actor_critic = build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec)
    checkpoint = save_actor_critic(
        actor_critic,
        tmp_path / "snapshot.pt",
        config={"format": "versioned-test"},
        frames=100,
    )
    assert checkpoint.exists()

    opponent = load_greedy_opponent(
        checkpoint,
        structured_model_cfg,
        structured_obs_spec,
        action_spec,
        StructuredObservationEncoder(max_options=MAX_OPTIONS),
    )
    done = _run_episode(opponent)
    assert bool(done.any())
