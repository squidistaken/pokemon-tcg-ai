import torch
from tensordict import TensorDict
from torchrl.data import Composite
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.transforms import ActionMask

from src.env.decks.deck import load_deck
from src.env.observation.structured_observation_encoder import (
    StructuredObservationEncoder,
)
from src.env.tcg_env import TCGEnv
from src.policies.ppo_actor import build_ppo_actor_critic
from tests.conftest import DECK_PATH, MAX_OPTIONS

DECK = load_deck(DECK_PATH)


def _structured_env(seed: int = 0) -> EnvBase:
    """
    Build a masked structured-observation environment for rollout tests.

    :param seed: Environment seed.
    :return: TransformedEnv with the ActionMask transform.
    """
    env = TCGEnv(
        DECK,
        DECK,
        seed=seed,
        encoder=StructuredObservationEncoder(max_options=MAX_OPTIONS),
    )
    return TransformedEnv(env, ActionMask())


def _masked_obs(structured_obs_spec: Composite, batch: int, n_legal: int) -> TensorDict:
    """
    Build a zeroed observation batch with the first ``n_legal`` actions legal.

    :param structured_obs_spec: Env spec (observation + action mask).
    :param batch: Batch size.
    :param n_legal: Number of legal action slots, starting at index 0.
    :return: TensorDict with the ``observation`` groups and ``action_mask``.
    """
    obs = structured_obs_spec.zero((batch,))
    obs["action_mask"][:, :n_legal] = True
    return obs


def test_operator_writes_int64_action(
    structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    The policy operator matches the RandomMaskedPolicy contract: it reads the
    mask and writes an int64 ``action``.
    """
    operator = build_ppo_actor_critic(
        structured_model_cfg, structured_obs_spec, action_spec
    )
    policy = operator.get_policy_operator()
    out = policy(_masked_obs(structured_obs_spec, batch=8, n_legal=5))
    assert out["action"].shape == (8,)
    assert out["action"].dtype == torch.int64


def test_actions_never_illegal(
    structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    ``MaskedCategorical`` assigns ~zero probability to illegal actions: sampled
    actions always fall on legal indices.
    """
    operator = build_ppo_actor_critic(
        structured_model_cfg, structured_obs_spec, action_spec
    )
    policy = operator.get_policy_operator()
    n_legal = 3
    td = _masked_obs(structured_obs_spec, batch=256, n_legal=n_legal)
    actions = policy(td)["action"]
    assert bool((actions < n_legal).all())
    assert bool(td["action_mask"].gather(-1, actions.unsqueeze(-1)).all())


def test_rollout_respects_mask(
    structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    Swapped into a live env rollout, every action the actor takes is legal
    under the mask of the state it acted on.
    """
    operator = build_ppo_actor_critic(
        structured_model_cfg, structured_obs_spec, action_spec
    )
    env = _structured_env(seed=1)
    try:
        rollout = env.rollout(
            60, policy=operator.get_policy_operator(), break_when_any_done=False
        )
    finally:
        env.close()
    masks = rollout["action_mask"]
    actions = rollout["action"]
    assert bool(masks.gather(-1, actions.unsqueeze(-1)).all())
