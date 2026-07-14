
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.transforms import ActionMask

from src.env.deck import load_deck
from src.env.flat_observation_encoder import FlatObservationEncoder
from src.env.tcg_env import TCGEnv
from src.policies.ppo_actor import build_ppo_actor_critic
from tests.conftest import DECK_PATH, FLAT_DIM, N_ACTIONS

DECK = load_deck(DECK_PATH)


def _flat_env(seed: int = 0) -> EnvBase:
    """
    Build a masked flat-observation environment for rollout tests.

    :param seed: Environment seed.
    :return: TransformedEnv with the ActionMask transform.
    """
    env = TCGEnv(DECK, DECK, seed=seed, encoder=FlatObservationEncoder())
    return TransformedEnv(env, ActionMask())


def test_operator_writes_int64_action(model_cfg, flat_obs_spec, action_spec) -> None:
    """
    The policy operator matches the RandomMaskedPolicy contract: it reads the
    mask and writes an int64 ``action``.
    """
    operator = build_ppo_actor_critic(model_cfg, flat_obs_spec, action_spec)
    policy = operator.get_policy_operator()
    mask = torch.zeros(8, N_ACTIONS, dtype=torch.bool)
    mask[:, :5] = True
    td = TensorDict(
        {"observation": {"observation": torch.randn(8, FLAT_DIM)}, "action_mask": mask},
        batch_size=[8],
    )
    out = policy(td)
    assert out["action"].shape == (8,)
    assert out["action"].dtype == torch.int64


def test_actions_never_illegal(model_cfg, flat_obs_spec, action_spec) -> None:
    """
    ``MaskedCategorical`` assigns ~zero probability to illegal actions: sampled
    actions always fall on legal indices.
    """
    operator = build_ppo_actor_critic(model_cfg, flat_obs_spec, action_spec)
    policy = operator.get_policy_operator()
    n_legal = 3
    mask = torch.zeros(256, N_ACTIONS, dtype=torch.bool)
    mask[:, :n_legal] = True
    td = TensorDict(
        {"observation": {"observation": torch.randn(256, FLAT_DIM)}, "action_mask": mask},
        batch_size=[256],
    )
    actions = policy(td)["action"]
    assert bool((actions < n_legal).all())
    assert bool(mask.gather(-1, actions.unsqueeze(-1)).all())


def test_rollout_respects_mask(model_cfg, flat_obs_spec, action_spec) -> None:
    """
    Swapped into a live env rollout, every action the actor takes is legal
    under the mask of the state it acted on.
    """
    operator = build_ppo_actor_critic(model_cfg, flat_obs_spec, action_spec)
    env = _flat_env(seed=1)
    try:
        rollout = env.rollout(60, policy=operator.get_policy_operator(), break_when_any_done=False)
    finally:
        env.close()
    masks = rollout["action_mask"]
    actions = rollout["action"]
    assert bool(masks.gather(-1, actions.unsqueeze(-1)).all())
