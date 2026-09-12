from functools import partial
from pathlib import Path

import torch
from torchrl.collectors import Collector
from torchrl.envs import EnvBase, SerialEnv, TransformedEnv
from torchrl.envs.transforms import ActionMask
from torchrl.envs.utils import check_env_specs

from src.env.battle_handle import BattleHandle
from src.env.decks.deck import load_deck
from src.env.opponents.random_opponent import RandomOpponent
from src.env.tcg_env import TCGEnv
from src.policies.random_masked_policy import RandomMaskedPolicy

DECK = load_deck(str(Path(__file__).parents[1] / "decks" / "example.csv"))


def make_env(seed: int = 0) -> EnvBase:
    """
    Build a masked test environment.

    :param seed: Seed for the environment instance.
    :return: TransformedEnv with the ActionMask transform applied.
    """
    return TransformedEnv(TCGEnv(DECK, DECK, seed=seed), ActionMask())


def test_env_specs() -> None:
    """
    The environment passes torchrl's spec consistency check.
    """
    env = make_env()
    check_env_specs(env)
    env.close()


def test_full_episode_with_random_policy() -> None:
    """
    A rollout with the random masked policy reaches a terminal state with
    a valid terminal reward.
    """
    env = make_env(seed=1)
    rollout = env.rollout(max_steps=5000, policy=RandomMaskedPolicy())
    done = rollout["next", "done"].reshape(-1)
    assert done[-1].item(), "episode did not finish within the step budget"
    final_reward = rollout["next", "reward"].reshape(-1)[-1].item()
    assert final_reward in (-1.0, 0.0, 1.0)
    intermediate_rewards = rollout["next", "reward"].reshape(-1)[:-1]
    assert torch.all(intermediate_rewards == 0.0)
    env.close()


def test_actions_respect_mask() -> None:
    """
    Every action sampled by the random masked policy was legal under the
    mask of the state it was sampled in.
    """
    env = make_env(seed=2)
    rollout = env.rollout(max_steps=500, policy=RandomMaskedPolicy())
    masks = rollout["action_mask"]
    actions = rollout["action"]
    assert masks.gather(-1, actions.unsqueeze(-1)).all()
    env.close()


def test_concurrent_battle_handles() -> None:
    """
    Two battle handles can run interleaved in the same process and the
    engine's select player matches the observation's yourIndex.
    """
    opponent = RandomOpponent(seed=3)
    handles = [BattleHandle(), BattleHandle()]
    observations = [handle.start(DECK, DECK) for handle in handles]
    for _ in range(50):
        for slot, handle in enumerate(handles):
            observation = observations[slot]
            state = observation.current
            assert state is not None
            if state.result != -1:
                continue
            assert handle.select_player == state.yourIndex
            observations[slot] = handle.select(opponent(observation))
    for handle in handles:
        handle.finish()


def test_serial_collector() -> None:
    """
    The Collector produces correctly shaped batches over a SerialEnv.
    """
    vec_env = SerialEnv(2, [partial(make_env, seed) for seed in (10, 11)])
    # Opt out of torchrl's automatic policy-transform registration: the random
    # masked policy only reads "action_mask", so the InitTracker transform the
    # collector's heuristic wants to add is not needed, and this env's transforms
    # are managed explicitly in make_env.
    collector = Collector(
        create_env_fn=vec_env,
        policy=RandomMaskedPolicy(),
        frames_per_batch=64,
        total_frames=128,
        auto_register_policy_transforms=False,
    )
    batches = list(collector)
    collector.shutdown()
    assert len(batches) == 2
    for batch in batches:
        assert batch.numel() == 64
        assert batch["next", "reward"].shape[-1] == 1
        assert batch["action"].dtype == torch.int64
