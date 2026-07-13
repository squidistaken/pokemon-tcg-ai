from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.env.opponent_pool import OpponentPool
from src.env.random_opponent import RandomOpponent
from src.policies.random_masked_policy import RandomMaskedPolicy
from src.training.env_factory import make_env_factories
from src.training.trainer import Trainer

DECK_PATH = str(Path(__file__).parents[1] / "decks" / "example.csv")


def make_cfg(num_workers: int = 2) -> DictConfig:
    """
    Build a minimal config for the environment factories.

    :param num_workers: Number of environment workers.
    :return: OmegaConf config with seed and env sections.
    """
    return OmegaConf.create(
        {
            "seed": 0,
            "env": {
                "deck0": DECK_PATH,
                "deck1": DECK_PATH,
                "max_options": 96,
                "num_workers": num_workers,
                "parallel": False,
            },
        }
    )


def make_opponent_pool() -> OpponentPool:
    """
    Build a two-member random opponent pool.

    Module-level so the factory stays picklable for ParallelEnv workers.

    :return: OpponentPool with two seeded random opponents.
    """
    return OpponentPool([RandomOpponent(seed=0), RandomOpponent(seed=1)], seed=2)


def test_trainer_collects_frames() -> None:
    """
    The base trainer runs pure collection over a SerialEnv and reports stats.
    """
    trainer = Trainer(
        env_factories=make_env_factories(make_cfg()),
        policy=RandomMaskedPolicy(),
        frames_per_batch=64,
        total_frames=128,
        use_parallel_env=False,
    )
    stats = trainer.train()
    assert stats["frames"] == 128
    assert stats["episodes"] > 0
    assert 0.0 <= stats["win_rate"] <= 1.0


def test_trainer_with_opponent_pool() -> None:
    """
    Environments built with an opponent pool factory collect normally.
    """
    trainer = Trainer(
        env_factories=make_env_factories(make_cfg(), opponent_factory=make_opponent_pool),
        policy=RandomMaskedPolicy(),
        frames_per_batch=64,
        total_frames=64,
        use_parallel_env=False,
    )
    stats = trainer.train()
    assert stats["frames"] == 64


def test_opponent_pool_resampling() -> None:
    """
    on_reset draws pool members according to the sampling weights.
    """
    first = RandomOpponent(seed=0)
    second = RandomOpponent(seed=1)
    pool = OpponentPool([first, second], seed=3)
    drawn = set()
    for _ in range(50):
        pool.on_reset()
        drawn.add(id(pool.active))
    assert drawn == {id(first), id(second)}

    weighted_pool = OpponentPool([first, second], weights=[1.0, 0.0], seed=4)
    for _ in range(20):
        weighted_pool.on_reset()
        assert weighted_pool.active is first
