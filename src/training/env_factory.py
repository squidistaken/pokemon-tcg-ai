from functools import partial
from typing import Callable

from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.transforms import ActionMask

from cg.api import Observation
from src.env.deck import load_deck
from src.env.tcg_env import TCGEnv

OpponentFactory = Callable[[], Callable[[Observation], list[int]]]


def make_env(
        deck0: list[int],
        deck1: list[int],
        max_options: int,
        seed: int,
        opponent_factory: OpponentFactory | None = None,
) -> EnvBase:
    """
    Build a single masked TCG environment instance.

    Module-level so the factory stays picklable for ParallelEnv workers.

    :param deck0: 60 card IDs for player 0.
    :param deck1: 60 card IDs for player 1.
    :param max_options: Padded size of the option space.
    :param seed: Seed for this environment instance.
    :param opponent_factory: Builds the opponent for this instance (e.g. an
        OpponentPool for self-play); the default random opponent if None.
    :return: TransformedEnv with the ActionMask transform applied.
    """
    opponent = opponent_factory() if opponent_factory is not None else None
    base_env = TCGEnv(deck0=deck0, deck1=deck1, max_options=max_options, seed=seed, opponent=opponent)
    return TransformedEnv(base_env, ActionMask())


def make_env_factories(cfg: DictConfig, opponent_factory: OpponentFactory | None = None) -> list[Callable[[], EnvBase]]:
    """
    Build one environment factory per worker from the Hydra config.

    :param cfg: Hydra configuration with ``seed`` and an ``env`` section.
    :param opponent_factory: Opponent factory forwarded to every instance.
    :return: List of ``cfg.env.num_workers`` picklable environment factories.
    """
    deck0 = load_deck(to_absolute_path(cfg.env.deck0))
    deck1 = load_deck(to_absolute_path(cfg.env.deck1))
    return [
        partial(make_env, deck0, deck1, cfg.env.max_options, cfg.seed + worker, opponent_factory)
        for worker in range(cfg.env.num_workers)
    ]
