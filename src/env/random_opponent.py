import random

from cg.api import Observation


class RandomOpponent:
    """
    Opponent policy that selects uniformly at random among the legal options.

    Used inside :class:`~src.env.tcg_env.TCGEnv` to play the non-agent seat.
    Unlike the agent, it answers a full selection (including multi-select)
    in a single call, since it talks to the engine directly.
    """

    def __init__(self, seed: int | None = None) -> None:
        """
        :param seed: Seed for the internal random number generator.
        """
        self._rng = random.Random(seed)

    def __call__(self, observation: Observation) -> list[int]:
        """
        Choose a random legal selection for the given observation.

        :param observation: Current engine observation with a non-None select.
        :return: List of chosen option indices.
        """
        select = observation.select
        count = self._rng.randint(select.minCount, select.maxCount)
        return self._rng.sample(range(len(select.option)), count)

    def seed(self, seed: int) -> None:
        """
        Reseed the internal random number generator.

        :param seed: New seed value.
        """
        self._rng.seed(seed)
