import random
from collections.abc import Callable

from cg.api import Observation


class OpponentPool:
    """
    Callable opponent that samples one of its members for each episode.

    This is the self-play entry point: the environment calls
    :meth:`on_reset` at every episode start, at which point the pool draws
    the opponent that will play the whole episode. During training, frozen
    policy snapshots are added via :meth:`add` so newer agents keep facing
    a league of past versions instead of only the latest one.
    """

    def __init__(
            self,
            opponents: list[Callable[[Observation], list[int]]],
            weights: list[float] | None = None,
            seed: int | None = None,
    ) -> None:
        """
        :param opponents: Initial pool members, each mapping an observation
            to a list of chosen option indices.
        :param weights: Sampling weight per member; uniform if None.
        :param seed: Seed for the sampling random number generator.
        """
        if not opponents:
            raise ValueError("OpponentPool needs at least one opponent.")
        if weights is not None and len(weights) != len(opponents):
            raise ValueError("weights must have the same length as opponents.")
        self._opponents = list(opponents)
        self._weights = list(weights) if weights is not None else [1.0] * len(opponents)
        self._rng = random.Random(seed)
        self._active = self._opponents[0]

    @property
    def active(self) -> Callable[[Observation], list[int]]:
        """
        The member currently playing, as drawn by the last :meth:`on_reset`.

        :return: The active opponent callable.
        """
        return self._active

    def add(self, opponent: Callable[[Observation], list[int]], weight: float = 1.0) -> None:
        """
        Add a new member to the pool (e.g. a frozen policy snapshot).

        :param opponent: Opponent callable to add.
        :param weight: Sampling weight of the new member.
        """
        self._opponents.append(opponent)
        self._weights.append(weight)

    def on_reset(self) -> None:
        """
        Draw the member that will play the next episode.
        """
        self._active = self._rng.choices(self._opponents, weights=self._weights, k=1)[0]

    def __call__(self, observation: Observation) -> list[int]:
        """
        Delegate the selection to the active member.

        :param observation: Current engine observation.
        :return: List of chosen option indices.
        """
        return self._active(observation)

    def seed(self, seed: int) -> None:
        """
        Reseed the pool sampler and all members that support seeding.

        :param seed: New seed value.
        """
        self._rng.seed(seed)
        for offset, opponent in enumerate(self._opponents):
            if hasattr(opponent, "seed"):
                opponent.seed(seed + 1 + offset)
