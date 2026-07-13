"""
Legacy flat-vector observation encoder.

**Deprecated.** Replaced by :class:`StructuredObservationEncoder`
(``structured_observation_encoder.py``), which emits card IDs as embedding
indices, per-option features aligned with the action mask, and padded zone
tables instead of a single flat vector.

Kept only for regression testing and as a reference baseline — do not use in
new code. To compare the structured output against the old flat representation
in a test::

    from src.env.flat_observation_encoder import FlatObservationEncoder
    from src.env.structured_observation_encoder import StructuredObservationEncoder

    legacy = FlatObservationEncoder()
    current = StructuredObservationEncoder()
    # both inherit from ObservationEncoder
"""

import torch
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded

from cg.api import Observation, PlayerState

from .observation_encoder import ObservationEncoder


class FlatObservationEncoder(ObservationEncoder):
    """
    Legacy encoder mapping an engine observation to a flat float vector
    wrapped in a minimal tensordict.

    .. deprecated::
        Use :class:`~src.env.structured_observation_encoder.StructuredObservationEncoder`
        instead. This class exists only for comparison and regression tests.
    """

    GLOBAL_FEATURES = 8
    SELECT_FEATURES = 6
    PLAYER_FEATURES = 11

    def __init__(self) -> None:
        super().__init__()
        self._dim = self.GLOBAL_FEATURES + self.SELECT_FEATURES + 2 * self.PLAYER_FEATURES

    @property
    def dim(self) -> int:
        """
        Size of the flat encoded observation vector.

        :return: ``36``.
        """
        return self._dim

    def spec(self) -> Composite:
        """
        Build the TorchRL spec: a single ``Unbounded`` entry of shape
        ``(self.dim,)`` under the ``"observation"`` key.

        :return: Minimal composite spec.
        """
        return Composite(
            observation=Unbounded(shape=(self._dim,), dtype=torch.float32),
        )

    def encode(
            self,
            observation: Observation,
            agent_seat: int,
            chosen_count: int,
    ) -> TensorDict:
        """
        Encode an observation as a flat float vector wrapped in a tensordict.

        :param observation: Current engine observation.
        :param agent_seat: Player index (0 or 1) of the agent.
        :param chosen_count: Number of options already picked in an ongoing
            multi-select accumulation.
        :return: Tensordict with a single ``"observation"`` key.
        """
        state = observation.current
        features: list[float] = [
            state.turn / 50.0,
            state.turnActionCount / 20.0,
            float(agent_seat),
            1.0 if state.firstPlayer == agent_seat else 0.0,
            float(state.supporterPlayed),
            float(state.stadiumPlayed),
            float(state.energyAttached),
            float(state.retreated),
        ]
        select = observation.select
        if select is None:
            features += [0.0] * self.SELECT_FEATURES
        else:
            features += [
                float(select.type) / 10.0,
                float(select.context) / 48.0,
                select.minCount / 6.0,
                select.maxCount / 6.0,
                len(select.option) / 60.0,
                chosen_count / 6.0,
            ]
        features += self._player_features(state.players[agent_seat])
        features += self._player_features(state.players[1 - agent_seat])
        return TensorDict(
            {"observation": torch.tensor(features, dtype=torch.float32)},
            batch_size=torch.Size(()),
        )

    def _player_features(self, player: PlayerState) -> list[float]:
        """
        Encode one player's board state.

        :param player: Player state to encode.
        :return: List of ``PLAYER_FEATURES`` floats.
        """
        active = player.active[0] if len(player.active) > 0 else None
        hp_fraction = active.hp / active.maxHp if active is not None and active.maxHp > 0 else 0.0
        return [
            1.0 if len(player.active) > 0 else 0.0,
            hp_fraction,
            player.deckCount / 60.0,
            player.handCount / 15.0,
            len(player.prize) / 6.0,
            len(player.bench) / max(player.benchMax, 1),
            float(player.poisoned),
            float(player.burned),
            float(player.asleep),
            float(player.paralyzed),
            float(player.confused),
        ]