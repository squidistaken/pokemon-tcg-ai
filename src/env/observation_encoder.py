import torch

from cg.api import Observation, PlayerState


class FlatObservationEncoder:
    """
    Placeholder encoder mapping an engine observation to a flat float vector.

    Deliberately minimal: it exists so the environment is fully
    tensor-compatible and the collector pipeline can run end to end. It will
    be replaced by a richer card-aware encoder (embeddings, per-option
    features) once the agent is built, so nothing here should be considered
    a modelling decision.
    """

    GLOBAL_FEATURES = 8
    SELECT_FEATURES = 6
    PLAYER_FEATURES = 11

    @property
    def dim(self) -> int:
        """
        Size of the encoded observation vector.

        :return: Number of features produced by :meth:`encode`.
        """
        return self.GLOBAL_FEATURES + self.SELECT_FEATURES + 2 * self.PLAYER_FEATURES

    def encode(self, observation: Observation, agent_seat: int, chosen_count: int) -> torch.Tensor:
        """
        Encode an observation from the agent's perspective.

        :param observation: Current engine observation (``current`` must be set).
        :param agent_seat: Player index (0 or 1) of the agent.
        :param chosen_count: Number of options already picked in an ongoing
            multi-select accumulation.
        :return: Float32 tensor of shape ``(self.dim,)``.
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
        return torch.tensor(features, dtype=torch.float32)

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
