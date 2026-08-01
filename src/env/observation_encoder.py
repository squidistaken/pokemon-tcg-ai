from abc import ABC, abstractmethod

from tensordict import TensorDict
from torchrl.data import Composite

from cg.api import Observation


class ObservationEncoder(ABC):
    """
    Interface for an observation encoder: transforms an engine
    :class:`cg.api.Observation` into a TorchRL tensordict.

    Implementations must provide a TorchRL spec tree (for environment
    construction) and an encode method (for per-step conversion). The
    encoder is stateless — it does not keep a roll of previous observations.
    """

    @abstractmethod
    def spec(self) -> Composite:
        """
        Build the TorchRL spec tree describing the encoded observation.

        :return: A :class:`~torchrl.data.Composite` spec matching the
            structure returned by :meth:`encode`.
        """
        raise NotImplementedError

    @abstractmethod
    def encode(
            self,
            observation: Observation,
            agent_seat: int,
            already_chosen_option_count: int,
    ) -> TensorDict:
        """
        Encode an engine observation from the agent's perspective.

        :param observation: Current engine observation.
        :param agent_seat: Player index (0 or 1) of the agent.
        :param already_chosen_option_count: Number of options already picked in
            an ongoing multi-select accumulation.
        :return: :class:`~tensordict.TensorDict` matching the structure
            declared by :meth:`spec`.
        """
        raise NotImplementedError

    def update_already_chosen_option_count(  # noqa: PLR6301 - optional instance extension point
            self,
            encoded: TensorDict,
            already_chosen_option_count: int,
    ) -> bool:
        """
        Update a cached encoding for the next partial multi-select pick.

        Encoders may override this when the pick count is the only field that
        changes between partial picks. Returning False asks inference to call
        :meth:`encode` again, preserving correctness for other encoders.

        :param encoded: Previously encoded observation to update in place.
        :param already_chosen_option_count: New number of accumulated picks.
        :return: True when the cached encoding was updated, otherwise False.
        """
        del encoded, already_chosen_option_count
        return False
