from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """
    Interface for a trainer: something that runs until a training budget is
    exhausted and reports aggregate statistics.
    """

    @abstractmethod
    def train(self) -> dict[str, float]:
        """
        Run training to completion.

        :return: Aggregate statistics for the run (e.g. frames, episodes, fps).
        """
        raise NotImplementedError
