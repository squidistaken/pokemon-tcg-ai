import logging

from torch import nn

from src.training.evaluator import Evaluator

logger = logging.getLogger(__name__)


class MultiEvaluator:
    """
    Runs several :class:`~src.training.evaluator.Evaluator` instances together.

    One reference opponent rarely answers the whole question. A random opponent
    is cheap and interpretable but saturates: once the policy reliably beats it
    the curve flattens and stops separating runs. A frozen snapshot keeps
    discriminating for far longer but its absolute number means nothing on its
    own, because it depends on which snapshot happened to be frozen. Scoring
    against both on the same policy at the same frame count gives one metric
    that stays readable late and one that stays comparable across runs.

    Metrics from each sub-evaluator are namespaced by its label
    (``"random/win_rate"``, ``"snapshot/win_rate"``), so callbacks and W&B see
    them as separate series rather than one overwriting the other.

    Each sub-evaluator owns its own environment, so evaluation cost is the sum
    of the parts. Two evaluators at 60 episodes cost the same as one at 120.
    """

    def __init__(self, evaluators: list[Evaluator]) -> None:
        """
        :param evaluators: Evaluators to run, each with a distinct
            :attr:`~src.training.evaluator.Evaluator.name`.
        :raises ValueError: If no evaluators are given, or two share a name
            (their metrics would silently collide under the same prefix).
        """
        if not evaluators:
            raise ValueError("MultiEvaluator needs at least one evaluator.")
        names = [evaluator.name for evaluator in evaluators]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(
                f"MultiEvaluator evaluator names must be unique; duplicated: "
                f"{sorted(duplicates)}"
            )
        self._evaluators = evaluators

    def evaluate(self, policy: nn.Module) -> dict[str, float]:
        """
        Score the policy against every reference opponent in turn.

        :param policy: Collection policy, passed unchanged to each evaluator.
        :return: Merged metrics, each key prefixed with its evaluator's name.
        """
        merged: dict[str, float] = {}
        for evaluator in self._evaluators:
            for key, value in evaluator.evaluate(policy).items():
                merged[f"{evaluator.name}/{key}"] = value
        return merged

    def close(self) -> None:
        """
        Release every sub-evaluator's environment.
        """
        for evaluator in self._evaluators:
            evaluator.close()
