from pathlib import Path

import torch
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
    torch.manual_seed(0)
    trainer = Trainer(
        env_factories=make_env_factories(make_cfg()),
        policy=RandomMaskedPolicy(),
        frames_per_batch=128,
        total_frames=512,
        use_parallel_env=False,
    )
    stats = trainer.train()
    assert stats["frames"] == 512
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


class DyingCollector:
    """
    Collector wrapper that raises torchrl's worker-death error after one batch.

    A real death cannot be staged in-process (the engine aborts the whole
    worker), but the trainer only ever sees it as this ``RuntimeError``, so
    reproducing the message exercises the same recovery path.

    Wraps rather than patches the instance because ``__iter__`` is looked up on
    the type, so assigning it to a collector instance would be ignored.
    """

    def __init__(self, collector) -> None:
        """
        :param collector: The real collector to delegate to.
        """
        self._collector = collector

    def __iter__(self):
        """
        Yield exactly one batch, then die the way a lost worker does.

        :return: Generator over collected batches.
        :raises RuntimeError: After the first batch.
        """
        for batch in self._collector:
            yield batch
            raise RuntimeError("Cannot proceed, worker 0 dead.")

    def shutdown(self) -> None:
        """
        Tear the wrapped collector down.
        """
        self._collector.shutdown()


class WorkerDeathTrainer(Trainer):
    """
    Trainer whose first ``failures`` collectors die after one batch each.
    """

    def __init__(self, *args, failures: int = 1, **kwargs) -> None:
        """
        :param args: Forwarded to :class:`~src.training.trainer.Trainer`.
        :param failures: How many collectors should die before one survives.
        :param kwargs: Forwarded to :class:`~src.training.trainer.Trainer`.
        """
        super().__init__(*args, **kwargs)
        self.remaining_failures = failures
        self.collectors_built = 0

    def _make_collector(self, total_frames: int):
        """
        Build a collector, wrapped to die while failures remain.

        :param total_frames: Frames the collector should still collect.
        :return: The collector, possibly wrapped in :class:`DyingCollector`.
        """
        self.collectors_built += 1
        collector = super()._make_collector(total_frames)
        if self.remaining_failures <= 0:
            return collector
        self.remaining_failures -= 1
        return DyingCollector(collector)


def test_trainer_recovers_from_a_dead_worker() -> None:
    """
    A dead environment worker costs the in-flight batch, not the run.

    The learner lives in this process and is untouched when a worker dies, so
    the collector is rebuilt and collection continues toward the same budget
    rather than the run ending early.
    """
    trainer = WorkerDeathTrainer(
        env_factories=make_env_factories(make_cfg()),
        policy=RandomMaskedPolicy(),
        frames_per_batch=64,
        total_frames=256,
        use_parallel_env=False,
        failures=2,
    )

    stats = trainer.train()

    assert stats["frames"] == 256
    assert trainer.collectors_built == 3
    assert trainer.remaining_failures == 0


def test_trainer_stops_restarting_once_the_budget_is_spent() -> None:
    """
    An environment that dies forever fails the run instead of looping.
    """
    trainer = WorkerDeathTrainer(
        env_factories=make_env_factories(make_cfg()),
        policy=RandomMaskedPolicy(),
        frames_per_batch=64,
        total_frames=1024,
        use_parallel_env=False,
        failures=99,
        max_collector_restarts=2,
    )

    try:
        trainer.train()
    except RuntimeError as error:
        assert "worker 0 dead" in str(error)
    else:
        raise AssertionError("expected the run to fail once restarts ran out")
    assert trainer.collectors_built == 3


def test_trainer_does_not_restart_on_an_ordinary_error() -> None:
    """
    Only worker deaths are recovered from; a real bug still fails fast.
    """

    class BuggyTrainer(Trainer):
        """Trainer whose update raises a non-worker error."""

        def _update(self, data):  # noqa: ARG002, PLR6301
            """
            :param data: Ignored.
            :raises RuntimeError: Always.
            """
            raise RuntimeError("shape mismatch in loss")

    trainer = BuggyTrainer(
        env_factories=make_env_factories(make_cfg()),
        policy=RandomMaskedPolicy(),
        frames_per_batch=64,
        total_frames=256,
        use_parallel_env=False,
    )

    try:
        trainer.train()
    except RuntimeError as error:
        assert "shape mismatch" in str(error)
    else:
        raise AssertionError("expected the bug to propagate")
