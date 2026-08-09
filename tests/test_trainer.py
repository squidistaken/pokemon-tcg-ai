import os
import signal
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torchrl import _utils as torchrl_utils

from src.env.opponent_pool import OpponentPool
from src.env.random_opponent import RandomOpponent
from src.policies.random_masked_policy import RandomMaskedPolicy
from src.training.env_factory import make_env_factories
from src.training.pipe_timeout import apply_pipe_timeout
from src.training.trainer import _MAX_BARREN_RESTARTS, Trainer, _is_worker_death

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
        env_factories=make_env_factories(
            make_cfg(), opponent_factory=make_opponent_pool
        ),
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


class _ScriptedCollector:
    """
    Stand-in for a torchrl Collector with a scripted lifetime.

    Yields ``batches`` synthetic rollouts and then either stops (a collector
    that reached its budget) or raises the error torchrl reports when a
    ParallelEnv worker process has died. Substituted for the real thing so the
    restart path is exercised without having to kill a subprocess, which is
    what the engine abort this guards against actually does.

    :param batches: Rollouts to yield before stopping or dying.
    :param frames_per_batch: Frames in each yielded rollout.
    :param error: Exception to raise once the batches run out; None stops.
    """

    def __init__(
        self,
        batches: int,
        frames_per_batch: int,
        error: BaseException | None = None,
    ) -> None:
        self._batches = batches
        self._frames_per_batch = frames_per_batch
        self._error = error
        self.shutdown_calls = 0

    def __iter__(self):
        for _ in range(self._batches):
            yield _synthetic_batch(self._frames_per_batch)
        if self._error is not None:
            raise self._error

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _synthetic_batch(frames: int) -> TensorDict:
    """
    Build the minimum batch the trainer's bookkeeping reads.

    :param frames: Number of transitions in the batch.
    :return: TensorDict with the "next" done/terminated/reward keys, last step
        terminal.
    """
    done = torch.zeros(frames, 1, dtype=torch.bool)
    done[-1] = True
    reward = torch.zeros(frames, 1)
    reward[-1] = 1.0
    return TensorDict(
        {
            "next": TensorDict(
                {"done": done, "terminated": done.clone(), "reward": reward},
                batch_size=[frames],
            )
        },
        batch_size=[frames],
    )


class _ScriptedTrainer(Trainer):
    """
    Trainer handing out a prepared queue of collectors instead of building them.

    :param collectors: Collectors to return from successive
        ``_make_collector`` calls, in order.
    """

    def __init__(self, collectors: list[_ScriptedCollector], **kwargs) -> None:
        super().__init__(env_factories=[], policy=RandomMaskedPolicy(), **kwargs)
        self.collectors = collectors
        self.budgets: list[int] = []
        self.restarts: list[int] = []

    def _make_collector(self, remaining_frames: int):
        self.budgets.append(remaining_frames)
        return self.collectors[len(self.budgets) - 1]

    def _prepare_restart(self, restart_index: int) -> None:
        super()._prepare_restart(restart_index)
        self.restarts.append(restart_index)


WORKER_DEATH = RuntimeError(
    "At least one process failed. Check for more infos in the log."
)


def test_worker_death_is_told_apart_from_ordinary_failures() -> None:
    """
    Only torchrl's dead-process reports and broken pipes count as worker death.
    """
    assert _is_worker_death(WORKER_DEATH)
    assert _is_worker_death(RuntimeError("Cannot proceed, worker 6 dead."))
    assert _is_worker_death(BrokenPipeError())
    assert not _is_worker_death(RuntimeError("shape '[2, 3]' is invalid for input"))


def test_trainer_resumes_collection_after_a_worker_dies() -> None:
    """
    A dead pool costs the in-flight batch, not the run: the frame count carries
    over and the replacement collector is asked only for what is still owed.
    """
    trainer = _ScriptedTrainer(
        collectors=[
            _ScriptedCollector(batches=2, frames_per_batch=64, error=WORKER_DEATH),
            _ScriptedCollector(batches=6, frames_per_batch=64),
        ],
        frames_per_batch=64,
        total_frames=512,
        max_collector_restarts=3,
    )
    stats = trainer.train()

    assert stats["frames"] == 512
    assert stats["episodes"] == 8
    # 512 owed up front, then only the 384 the dead pool never delivered.
    assert trainer.budgets == [512, 384]
    assert trainer.restarts == [1]
    assert [collector.shutdown_calls for collector in trainer.collectors] == [1, 1]


def test_trainer_rebuilds_env_factories_with_a_shifted_seed() -> None:
    """
    Restarted workers are re-seeded, so they do not replay the matchup and seat
    sequence the dead pool already played.
    """
    rebuilt: list[int] = []

    def rebuild(restart_index: int) -> list:
        rebuilt.append(restart_index)
        return []

    trainer = _ScriptedTrainer(
        collectors=[
            _ScriptedCollector(batches=1, frames_per_batch=64, error=WORKER_DEATH),
            _ScriptedCollector(batches=1, frames_per_batch=64, error=WORKER_DEATH),
            _ScriptedCollector(batches=2, frames_per_batch=64),
        ],
        frames_per_batch=64,
        total_frames=256,
        max_collector_restarts=3,
        rebuild_env_factories=rebuild,
    )
    trainer.train()

    assert rebuilt == [1, 2]


def test_trainer_stops_when_the_restart_budget_is_spent() -> None:
    """
    The budget is a real ceiling; past it the death is reported, not absorbed.
    """
    trainer = _ScriptedTrainer(
        collectors=[
            _ScriptedCollector(batches=1, frames_per_batch=64, error=WORKER_DEATH),
            _ScriptedCollector(batches=1, frames_per_batch=64, error=WORKER_DEATH),
        ],
        frames_per_batch=64,
        total_frames=512,
        max_collector_restarts=1,
    )
    with pytest.raises(RuntimeError, match="At least one process failed"):
        trainer.train()
    assert trainer.restarts == [1]


def test_trainer_gives_up_on_a_pool_that_never_completes_a_batch() -> None:
    """
    Restarts that collect nothing are a pool failing for a reason a restart
    cannot fix, so the loop stops well inside a generous restart budget.
    """
    trainer = _ScriptedTrainer(
        collectors=[
            _ScriptedCollector(batches=0, frames_per_batch=64, error=WORKER_DEATH)
            for _ in range(10)
        ],
        frames_per_batch=64,
        total_frames=512,
        max_collector_restarts=10,
    )
    with pytest.raises(RuntimeError, match="At least one process failed"):
        trainer.train()
    assert len(trainer.budgets) == _MAX_BARREN_RESTARTS


def test_trainer_does_not_restart_on_an_ordinary_failure() -> None:
    """
    A bug in the update is not a dead worker and must not be retried.
    """
    trainer = _ScriptedTrainer(
        collectors=[
            _ScriptedCollector(
                batches=1, frames_per_batch=64, error=RuntimeError("bad shapes")
            )
        ],
        frames_per_batch=64,
        total_frames=512,
        max_collector_restarts=5,
    )
    with pytest.raises(RuntimeError, match="bad shapes"):
        trainer.train()
    assert trainer.restarts == []


def test_trainer_without_a_restart_budget_fails_on_the_first_death() -> None:
    """
    The default is unchanged behaviour: worker death ends the run.
    """
    trainer = _ScriptedTrainer(
        collectors=[
            _ScriptedCollector(batches=1, frames_per_batch=64, error=WORKER_DEATH)
        ],
        frames_per_batch=64,
        total_frames=512,
    )
    with pytest.raises(RuntimeError, match="At least one process failed"):
        trainer.train()
    assert trainer.restarts == []


def test_seed_offset_shifts_every_worker_seed() -> None:
    """
    Restart factories seed a disjoint block, one full pool width per restart.
    """
    cfg = make_cfg(num_workers=4)
    baseline = [factory.args[2] for factory in make_env_factories(cfg)]
    shifted = [factory.args[2] for factory in make_env_factories(cfg, seed_offset=4)]

    assert baseline == [0, 1, 2, 3]
    assert shifted == [4, 5, 6, 7]


class _WorkerKillingTrainer(Trainer):
    """
    Trainer that SIGKILLs a live worker partway through its first pool.

    SIGKILL leaves the parent with exactly what the engine's abort leaves it
    with: a vanished process and a dead pipe. Nothing short of killing a real
    process exercises torchrl's liveness check, the teardown of a half-dead
    pool or the rebuild, which is where the risk in this path actually sits --
    a scripted exception proves only that the branch is reachable.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pools: list[list[int]] = []
        self.killed = False

    def _collect(self, collector, progress_bar, totals, start_time) -> None:
        # ParallelEnv exposes its processes only privately and offers no public
        # accessor; an AttributeError here is the right failure if torchrl ever
        # renames it, rather than a test that quietly kills nothing. The cast is
        # for `env` too: the trainer is typed against the narrower
        # TrainingCollector protocol, and only the `sync` collector this test
        # builds owns a batched environment at all.
        workers = cast(
            list[BaseProcess],
            cast(Any, collector).env._workers,  # noqa: SLF001 - no public accessor
        )
        self.pools.append([cast(int, worker.pid) for worker in workers])
        original_update = self._update

        def update_then_kill(data):
            result = original_update(data)
            if not self.killed:
                self.killed = True
                os.kill(self.pools[0][-1], signal.SIGKILL)
            return result

        self._update = update_then_kill
        try:
            super()._collect(collector, progress_bar, totals, start_time)
        finally:
            self._update = original_update


def test_trainer_survives_a_real_worker_process_dying() -> None:
    """
    A killed ParallelEnv worker costs the pool, not the run.

    The replacement pool is a genuinely new set of processes, and the run
    reaches its full budget across the two.
    """
    cfg = make_cfg()
    cfg.env.parallel = True
    trainer = _WorkerKillingTrainer(
        env_factories=make_env_factories(cfg),
        policy=RandomMaskedPolicy(),
        frames_per_batch=128,
        total_frames=512,
        use_parallel_env=True,
        mp_start_method="fork",
        max_collector_restarts=2,
        rebuild_env_factories=lambda index: make_env_factories(
            cfg, seed_offset=2 * index
        ),
    )
    stats = trainer.train()

    assert stats["frames"] == 512
    assert len(trainer.pools) == 2
    assert set(trainer.pools[0]).isdisjoint(trainer.pools[1])


def test_pipe_timeout_overrides_the_torchrl_default() -> None:
    """
    The configured detection timeout reaches torchrl before a pool is built.

    Both sides of the pool read torchrl's module attribute when the workers
    start, so setting it is what makes a shorter timeout take effect; an
    environment variable set after import would not, since torchrl reads it
    once.
    """
    original = torchrl_utils.BATCHED_PIPE_TIMEOUT
    try:
        apply_pipe_timeout(180.0)
        assert torchrl_utils.BATCHED_PIPE_TIMEOUT == 180.0
    finally:
        torchrl_utils.BATCHED_PIPE_TIMEOUT = original


def test_pipe_timeout_of_none_keeps_the_torchrl_default() -> None:
    """
    Opting out with None leaves torchrl's own timeout untouched.
    """
    original = torchrl_utils.BATCHED_PIPE_TIMEOUT
    try:
        apply_pipe_timeout(None)
        assert original == torchrl_utils.BATCHED_PIPE_TIMEOUT
    finally:
        torchrl_utils.BATCHED_PIPE_TIMEOUT = original


@pytest.mark.parametrize("seconds", [0.0, -1.0])
def test_pipe_timeout_rejects_non_positive_values(seconds: float) -> None:
    """
    Zero is a config error, not an opt-out.

    Reading it as "keep the default" would let a typo silently restore the
    2h47m detection wait, and taking it literally would fail the pool on its
    first step. Neither is what someone writing 0 wants, so it raises.
    """
    original = torchrl_utils.BATCHED_PIPE_TIMEOUT
    with pytest.raises(ValueError, match="must be positive or None"):
        apply_pipe_timeout(seconds)
    assert original == torchrl_utils.BATCHED_PIPE_TIMEOUT
