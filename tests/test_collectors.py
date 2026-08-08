from itertools import pairwise

import pytest
import torch
from tensordict import TensorDict, lazy_stack
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.data import Bounded, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from src.training.collectors import (
    AsyncCollectorOptions,
    CollectorKind,
    _EnvStreamAssembler,
    _explain_mapping_exhaustion,
    _WorkerRolloutAssembler,
    build_collector,
    is_off_policy,
    parse_collector_kind,
    requires_weight_sync,
)


class CountingEnv(EnvBase):
    """
    Minimal two-action environment whose observation counts its own steps.

    Stands in for :class:`~src.env.tcg_env.TCGEnv` wherever a test is about the
    collector wiring rather than about the game: it is cheap enough to run under
    four collectors in a unit test, and its observation is a step counter, so a
    row of collected data can be checked for being a real contiguous trajectory
    rather than assumed to be one.

    :param seed: Unused; accepted for the factory signature.
    """

    def __init__(self, seed: int = 0) -> None:  # noqa: ARG002
        super().__init__(device="cpu")
        self.observation_spec = Composite(
            observation=Unbounded(shape=(1,), dtype=torch.float32)
        )
        self.action_spec = Categorical(2)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.done_spec = Composite(
            done=Bounded(low=0, high=1, shape=(1,), dtype=torch.bool),
            terminated=Bounded(low=0, high=1, shape=(1,), dtype=torch.bool),
            truncated=Bounded(low=0, high=1, shape=(1,), dtype=torch.bool),
        )
        self._step_count = 0

    def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:  # noqa: ARG002
        self._step_count = 0
        return TensorDict(
            {
                "observation": torch.zeros(1),
                "done": torch.zeros(1, dtype=torch.bool),
                "terminated": torch.zeros(1, dtype=torch.bool),
                "truncated": torch.zeros(1, dtype=torch.bool),
            },
            batch_size=[],
        )

    def _step(self, tensordict: TensorDict) -> TensorDict:  # noqa: ARG002
        self._step_count += 1
        done = self._step_count % 8 == 0
        return TensorDict(
            {
                "observation": torch.full((1,), float(self._step_count)),
                "reward": torch.zeros(1),
                "done": torch.tensor([done]),
                "terminated": torch.tensor([done]),
                "truncated": torch.zeros(1, dtype=torch.bool),
            },
            batch_size=[],
        )

    def _set_seed(self, seed: int | None) -> None:  # noqa: ARG002, PLR6301
        return None


class ThresholdPolicy(TensorDictModule):
    """
    Policy whose action is decided entirely by one parameter.

    Makes a weight update observable in the collected data: every action is
    ``1`` while the parameter is positive and ``0`` while it is negative, so a
    batch collected after a weight push either shows the new parameter or proves
    the push never arrived.
    """

    def __init__(self) -> None:
        module = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.fill_(1.0)
        super().__init__(module, in_keys=["observation"], out_keys=["_score"])

    def forward(self, tensordict, *args, **kwargs):  # noqa: ARG002
        """
        :param tensordict: Step tensordict carrying ``observation``.
        :return: The same tensordict with ``action`` written in.
        """
        weight = next(self.parameters())
        action = torch.ones_like(
            tensordict.get("observation")[..., 0], dtype=torch.int64
        ) * int(weight.item() > 0)
        tensordict.set("action", action)
        return tensordict


def make_counting_env() -> EnvBase:
    """
    Build one :class:`CountingEnv` (module-level so worker processes can pickle it).

    :return: A fresh counting environment.
    """
    return CountingEnv()


def make_transition(env_index: int, step: int) -> TensorDict:
    """
    Build one transition as the async collector would hand it over.

    :param env_index: Environment the transition came from.
    :param step: Step counter, used to check ordering survives regrouping.
    :return: A batch-size-``[]`` transition tensordict.
    """
    return TensorDict(
        {
            "observation": torch.full((1,), float(step)),
            "env_index": env_index,
            "next": TensorDict(
                {"observation": torch.full((1,), float(step + 1))}, batch_size=[]
            ),
        },
        batch_size=[],
    )


def test_parse_collector_kind_accepts_every_kind() -> None:
    """
    Every documented name resolves, so the config keys are not aspirational.
    """
    for kind in CollectorKind:
        assert parse_collector_kind(kind.value) is kind


def test_parse_collector_kind_rejects_unknown() -> None:
    """
    A typo fails loudly rather than falling back to the synchronous baseline.
    """
    with pytest.raises(ValueError, match="Unknown collector.type"):
        parse_collector_kind("multisync")


def test_weight_sync_and_off_policy_classification() -> None:
    """
    The two predicates answer different questions and must not be conflated.

    ``multi_sync`` runs a policy copy per worker (so it needs weight pushes) but
    its workers idle through the update (so its batches stay on-policy).
    ``async_batched`` is the mirror image: one in-process policy, no push
    needed, yet collection overlaps the update.
    """
    assert requires_weight_sync(CollectorKind.MULTI_SYNC)
    assert not is_off_policy(CollectorKind.MULTI_SYNC)
    assert not requires_weight_sync(CollectorKind.ASYNC_BATCHED)
    assert is_off_policy(CollectorKind.ASYNC_BATCHED)
    assert not requires_weight_sync(CollectorKind.SYNC)
    assert not is_off_policy(CollectorKind.SYNC)
    assert requires_weight_sync(CollectorKind.MULTI_ASYNC)
    assert is_off_policy(CollectorKind.MULTI_ASYNC)


def test_worker_rollout_assembler_stacks_rows() -> None:
    """
    Rollouts are buffered until a full batch exists, then stacked one per row.
    """
    assembler = _WorkerRolloutAssembler(rows=3)
    rollouts = [
        TensorDict({"observation": torch.arange(4.0) + 10 * index}, batch_size=[4])
        for index in range(3)
    ]
    assert assembler(rollouts[0]) is None
    assert assembler(rollouts[1]) is None
    batch = assembler(rollouts[2])

    assert batch is not None
    assert tuple(batch.shape) == (3, 4)
    for index in range(3):
        assert torch.equal(batch["observation"][index], torch.arange(4.0) + 10 * index)


def test_env_stream_assembler_separates_interleaved_environments() -> None:
    """
    Interleaved transitions are regrouped into one contiguous row per environment.

    This is the property everything downstream depends on: a row handed to GAE
    must be one environment's consecutive steps. Feeding the flat batch straight
    through would put env 0's step 1 next to env 1's step 0 and call the pair a
    trajectory.
    """
    assembler = _EnvStreamAssembler(num_envs=2)
    interleaved = lazy_stack(
        [
            make_transition(0, 0),
            make_transition(1, 0),
            make_transition(1, 1),
            make_transition(0, 1),
        ]
    )

    batch = assembler(interleaved)

    assert batch is not None
    assert tuple(batch.shape) == (2, 2)
    assert torch.equal(batch["observation"][0].reshape(-1), torch.tensor([0.0, 1.0]))
    assert torch.equal(batch["observation"][1].reshape(-1), torch.tensor([0.0, 1.0]))
    # The index only exists to make the split possible; carrying a non-tensor
    # entry into the update would break the reshape/permutation the epoch loop does.
    assert "env_index" not in list(batch.keys())


def test_env_stream_assembler_carries_the_remainder() -> None:
    """
    A faster environment's surplus leads its next row rather than being dropped.

    The batch is rectangular, so its width is what the *slowest* environment can
    supply; the difference has to survive to the next batch, and it has to
    survive in order, or the fast environment's row silently skips steps.
    """
    assembler = _EnvStreamAssembler(num_envs=2)
    first = lazy_stack(
        [make_transition(0, 0), make_transition(0, 1), make_transition(1, 0)]
    )

    batch = assembler(first)
    assert batch is not None
    assert tuple(batch.shape) == (2, 1)

    second = lazy_stack([make_transition(1, 1), make_transition(1, 2)])
    batch = assembler(second)

    assert batch is not None
    assert tuple(batch.shape) == (2, 1)
    # Env 0's step 1 was buffered by the first call, not discarded.
    assert torch.equal(batch["observation"][0].reshape(-1), torch.tensor([1.0]))
    assert torch.equal(batch["observation"][1].reshape(-1), torch.tensor([1.0]))


def test_env_stream_assembler_waits_for_a_silent_environment() -> None:
    """
    No batch is emitted while an environment has contributed nothing.

    Emitting one would mean either a ragged batch or a row invented for the
    missing environment.
    """
    assembler = _EnvStreamAssembler(num_envs=2)
    assert assembler(lazy_stack([make_transition(0, 0)])) is None


def test_env_stream_assembler_requires_the_environment_index() -> None:
    """
    Without the index the batch cannot be split, and that is an error, not a
    reason to guess.
    """
    assembler = _EnvStreamAssembler(num_envs=1)
    with pytest.raises(KeyError, match="env_index"):
        assembler(TensorDict({"observation": torch.zeros(2, 1)}, batch_size=[2]))


@pytest.mark.parametrize(
    "kind",
    [CollectorKind.SYNC, CollectorKind.MULTI_SYNC, CollectorKind.MULTI_ASYNC],
)
def test_every_kind_yields_row_time_batches(kind: CollectorKind) -> None:
    """
    Whichever collector is selected, the learner sees ``(rows, time)`` batches
    of contiguous trajectory, which is the contract PPO is written against.
    """
    from torchrl.envs import SerialEnv

    factories = [make_counting_env, make_counting_env]
    collector = build_collector(
        kind,
        env_factories=factories,
        make_vec_env=lambda: SerialEnv(2, factories),
        policy=ThresholdPolicy(),
        frames_per_batch=8,
        total_frames=16,
        collector_kwargs={},
        options=AsyncCollectorOptions(),
    )
    try:
        batch = next(iter(collector))
        assert batch.ndim == 2
        assert batch.numel() == 8
        for row in batch["observation"]:
            steps = row.reshape(-1)
            # Consecutive within an episode; the counter restarts at a reset.
            assert all(
                later == earlier + 1 or later == 1.0
                for earlier, later in pairwise(steps)
            )
    finally:
        collector.shutdown()


def test_multi_sync_workers_collect_with_the_learner_s_current_weights() -> None:
    """
    Weight changes in this process reach the worker processes' policy copies.

    The failure this guards against is silent and total: if they did not, the
    multiprocess collectors would keep collecting at full speed against the
    weights their workers were forked with, and the run would look healthy while
    learning nothing. The policy's action here is decided by the sign of its one
    parameter, so the collected actions say directly which weights the workers
    used.

    What the test does *not* pin down is the mechanism. On a CPU policy torchrl
    puts the weights in shared memory, so the change propagates whether or not
    :meth:`update_policy_weights_` is called; the call is what covers the cases
    where it does not (a CUDA policy is not shared, and is copied over the pipe
    on request), and it is called here because that is what the trainer does.
    """
    policy = ThresholdPolicy()
    factories = [make_counting_env, make_counting_env]
    collector = build_collector(
        CollectorKind.MULTI_SYNC,
        env_factories=factories,
        make_vec_env=lambda: make_counting_env(),
        policy=policy,
        frames_per_batch=8,
        total_frames=32,
        collector_kwargs={},
        options=AsyncCollectorOptions(),
    )
    try:
        iterator = iter(collector)
        assert (next(iterator)["action"] == 1).all()

        with torch.no_grad():
            next(policy.parameters()).fill_(-1.0)
        collector.update_policy_weights_()

        # The batch in flight when the weights changed may still carry the old
        # action, so the assertion is on the batch after it.
        next(iterator)
        assert (next(iterator)["action"] == 0).all()
    finally:
        collector.shutdown()


@pytest.mark.parametrize("kind", [CollectorKind.MULTI_SYNC, CollectorKind.MULTI_ASYNC])
def test_multiprocess_collectors_force_fork(
    kind: CollectorKind, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Building a multiprocess collector forces ``fork`` process-wide itself.

    TorchRL resolves its multiprocessing context from the global start method
    and falls back to ``spawn`` when it is unset. Under ``spawn`` everything the
    env factories close over is pickled, which hands each worker a private copy
    of what was meant to be shared memory -- the curriculum's distribution
    channel above all, which then freezes at whatever was published before
    collection began, silently.

    The guarantee is asserted here rather than in the trainer because
    ``build_collector`` is what starts the processes; leaving it to callers is
    what made this depend on test ordering in the first place.
    """
    import torch.multiprocessing as torch_mp

    recorded: list[str] = []
    monkeypatch.setattr(
        torch_mp,
        "set_start_method",
        lambda method, **_kwargs: recorded.append(method),
    )

    # Constructing the collector is enough; it is never iterated, so no worker
    # is actually started and the patched setter stays harmless.
    build_collector(
        kind,
        env_factories=[make_counting_env, make_counting_env],
        make_vec_env=make_counting_env,
        policy=ThresholdPolicy(),
        frames_per_batch=8,
        total_frames=8,
        collector_kwargs={},
        options=AsyncCollectorOptions(),
        mp_start_method="fork",
    ).shutdown()

    assert recorded == ["fork"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_weight_sync_crosses_the_device_boundary() -> None:
    """
    A CUDA learner's weights reach worker policy copies held on CPU.

    This is the configuration the throughput work recommends -- ``multi_sync``
    with ``agent.device=cuda`` and ``agent.collector_device=cpu``, so the update
    keeps the GPU while collection runs one CPU policy per worker instead of one
    CUDA context per worker. It only works if the weight push moves devices on
    the way, and the failure mode if it does not is the silent one: collection
    continues at full speed against stale weights.
    """
    policy = ThresholdPolicy().to("cuda")
    collector = build_collector(
        CollectorKind.MULTI_SYNC,
        env_factories=[make_counting_env, make_counting_env],
        make_vec_env=make_counting_env,
        policy=policy,
        frames_per_batch=8,
        total_frames=64,
        collector_kwargs={"policy_device": "cpu"},
        options=AsyncCollectorOptions(),
    )
    try:
        iterator = iter(collector)
        assert (next(iterator)["action"] == 1).all()

        with torch.no_grad():
            next(policy.parameters()).fill_(-1.0)
        collector.update_policy_weights_()

        # The batch already in flight may carry the old action; assert on the next.
        next(iterator)
        assert (next(iterator)["action"] == 0).all()
    finally:
        collector.shutdown()


@pytest.mark.parametrize("kind", [CollectorKind.MULTI_SYNC, CollectorKind.MULTI_ASYNC])
def test_cuda_collection_policy_is_rejected_under_fork(kind: CollectorKind) -> None:
    """
    A CUDA collection policy under the forking collectors fails at construction.

    Reached by the natural configuration ``agent.device=cuda`` with
    ``collector.type=multi_sync`` and no ``agent.collector_device``: CUDA cannot
    initialize in a forked child, so this used to die several frames inside
    torch's IPC machinery with a message that named neither the collector nor
    the setting to change. Needs no GPU, since the rejection reads the
    configured device rather than the hardware.
    """
    with pytest.raises(ValueError, match="agent.collector_device=cpu"):
        build_collector(
            kind,
            env_factories=[make_counting_env, make_counting_env],
            make_vec_env=make_counting_env,
            policy=ThresholdPolicy(),
            frames_per_batch=8,
            total_frames=16,
            collector_kwargs={"policy_device": "cuda"},
            options=AsyncCollectorOptions(),
            mp_start_method="fork",
        )


def test_mapping_exhaustion_is_recognized_through_the_cause_chain() -> None:
    """
    The mmap-limit failure is named, and unrelated RuntimeErrors are left alone.

    torchrl reports it as a generic "worker thread raised an exception" with the
    real cause chained underneath, so matching only the outermost message would
    miss it and matching too loosely would relabel every collector failure.
    """
    root = RuntimeError("unable to mmap 124 bytes from file <>: Cannot allocate memory")
    wrapped = RuntimeError("A collector worker thread raised an exception.")
    wrapped.__cause__ = root
    explanation = _explain_mapping_exhaustion(wrapped)
    assert explanation is not None
    assert "vm.max_map_count" in explanation
    assert "multi_sync" in explanation

    assert _explain_mapping_exhaustion(RuntimeError("worker 6 died")) is None
