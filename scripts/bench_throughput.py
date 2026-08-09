"""
Throughput benchmark and collection profiler for the TorchRL pipeline.

Two jobs, selected by ``--mode``:

``throughput`` (default)
    Times collection under a few environment/collector configurations and
    prints an fps table, so the numbers in ``docs/training-performance.md`` can
    be regenerated.

``profile``
    Answers *where the time goes*, which fps alone cannot. For one
    configuration it reports the wall clock split three ways -- policy forward,
    environment step (the IPC round trip plus the workers' own work), and the
    collector's tensordict bookkeeping -- alongside per-process CPU for the main
    process and the worker pool. This is what issue #86 asks for before any
    collector is changed: the suspicion was that the main process spends its 33
    ms per batched step on tensordict stacking and IPC of a large nested
    observation, and that had never been measured.

    The three-way split is only meaningful for ``--collector sync``, where the
    policy runs in this process. Under the other kinds inference happens in the
    workers (or behind an inference server), so the split is reported as not
    applicable and the CPU figures are what carry the result.

Either mode can run against one of two scenarios. By default the config is
built directly with OmegaConf: a mirror match with no deck pool and no
self-play league, cheap and reproducible, but a clean pipeline rather than
what a training run sees. Passing ``--config-name`` instead composes the real
Hydra config and reuses the same factories ``src/train.py`` does, which brings
in the deck pool and the league whose opponent forward runs inside every
worker. Which scenario is measured changes the answer, so prefer the second
when the question is about a run you actually intend to launch.

Run from the repository root::

    uv run python scripts/bench_throughput.py
    uv run python scripts/bench_throughput.py --mode profile --policy ppo \\
        --collector sync --num-workers 16

    # Against a real run's configuration, league included.
    uv run python scripts/bench_throughput.py --mode profile --policy ppo \\
        --collector multi_sync --device cpu --num-workers 16 \\
        --config-name ppo_selfplay_multideck \\
        --override model/backbone=transformer --override model/head=pointer \\
        --checkpoint-dir outputs/<group>/<run>/checkpoints

Note the engine RNG is not seedable (see the docs), so exact fps varies a few
percent run to run and is hardware dependent; the ratios are stable.
"""

import argparse
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch.multiprocessing as torch_mp
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torchrl.envs import EnvBase

from src.policies.random_masked_policy import RandomMaskedPolicy
from src.training.collectors import CollectorKind, TrainingCollector
from src.training.env_factory import make_env_factories
from src.training.trainer import Trainer

DEFAULT_DECK = str(REPO_ROOT / "decks" / "example.csv")

#: Seconds between CPU samples.
_CPU_SAMPLE_INTERVAL = 0.5

#: Clock ticks per second.
_CLOCK_TICKS = os.sysconf("SC_CLK_TCK")


@dataclass(frozen=True)
class BenchCase:
    """
    One benchmark configuration to time.

    :param label: Human-readable name shown in the results table.
    :param num_workers: Number of environment instances.
    :param parallel: Use multiprocess ParallelEnv instead of single-process
        SerialEnv. Read only by the ``sync`` collector; the others own their
        worker processes directly.
    :param collector: Which collector kind to build.
    """

    label: str
    num_workers: int
    parallel: bool
    collector: CollectorKind = CollectorKind.SYNC


DEFAULT_CASES: list[BenchCase] = [
    BenchCase("naive single env", num_workers=1, parallel=True),
    BenchCase("SerialEnv(8)", num_workers=8, parallel=False),
    BenchCase("ParallelEnv(8, fork)", num_workers=8, parallel=True),
    BenchCase("ParallelEnv(16, fork)", num_workers=16, parallel=True),
    BenchCase(
        "MultiSync(16)",
        num_workers=16,
        parallel=True,
        collector=CollectorKind.MULTI_SYNC,
    ),
    BenchCase(
        "MultiAsync(16)",
        num_workers=16,
        parallel=True,
        collector=CollectorKind.MULTI_ASYNC,
    ),
]


@dataclass
class StepTimings:
    """
    Wall-clock seconds attributed to each part of the main process's loop.

    :param policy: Inside the policy's forward pass.
    :param env_step: Inside ``env.step_and_maybe_reset``; for a ParallelEnv,
        the round trip to the workers and back, so it covers both the IPC and
        the environment work the parent is blocked on.
    :param total: Wall clock across the whole collection loop.
    """

    policy: float = 0.0
    env_step: float = 0.0
    total: float = 0.0

    @property
    def other(self) -> float:
        """
        :return: Time in the collector's own tensordict bookkeeping.
        """
        return max(0.0, self.total - self.policy - self.env_step)


@dataclass
class CpuUsage:
    """
    CPU seconds consumed over a run, split between this process and its workers.

    :param main: CPU seconds burned by the main process.
    :param workers: CPU seconds summed over every worker process.
    :param elapsed: Wall-clock seconds the sampling covered.
    :param worker_count: Distinct worker processes seen.
    """

    main: float = 0.0
    workers: float = 0.0
    elapsed: float = 0.0
    worker_count: int = 0

    @property
    def main_percent(self) -> float:
        """:return: Main-process CPU as a percentage of one core."""
        return 100.0 * self.main / max(self.elapsed, 1e-9)

    @property
    def worker_percent(self) -> float:
        """:return: Combined worker CPU as a percentage of one core."""
        return 100.0 * self.workers / max(self.elapsed, 1e-9)

    @property
    def total_percent(self) -> float:
        """:return: Whole-pipeline CPU as a percentage of one core."""
        return self.main_percent + self.worker_percent


@dataclass
class ProfileResult:
    """
    Everything one profiled run reports.

    :param fps: Throughput in frames (agent decisions) per second.
    :param timings: Main-process step attribution.
    :param cpu: Per-process CPU usage.
    :param attributable: Whether the step attribution is meaningful, i.e.
        whether the policy actually ran in this process.
    """

    fps: float
    timings: StepTimings
    cpu: CpuUsage
    attributable: bool


def _process_cpu_seconds(pid: int) -> float:
    """
    Read one process's cumulative CPU time.

    :param pid: Process to read.
    :return: User plus system CPU seconds, or 0.0 if the process is gone.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return 0.0
    fields = stat[stat.rfind(")") + 2 :].split()
    # Fields 14/15 counting from 1 are the 12th/13th after the state field.
    return (int(fields[11]) + int(fields[12])) / _CLOCK_TICKS


class _CpuSampler:
    """
    Background sampler recording CPU time for this process and its children.

    :param interval: Seconds between samples.
    """

    def __init__(self, interval: float = _CPU_SAMPLE_INTERVAL) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._per_pid: dict[int, float] = {}
        self._baseline = 0.0
        self._started = 0.0

    def __enter__(self) -> Self:
        """:return: The started sampler."""
        self._baseline = _process_cpu_seconds(os.getpid())
        self._started = time.time()
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Stop sampling; a last sample is taken before the thread joins."""
        self._sample()
        self._stop.set()
        self._thread.join(timeout=self._interval * 4)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def _sample(self) -> None:
        """Record the current CPU total for every live worker process."""
        for child in torch_mp.active_children():
            if child.pid is None:
                continue
            seconds = _process_cpu_seconds(child.pid)
            self._per_pid[child.pid] = max(self._per_pid.get(child.pid, 0.0), seconds)

    def result(self) -> CpuUsage:
        """
        :return: CPU usage over the sampled window.
        """
        return CpuUsage(
            main=_process_cpu_seconds(os.getpid()) - self._baseline,
            workers=sum(self._per_pid.values()),
            elapsed=time.time() - self._started,
            worker_count=len(self._per_pid),
        )


class _ProfilingTrainer(Trainer):
    """
    Trainer that times the main process's collection loop.

    :param timings: Accumulator the instrumentation writes into.
    :param kwargs: Forwarded to :class:`~src.training.trainer.Trainer`.
    """

    def __init__(
        self, timings: StepTimings, device: str = "cpu", **kwargs: object
    ) -> None:
        super().__init__(**kwargs)  # pyright: ignore[reportArgumentType]
        self._timings = timings
        self._policy_device = device

    def _collector_kwargs(self) -> dict:
        """
        :return: ``policy_device``, so a CUDA policy is fed CUDA tensors. The
            environments step on CPU regardless, exactly as in training.
        """
        return {"policy_device": self._policy_device}

    def _make_collector(self, remaining_frames: int) -> TrainingCollector:
        collector = super()._make_collector(remaining_frames)
        _instrument(collector, self._timings)
        return collector


def _instrument(collector: TrainingCollector, timings: StepTimings) -> bool:
    """
    Wrap a collector's policy and environment step with timers.

    :param collector: Collector to instrument, in place.
    :param timings: Accumulator to add elapsed seconds to.
    :return: True if both hooks were installed. False means the collector runs
        its policy somewhere other than this process, so the attribution is not
        available -- reported rather than silently zeroed.
    """
    policy = getattr(collector, "_wrapped_policy", None)
    env = getattr(collector, "env", None)
    if policy is None or env is None or not hasattr(env, "step_and_maybe_reset"):
        return False

    def timed_policy(*args: object, **kwargs: object) -> object:
        start = time.perf_counter()
        try:
            return policy(*args, **kwargs)
        finally:
            timings.policy += time.perf_counter() - start

    original_step = env.step_and_maybe_reset

    def timed_step(*args: object, **kwargs: object) -> object:
        start = time.perf_counter()
        try:
            return original_step(*args, **kwargs)
        finally:
            timings.env_step += time.perf_counter() - start

    setattr(collector, "_wrapped_policy", timed_policy)  # noqa: B010
    env.step_and_maybe_reset = timed_step
    return True


def build_config(
    deck: str, num_workers: int, parallel: bool, encoder: str = "structured"
) -> DictConfig:
    """
    Build the minimal env config the factories expect.

    :param deck: Path to the deck CSV used for both seats (a mirror match).
    :param num_workers: Number of environment instances.
    :param parallel: Whether the trainer should use ParallelEnv.
    :param encoder: Observation encoder name.
    :return: OmegaConf config with ``seed`` and an ``env`` section.
    """
    return OmegaConf.create(
        {
            "seed": 0,
            "env": {
                "deck0": deck,
                "deck1": deck,
                "max_options": 96,
                "num_workers": num_workers,
                "parallel": parallel,
                "mp_start_method": "fork",
                "serial_for_single": True,
                "encoder": encoder,
            },
        }
    )


def build_policy(
    kind: str,
    config: DictConfig,
    device: str = "cpu",
    model_overrides: list[str] | None = None,
) -> nn.Module:
    """
    Build the collection policy the benchmark runs.

    :param kind: ``random`` for the uniform masked baseline, ``ppo`` for the
        real actor built from the Hydra model config.
    :param config: Env config, extended in place with the composed model
        section when ``ppo`` is selected.
    :param device: Device the policy is placed on.
    :param model_overrides: Extra Hydra overrides selecting the architecture,
    :return: A policy module the collector can drive.
    :raises ValueError: If ``kind`` is neither of the two.
    """
    if kind == "random":
        return RandomMaskedPolicy()
    if kind != "ppo":
        raise ValueError(f"Unknown --policy '{kind}'; expected 'random' or 'ppo'.")

    from hydra import compose, initialize_config_dir
    from torchrl.modules import ActorValueOperator

    from src.policies.ppo_actor import build_actor_critic, build_ppo_operator
    from src.training.env_factory import build_probe_specs

    with initialize_config_dir(config_dir=str(REPO_ROOT / "conf"), version_base=None):
        model_cfg = compose(
            config_name="config", overrides=["agent=ppo", *(model_overrides or [])]
        )
    config.model = model_cfg.model

    obs_spec, action_spec = build_probe_specs(config)
    operator: ActorValueOperator = build_ppo_operator(  # pyright: ignore[reportAssignmentType]
        build_actor_critic(config, obs_spec, action_spec), action_spec
    ).to(device)
    return operator.get_policy_operator().select_out_keys("action", "action_log_prob")


@dataclass(frozen=True)
class RunScenario:
    """
    The environment factories and policy one profiled configuration runs.

    :param env_factories: One factory per worker, already carrying whatever
        opponent the configuration puts inside the environment.
    :param policy: Collection policy.
    :param collector_device: Device the collection policy runs on, which under
        the multiprocess collectors is deliberately not the update's device.
    """

    env_factories: list[Callable[[], EnvBase]]
    policy: nn.Module
    collector_device: str


def build_synthetic_scenario(
    case: BenchCase,
    deck: str,
    policy_kind: str,
    device: str,
    model_overrides: list[str] | None,
) -> RunScenario:
    """
    Build the self-contained scenario: a mirror match, no deck pool, no league.

    Cheap and reproducible, and what the throughput sweep uses. It measures a
    clean collection pipeline rather than what a training run sees, because the
    self-play opponent forward that runs inside every worker is absent. Use
    :func:`build_run_scenario` to profile a real run.

    :param case: Configuration being benchmarked.
    :param deck: Deck CSV path passed to both seats.
    :param policy_kind: Which policy to collect with (see :func:`build_policy`).
    :param device: Device the policy runs on.
    :param model_overrides: Hydra overrides selecting the architecture.
    :return: The scenario to time.
    """
    config = build_config(deck, case.num_workers, case.parallel)
    policy = build_policy(policy_kind, config, device, model_overrides)
    return RunScenario(make_env_factories(config), policy, device)


def build_run_scenario(
    case: BenchCase,
    config_name: str,
    overrides: list[str],
    device: str,
    checkpoint_dir: str | None,
) -> RunScenario:
    """
    Build the scenario an actual training run collects with.

    Composes the same Hydra config ``src/train.py`` would and reuses the same
    factory helpers, so the profile includes what the synthetic scenario leaves
    out: the deck pool and its sampling, and above all the self-play league,
    whose opponent forward runs *inside every worker* and which
    ``docs/training-performance.md`` section 3 measured at 46% of throughput.
    That matters for the collector choice specifically, because the
    multiprocess kinds add a second per-worker forward on top of it.

    The league is pointed at an existing snapshot directory rather than the run
    directory ``src/train.py`` would create, since a fresh run starts with an
    empty pool and a random warmup opponent, i.e. without the very cost this is
    here to measure.

    :param case: Configuration being benchmarked; supplies the worker count.
    :param config_name: Hydra config name, e.g. ``ppo_selfplay_multideck``.
    :param overrides: Hydra overrides, as passed on a training command line.
    :param device: Device the collection policy runs on.
    :param checkpoint_dir: Directory of existing snapshots the workers' leagues
        draw from. None leaves the pool empty, which profiles the warmup phase.
    :return: The scenario to time.
    """
    from hydra import compose, initialize_config_dir
    from torchrl.modules import ActorValueOperator

    from src.policies.ppo_actor import build_actor_critic, build_ppo_operator
    from src.training.env_factory import build_probe_specs
    from src.training.self_play import build_opponent_factory

    with initialize_config_dir(config_dir=str(REPO_ROOT / "conf"), version_base=None):
        config = compose(
            config_name=config_name,
            overrides=[*overrides, f"env.num_workers={case.num_workers}"],
        )

    obs_spec, action_spec = build_probe_specs(config)
    operator: ActorValueOperator = build_ppo_operator(  # pyright: ignore[reportAssignmentType]
        build_actor_critic(config, obs_spec, action_spec), action_spec
    ).to(device)
    policy = operator.get_policy_operator().select_out_keys("action", "action_log_prob")

    opponent_factory = (
        build_opponent_factory(config, obs_spec, action_spec, checkpoint_dir)
        if checkpoint_dir is not None
        else None
    )
    factories = make_env_factories(config, opponent_factory=opponent_factory)
    return RunScenario(factories, policy, device)


def run_case(
    case: BenchCase,
    deck: str,
    frames_per_batch: int,
    total_frames: int,
    policy_kind: str = "random",
    device: str = "cpu",
    model_overrides: list[str] | None = None,
    scenario: RunScenario | None = None,
) -> ProfileResult:
    """
    Time a single configuration.

    :param case: The configuration to benchmark.
    :param deck: Deck CSV path passed to both seats.
    :param frames_per_batch: Frames collected per collector iteration.
    :param total_frames: Total collection budget for the timed run.
    :param policy_kind: Which policy to collect with (see :func:`build_policy`).
    :param device: Device the policy runs on.
    :param model_overrides: Hydra overrides selecting the architecture.
    :param scenario: Prebuilt environments and policy. None builds the
        synthetic mirror-match scenario from the arguments above.
    :return: Throughput, attribution and CPU usage for the run.
    """
    if scenario is None:
        scenario = build_synthetic_scenario(
            case, deck, policy_kind, device, model_overrides
        )
    timings = StepTimings()
    trainer = _ProfilingTrainer(
        timings,
        device=scenario.collector_device,
        env_factories=scenario.env_factories,
        policy=scenario.policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        use_parallel_env=case.parallel,
        mp_start_method="fork",
        serial_for_single=True,
        collector_type=case.collector,
    )
    with _CpuSampler() as sampler:
        started = time.perf_counter()
        stats = trainer.train()
        timings.total = time.perf_counter() - started
    return ProfileResult(
        fps=stats["fps"],
        timings=timings,
        cpu=sampler.result(),
        # The attribution hooks only fire when the policy runs here, which is
        # exactly when the two buckets add up to less than the wall clock.
        attributable=timings.policy > 0.0 or timings.env_step > 0.0,
    )


def print_throughput(results: list[tuple[BenchCase, ProfileResult]]) -> None:
    """
    Print the fps table and the speedup against the first case.

    :param results: One entry per benchmarked case, in run order.
    """
    baseline_fps = results[0][1].fps
    print("\n=== throughput, speedup vs " + results[0][0].label + " ===")
    print(f"{'case':24s} {'fps':>8s} {'speedup':>8s} {'CPU %':>9s} {'main %':>8s}")
    for case, result in results:
        print(
            f"{case.label:24s} {result.fps:8.0f} {result.fps / baseline_fps:7.2f}x "
            f"{result.cpu.total_percent:8.0f} {result.cpu.main_percent:8.0f}"
        )


def print_profile(case: BenchCase, result: ProfileResult) -> None:
    """
    Print the step attribution and CPU breakdown for one configuration.

    :param case: The configuration that was profiled.
    :param result: What the run measured.
    """
    timings = result.timings
    cpu = result.cpu
    print(f"\n=== profile: {case.label} ({case.collector.value}) ===")
    print(f"{'throughput':22s} {result.fps:8.0f} fps")
    print(f"{'wall clock':22s} {timings.total:8.1f} s")
    if result.attributable:
        for label, seconds in (
            ("policy forward", timings.policy),
            ("env step (IPC + envs)", timings.env_step),
            ("collector tensordict", timings.other),
        ):
            share = 100.0 * seconds / max(timings.total, 1e-9)
            print(f"  {label:20s} {seconds:8.1f} s  {share:5.1f}%")
    else:
        print(
            "  step attribution not available: this collector runs the policy "
            "outside the main process."
        )
    print(f"{'main process CPU':22s} {cpu.main_percent:8.0f}% of one core")
    print(
        f"{'worker CPU':22s} {cpu.worker_percent:8.0f}% of one core "
        f"({cpu.worker_count} processes)"
    )
    print(f"{'total CPU':22s} {cpu.total_percent:8.0f}% of one core")


def parse_args() -> argparse.Namespace:
    """
    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("throughput", "profile"),
        default="throughput",
        help="Sweep the standard cases, or profile one configuration in detail.",
    )
    parser.add_argument(
        "--deck", default=DEFAULT_DECK, help="Deck CSV used for both seats."
    )
    parser.add_argument(
        "--total-frames", type=int, default=65536, help="Frames per timed run."
    )
    parser.add_argument(
        "--frames-per-batch", type=int, default=2048, help="Frames per collector batch."
    )
    parser.add_argument(
        "--policy",
        choices=("random", "ppo"),
        default="random",
        help="Collection policy. 'ppo' builds the real actor, which is what "
        "makes the policy-forward share of the profile meaningful.",
    )
    parser.add_argument(
        "--collector",
        choices=tuple(kind.value for kind in CollectorKind),
        default=CollectorKind.SYNC.value,
        help="Collector kind to profile (--mode profile only).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Environment count to profile (--mode profile only).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device the collection policy runs on. Which part of the loop "
        "dominates depends on this, so it is worth measuring both.",
    )
    parser.add_argument(
        "--model-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Hydra override selecting the architecture, repeatable, e.g. "
        "--model-override model/backbone=transformer. Ignored under --policy random.",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Profile a real training config by name, e.g. ppo_selfplay_multideck. "
        "Composes it exactly as src/train.py would, so the deck pool and the "
        "self-play league are included. Without this the benchmark runs a "
        "synthetic mirror match with no league, which understates worker cost.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Hydra override applied to --config-name, repeatable. Pass the same "
        "ones the training script does.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Existing snapshot directory the workers' self-play leagues draw "
        "from (--config-name only). Without it the pool is empty and every "
        "worker faces the random warmup opponent, which is not what a run in "
        "progress costs.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run the selected mode and print its report.
    """
    args = parse_args()

    if args.mode == "profile":
        case = BenchCase(
            label=f"{args.policy} policy, {args.num_workers} envs",
            num_workers=args.num_workers,
            parallel=True,
            collector=CollectorKind(args.collector),
        )
        scenario = (
            build_run_scenario(
                case,
                args.config_name,
                args.override,
                args.device,
                args.checkpoint_dir,
            )
            if args.config_name
            else None
        )
        result = run_case(
            case,
            args.deck,
            args.frames_per_batch,
            args.total_frames,
            args.policy,
            args.device,
            args.model_override,
            scenario,
        )
        print_profile(case, result)
        return

    results: list[tuple[BenchCase, ProfileResult]] = []
    for case in DEFAULT_CASES:
        result = run_case(
            case,
            args.deck,
            args.frames_per_batch,
            args.total_frames,
            args.policy,
            args.device,
            args.model_override,
        )
        results.append((case, result))
        print(f"{case.label:24s} fps={result.fps:8.0f}")
    print_throughput(results)


if __name__ == "__main__":
    main()
