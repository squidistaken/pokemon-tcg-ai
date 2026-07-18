"""
Throughput benchmark for the TorchRL collection pipeline.

Measures collection throughput (fps, agent decisions per second) under the
random policy for a few env configurations, so the numbers in
``docs/torchrl_environment.md`` (Throughput section) can be regenerated.

Builds the config directly with OmegaConf and drives :class:`Trainer` rather
than going through the Hydra ``src/train.py`` entry point, so the benchmark
stays runnable regardless of the CLI wiring.

Run from the repository root::

    uv run python scripts/bench_throughput.py

Note the engine RNG is not seedable (see the docs), so exact fps varies a few
percent run to run and is hardware dependent; the ratios are stable.
"""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omegaconf import DictConfig, OmegaConf

from src.policies.random_masked_policy import RandomMaskedPolicy
from src.training.env_factory import make_env_factories
from src.training.trainer import Trainer

DEFAULT_DECK = str(REPO_ROOT / "decks" / "example.csv")


@dataclass(frozen=True)
class BenchCase:
    """
    One benchmark configuration to time.

    :param label: Human-readable name shown in the results table.
    :param num_workers: Number of environment instances.
    :param parallel: Use multiprocess ParallelEnv instead of single-process SerialEnv.
    """

    label: str
    num_workers: int
    parallel: bool


DEFAULT_CASES: list[BenchCase] = [
    BenchCase("naive single env", num_workers=1, parallel=True),
    BenchCase("SerialEnv(8)", num_workers=8, parallel=False),
    BenchCase("ParallelEnv(8, fork)", num_workers=8, parallel=True),
    BenchCase("ParallelEnv(16, fork)", num_workers=16, parallel=True),
]


def build_config(deck: str, num_workers: int, parallel: bool) -> DictConfig:
    """
    Build the minimal env config the factories expect.

    :param deck: Path to the deck CSV used for both seats (a mirror match).
    :param num_workers: Number of environment instances.
    :param parallel: Whether the trainer should use ParallelEnv.
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
            },
        }
    )


def run_case(case: BenchCase, deck: str, frames_per_batch: int, total_frames: int) -> float:
    """
    Time a single configuration and return its throughput.

    :param case: The configuration to benchmark.
    :param deck: Deck CSV path passed to both seats.
    :param frames_per_batch: Frames collected per collector iteration.
    :param total_frames: Total collection budget for the timed run.
    :return: Throughput in frames (agent decisions) per second.
    """
    config = build_config(deck, case.num_workers, case.parallel)
    trainer = Trainer(
        env_factories=make_env_factories(config),
        policy=RandomMaskedPolicy(),
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        use_parallel_env=case.parallel,
        mp_start_method="fork",
        serial_for_single=True,
    )
    stats = trainer.train()
    return stats["fps"]


def main() -> None:
    """
    Run every benchmark case and print a throughput/speedup table.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", default=DEFAULT_DECK, help="Deck CSV used for both seats.")
    parser.add_argument("--total-frames", type=int, default=65536, help="Frames per timed run.")
    parser.add_argument("--frames-per-batch", type=int, default=2048, help="Frames per collector batch.")
    args = parser.parse_args()

    results: list[tuple[BenchCase, float]] = []
    for case in DEFAULT_CASES:
        fps = run_case(case, args.deck, args.frames_per_batch, args.total_frames)
        results.append((case, fps))
        print(f"{case.label:24s} fps={fps:8.0f}")

    baseline_fps = results[0][1]
    print(f"\n=== throughput ({args.total_frames} frames), speedup vs {results[0][0].label} ===")
    for case, fps in results:
        print(f"{case.label:24s} {fps:8.0f} fps  ({fps / baseline_fps:5.2f}x)")


if __name__ == "__main__":
    main()
