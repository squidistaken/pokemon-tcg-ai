"""
Compare training arms: fixed-deck, uniform deck sampling, and PLR curriculum.

Runs up to three arms per seed -- fixed (single deck pair, ``env=default``),
uniform (pooled i.i.d., ``env=multideck_v2``), and curriculum (same pool under
PLR, ``env=curriculum_v2``) -- and extracts the final evaluation win rate of
each against every reference opponent.

All arms use ``train=ppo_selfplay``, which scores each round against both the
oldest league snapshot and a random opponent (``train.eval_opponents``). The
snapshot keeps discriminating once random saturates; random stays comparable
across arms and runs.

Every arm logs to W&B under a shared group, one run per arm, so their curves
overlay directly. Pass ``--no-wandb`` to turn that off.

Usage::

    # All three arms, curriculum first, 2M frames
    python -m scripts.curriculum_ab --arms curriculum,uniform,fixed \
        --seeds 1 --frames 2000000 --workers 16

    # Curriculum vs uniform only
    python -m scripts.curriculum_ab --arms uniform,curriculum --seeds 1

    # Quick local check with no W&B
    python -m scripts.curriculum_ab --arms curriculum --frames 50000 --no-wandb
"""

import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]

#: Wall-clock stamp of this invocation, so re-running an arm at the same seed
#: writes somewhere fresh. ``src.train`` refuses to reuse a snapshot directory:
#: a shared one mixes the previous run's league into this one.
LAUNCH_STAMP = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")

#: Per-arm environment config. ``curriculum`` is None for arms with no deck
#: pool, where the curriculum keys do not apply at all.
#:
#: The two pooled arms must name configs over the *same* corpus and holdout
#: split -- ``multideck_v2`` and ``curriculum_v2`` differ only in whether the
#: curriculum is enabled. Pointing them at different pools would confound the
#: comparison with the corpus change.
ARM_SPECS = {
    "fixed": {
        "env": "default",
        "curriculum": None,
    },
    "uniform": {
        "env": "multideck_v2",
        "curriculum": False,
    },
    "curriculum": {
        "env": "curriculum_v2",
        "curriculum": True,
    },
}


@dataclass
class ArmResult:
    """
    Outcome of one arm across seeds.

    :param name: Arm label.
    :param win_rates: Per seed, the final win rate against each reference
        opponent (``{"random": 0.8, "first_snapshot": 0.4}``).
    :param failures: Seeds whose run did not produce a result.
    """

    name: str
    win_rates: list[dict[str, float]] = field(default_factory=list)
    failures: list[int] = field(default_factory=list)

    def mean(self, opponent: str) -> float:
        """
        Mean win rate against one reference opponent across seeds.

        :param opponent: Reference opponent label.
        :return: Mean win rate, or NaN if no seed reported that opponent.
        """
        values = [rates[opponent] for rates in self.win_rates if opponent in rates]
        return statistics.fmean(values) if values else float("nan")

    @property
    def opponents(self) -> list[str]:
        """
        :return: Reference opponents seen across this arm's runs, sorted.
        """
        return sorted({name for rates in self.win_rates for name in rates})


def run_arm(
    seed: int,
    arm: str,
    frames: int,
    workers: int,
    eval_interval: int,
    eval_episodes: int,
    output_root: Path,
    wandb_group: str | None,
    extra: list[str],
) -> dict[str, float] | None:
    """
    Train one arm at one seed and return its final eval win rates.

    :param seed: Seed shared across arms for paired comparison.
    :param arm: Arm key (``"fixed"``, ``"uniform"``, ``"curriculum"``).
    :param frames: Total environment frames.
    :param workers: Environment workers.
    :param eval_interval: Frames between evaluation rounds.
    :param eval_episodes: Episodes per reference opponent per round.
    :param output_root: Root the run's own directory is created under.
    :param wandb_group: W&B group tying this experiment's arms together; None
        disables W&B for the run.
    :param extra: Additional Hydra overrides.
    :return: Final win rate per reference opponent, or None if the run
        produced no evaluation.
    """
    spec = ARM_SPECS[arm]
    run_dir = output_root / LAUNCH_STAMP / f"{arm}_seed{seed}"
    command = [
        sys.executable,
        "-m",
        "src.train",
        "agent=ppo",
        f"env={spec['env']}",
        "train=ppo_selfplay",
        f"seed={seed}",
        "set_seed=true",
        "agent.device=cuda",
        f"env.num_workers={workers}",
        f"collector.total_frames={frames}",
        f"train.eval_interval={eval_interval}",
        f"train.eval_episodes={eval_episodes}",
        f"hydra.run.dir={run_dir}",
    ]
    if wandb_group is None:
        command.append("callbacks=none")
    else:
        # The default wandb.name is derived from the agent/model choices, which
        # are identical across arms -- without overriding it all three runs
        # would land in W&B under the same name.
        command += [
            "callbacks=wandb",
            f"wandb.group={wandb_group}",
            f"wandb.name={arm}-s{seed}",
            f"wandb.job_type={arm}",
        ]
    command += extra
    if spec["curriculum"] is not None:
        # Only the enable flag is forced here; every other curriculum setting
        # is owned by the env config, so the two pooled arms stay identical
        # apart from this one key.
        insert_at = len(command) - len(extra)
        command[insert_at:insert_at] = [
            f"env.curriculum.enabled={str(spec['curriculum']).lower()}",
        ]
    completed = subprocess.run(
        command, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        sys.stderr.write(
            f"\n[{arm} seed={seed}] run failed:\n{completed.stderr[-3000:]}\n"
        )
        return None
    return _final_win_rates(completed.stdout + completed.stderr)


def _final_win_rates(output: str) -> dict[str, float] | None:
    """
    Extract the last evaluation win rate per reference opponent.

    The evaluator tags each round with the opponent it scored against
    (``Evaluation [random] over ...``), so runs scoring against several
    references keep them apart rather than collapsing to whichever logged last.

    :param output: Combined stdout and stderr of the run.
    :return: Mapping from opponent label to its final win rate, or None if the
        run reported no evaluation at all.
    """
    rates: dict[str, float] = {}
    for line in output.splitlines():
        if "Evaluation [" not in line or "win_rate=" not in line:
            continue
        label = line.split("Evaluation [", 1)[1].split("]", 1)[0]
        rates[label] = float(line.split("win_rate=")[1].split()[0])
    return rates or None


def main() -> None:
    """
    Run the selected arms across seeds and report results.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms",
        type=str,
        default="fixed,uniform,curriculum",
        help="comma-separated arms to run (fixed, uniform, curriculum)",
    )
    parser.add_argument("--seeds", type=int, default=1, help="paired seeds per arm")
    parser.add_argument("--frames", type=int, default=2_000_000, help="frames per run")
    parser.add_argument("--workers", type=int, default=16, help="environment workers")
    parser.add_argument(
        "--eval-interval", type=int, default=100_000, help="frames between eval rounds"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="episodes per reference opponent per round",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs" / "curriculum_ab",
        help="directory the runs write into",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="W&B group for this experiment's arms (default: curriculum_ab_<timestamp>)",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="disable W&B logging entirely"
    )
    parser.add_argument(
        "override", nargs="*", help="extra Hydra overrides applied to all arms"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    arm_names = [a.strip() for a in args.arms.split(",")]
    for name in arm_names:
        if name not in ARM_SPECS:
            parser.error(f"unknown arm {name!r}; expected fixed, uniform, curriculum")

    wandb_group = None
    if not args.no_wandb:
        wandb_group = args.wandb_group or (
            f"curriculum_ab_{datetime.now(UTC):%Y%m%d_%H%M%S}"
        )
        print(f"W&B group: {wandb_group}", flush=True)

    arms = {name: ArmResult(name) for name in arm_names}
    for seed in range(args.seeds):
        for name in arm_names:
            print(f"running {name} seed={seed} ...", flush=True)
            rates = run_arm(
                seed,
                name,
                args.frames,
                args.workers,
                args.eval_interval,
                args.eval_episodes,
                args.out,
                wandb_group,
                args.override,
            )
            if rates is None:
                arms[name].failures.append(seed)
            else:
                arms[name].win_rates.append(rates)
            print(f"  {name} seed={seed}: {rates}", flush=True)

    opponents = sorted({name for arm in arms.values() for name in arm.opponents})
    summary: dict = {
        "frames": args.frames,
        "seeds": args.seeds,
        "opponents": opponents,
        "arms": {
            name: {
                "win_rates": arm.win_rates,
                "mean": {opponent: arm.mean(opponent) for opponent in opponents},
            }
            for name, arm in arms.items()
        },
        "failures": {name: arm.failures for name, arm in arms.items()},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    _print_table(arms, opponents)


def _print_table(arms: dict[str, ArmResult], opponents: list[str]) -> None:
    """
    Print the arm-by-opponent mean win rates as a table.

    :param arms: Results per arm.
    :param opponents: Reference opponents to show as columns.
    """
    if not opponents:
        return
    width = max(len(name) for name in arms) + 2
    header = "arm".ljust(width) + "".join(f"{name:>20}" for name in opponents)
    print("\n" + header)
    print("-" * len(header))
    for name, arm in arms.items():
        row = name.ljust(width)
        row += "".join(f"{arm.mean(opponent):>20.3f}" for opponent in opponents)
        print(row)


if __name__ == "__main__":
    main()
