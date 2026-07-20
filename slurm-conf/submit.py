"""Submit a Hydra training configuration through Slurm."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

PROFILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROFILE_DIR.parent
CONF_DIR = PROJECT_ROOT / "conf"
JOB_SCRIPT = PROFILE_DIR / "run_job.sh"

SBATCH_OPTIONS = {
    "job_name": "--job-name",
    "time": "--time",
    "nodes": "--nodes",
    "ntasks": "--ntasks",
    "cpus_per_task": "--cpus-per-task",
    "mem": "--mem",
    "partition": "--partition",
    "account": "--account",
    "qos": "--qos",
    "constraint": "--constraint",
    "output": "--output",
    "error": "--error",
}
REQUIRED_SLURM_OPTIONS = {
    "job_name",
    "time",
    "nodes",
    "ntasks",
    "cpus_per_task",
    "mem",
    "partition",
    "gpu_type",
    "gpus_per_node",
}
SPECIAL_SLURM_OPTIONS = {"gpu_type", "gpus_per_node"}


def _profile_path(profile: str) -> Path:
    """Resolve a profile name inside the first-level slurm-conf directory."""
    path = Path(profile)
    if path.suffix == "":
        path = path.with_suffix(".yaml")
    if not path.is_absolute() and path.parts[:1] == (PROFILE_DIR.name,):
        path = PROJECT_ROOT / path
    elif not path.is_absolute():
        path = PROFILE_DIR / path
    path = path.resolve()
    if path.parent != PROFILE_DIR:
        raise ValueError(f"Slurm profiles must be direct children of {PROFILE_DIR}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_profile(path: Path) -> dict[str, Any]:
    """Load and resolve a Slurm profile."""
    config = OmegaConf.load(path)
    profile = OmegaConf.to_container(config, resolve=True)
    if not isinstance(profile, dict):
        raise TypeError(f"{path} must contain a mapping")
    return cast(dict[str, Any], profile)


def _absolute_log_path(value: str) -> str:
    """Make a Slurm log path absolute and create its parent directory."""
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _config_name(value: str) -> str:
    """Validate a complete Hydra config stored under conf/."""
    requested = Path(value)
    if requested.suffix == "":
        requested = requested.with_suffix(".yaml")
    if not requested.is_absolute() and requested.parts[:1] == (CONF_DIR.name,):
        path = PROJECT_ROOT / requested
    elif not requested.is_absolute():
        path = CONF_DIR / requested
    else:
        path = requested
    path = path.resolve()
    if not path.is_relative_to(CONF_DIR) or not path.is_file():
        raise FileNotFoundError(f"Hydra config not found: {path}")
    return path.relative_to(CONF_DIR).with_suffix("").as_posix()


def _uv_environment(gpu_type: str) -> str:
    """Return the uv environment required by a GPU type."""
    name = ".venv-rtx" if gpu_type == "rtx_pro_6000" else ".venv"
    python = PROJECT_ROOT / name / "bin" / "python"
    if not python.is_file():
        setup = (
            "./slurm-conf/setup_uv.sh --use-rtx"
            if name == ".venv-rtx"
            else "./slurm-conf/setup_uv.sh"
        )
        raise FileNotFoundError(f"required environment {name} is missing; run: {setup}")
    return name


def _build_command(
    profile_path: Path,
    profile: dict[str, Any],
    config_name: str,
    extra_overrides: list[str],
) -> list[str]:
    """Translate a profile into one explicit sbatch invocation."""
    slurm = profile.get("slurm")
    if not isinstance(slurm, dict):
        raise TypeError("Slurm config requires a top-level 'slurm' mapping")
    config_name = _config_name(config_name)

    unknown = set(slurm) - set(SBATCH_OPTIONS) - SPECIAL_SLURM_OPTIONS
    missing = REQUIRED_SLURM_OPTIONS - set(slurm)
    if unknown:
        raise ValueError(f"Unsupported slurm options: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"Missing slurm options: {', '.join(sorted(missing))}")

    gpu_type = slurm["gpu_type"]
    gpu_count = slurm["gpus_per_node"]
    if not isinstance(gpu_type, str) or not gpu_type:
        raise ValueError("slurm.gpu_type must be a non-empty string")
    if not isinstance(gpu_count, int) or gpu_count < 1:
        raise ValueError("slurm.gpus_per_node must be a positive integer")
    uv_environment = _uv_environment(gpu_type)

    command = ["sbatch", f"--chdir={PROJECT_ROOT}"]
    for key, option in SBATCH_OPTIONS.items():
        value = slurm.get(key)
        if value is None:
            continue
        if key in {"output", "error"}:
            value = _absolute_log_path(str(value))
        command.append(f"{option}={value}")
    command.append(f"--gpus-per-node={gpu_type}:{gpu_count}")

    command.extend(
        [
            str(JOB_SCRIPT),
            str(profile_path),
            config_name,
            str(gpu_count),
            uv_environment,
            "train.device=cuda",
            *extra_overrides,
        ]
    )
    return command


def main() -> None:
    """Parse a profile, show the exact command, and submit it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Complete Hydra YAML under conf/ (with or without .yaml)",
    )
    parser.add_argument(
        "--slurm-config",
        required=True,
        help="Scheduler YAML under slurm-conf/ (with or without .yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sbatch command without submitting it",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides appended after the profile overrides",
    )
    args = parser.parse_intermixed_args()

    try:
        profile_path = _profile_path(args.slurm_config)
        command = _build_command(
            profile_path,
            _load_profile(profile_path),
            args.config,
            args.overrides,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    print(shlex.join(command))

    if args.dry_run:
        return
    if shutil.which("sbatch") is None:
        raise RuntimeError("sbatch is not available; run this command on a Slurm login node")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
