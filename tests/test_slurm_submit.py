# ruff: noqa: SLF001
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMIT_PATH = PROJECT_ROOT / "slurm-conf" / "submit.py"
SPEC = importlib.util.spec_from_file_location("slurm_submit", SUBMIT_PATH)
assert SPEC is not None and SPEC.loader is not None
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)


@pytest.mark.parametrize(
    ("profile_name", "gpu_type", "environment"),
    [
        ("train_gpu", "a100", ".venv"),
        ("train_gpu_rtx", "rtx_pro_6000", ".venv-rtx"),
    ],
)
def test_gpu_profiles(
    monkeypatch: pytest.MonkeyPatch,
    profile_name: str,
    gpu_type: str,
    environment: str,
) -> None:
    selected: list[str] = []
    monkeypatch.setattr(
        submit,
        "_uv_environment",
        lambda value: selected.append(value) or environment,
    )
    profile_path = submit._profile_path(profile_name)

    command = submit._build_command(
        profile_path, submit._load_profile(profile_path), "config", []
    )

    assert selected == [gpu_type]
    assert f"--gpus-per-node={gpu_type}:1" in command
    assert environment in command


@pytest.mark.parametrize("key", ["nodes", "ntasks", "gpus_per_node"])
def test_rejects_distributed_resources(
    monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    profile_path = submit._profile_path("train_gpu")
    profile: dict[str, Any] = submit._load_profile(profile_path)
    profile["slurm"][key] = 2
    monkeypatch.setattr(submit, "_uv_environment", lambda _gpu_type: ".venv")

    with pytest.raises(ValueError, match=rf"slurm\.{key} must be 1"):
        submit._build_command(profile_path, profile, "config", [])


@pytest.mark.parametrize(
    "arguments",
    [
        ["--config", "config", "--slurm-config", "train_gpu", "--dry-run"],
        ["--slurm-config", "train_gpu", "--config", "config", "--dry-run"],
    ],
)
def test_config_argument_order(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
) -> None:
    monkeypatch.setattr(sys, "argv", [str(SUBMIT_PATH), *arguments])
    monkeypatch.setattr(submit, "_build_command", lambda *_args: ["sbatch", "worker"])

    submit.main()

    assert capsys.readouterr().out.strip() == "sbatch worker"
