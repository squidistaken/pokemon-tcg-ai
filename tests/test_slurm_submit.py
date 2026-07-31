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


def test_hydra_config_must_be_top_level() -> None:
    for value in ("ppo", "ppo.yaml", "conf/ppo.yaml"):
        assert submit._config_name(value) == "ppo"

    with pytest.raises(ValueError, match="must be direct children"):
        submit._config_name("agent/ppo")


@pytest.mark.parametrize(
    ("profile_name", "gpu_type", "environment", "config_name"),
    [
        ("train_gpu", "a100", ".venv", "baseline"),
        ("train_gpu_rtx", "rtx_pro_6000", ".venv-rtx", "ppo"),
    ],
)
def test_gpu_profiles(
    monkeypatch: pytest.MonkeyPatch,
    profile_name: str,
    gpu_type: str,
    environment: str,
    config_name: str,
) -> None:
    selected: list[str] = []
    monkeypatch.setattr(
        submit,
        "_uv_environment",
        lambda value, _module: selected.append(value) or environment,
    )
    profile_path = submit._profile_path(profile_name)

    command = submit._build_command(
        profile_path, submit._load_profile(profile_path), config_name, [], "src.train"
    )

    assert selected == [gpu_type]
    assert f"--gpus-per-node={gpu_type}:1" in command
    assert environment in command
    assert config_name in command
    assert "++agent.device=cuda" in command
    assert "src.train" in command


def test_cpu_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    selected: list[str] = []
    monkeypatch.setattr(
        submit,
        "_uv_environment",
        lambda value, _module: selected.append(value) or ".venv",
    )
    profile_path = submit._profile_path("train_cpu")

    command = submit._build_command(
        profile_path, submit._load_profile(profile_path), "baseline", [], "src.train"
    )

    assert selected == ["none"]
    assert not any(arg.startswith("--gpus-per-node") for arg in command)
    assert "++agent.device=cpu" in command
    assert "++agent.device=cuda" not in command
    assert ".venv" in command


def test_module_flows_to_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(submit, "_uv_environment", lambda _gpu_type, _module: ".venv")
    profile_path = submit._profile_path("train_cpu")

    command = submit._build_command(
        profile_path, submit._load_profile(profile_path), "eval_deck_field", [], "src.eval_deck_field"
    )

    assert "src.eval_deck_field" in command
    assert "src.train" not in command


def test_cpu_profile_requires_zero_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    profile_path = submit._profile_path("train_cpu")
    profile: dict[str, Any] = submit._load_profile(profile_path)
    profile["slurm"]["gpus_per_node"] = 1
    monkeypatch.setattr(submit, "_uv_environment", lambda _gpu_type, _module: ".venv")

    with pytest.raises(ValueError, match="gpus_per_node must be 0"):
        submit._build_command(profile_path, profile, "config", [], "src.train")


@pytest.mark.parametrize(
    "override",
    ["agent.device=cpu", "+agent.device=cuda", "++agent.device=cpu", "~agent.device"],
)
def test_rejects_user_device_override(
    monkeypatch: pytest.MonkeyPatch, override: str
) -> None:
    profile_path = submit._profile_path("train_gpu")
    monkeypatch.setattr(submit, "_uv_environment", lambda _gpu_type, _module: ".venv")

    with pytest.raises(ValueError, match="agent.device is set by the Slurm profile"):
        submit._build_command(
            profile_path, submit._load_profile(profile_path), "config", [override], "src.train"
        )


@pytest.mark.parametrize("key", ["nodes", "ntasks", "gpus_per_node"])
def test_rejects_distributed_resources(
    monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    profile_path = submit._profile_path("train_gpu")
    profile: dict[str, Any] = submit._load_profile(profile_path)
    profile["slurm"][key] = 2
    monkeypatch.setattr(submit, "_uv_environment", lambda _gpu_type, _module: ".venv")

    with pytest.raises(ValueError, match=rf"slurm\.{key} must be 1"):
        submit._build_command(profile_path, profile, "config", [], "src.train")


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


def test_module_defaults_to_train(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(
        sys, "argv", [str(SUBMIT_PATH), "--config", "config", "--slurm-config", "train_gpu", "--dry-run"]
    )
    monkeypatch.setattr(
        submit,
        "_build_command",
        lambda *args: (seen.append(args[-1]), ["sbatch", "worker"])[1],
    )

    submit.main()

    assert seen == ["src.train"]
    assert capsys.readouterr().out.strip() == "sbatch worker"


def test_module_flag_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SUBMIT_PATH),
            "--config", "eval_deck_field",
            "--slurm-config", "train_cpu",
            "--module", "src.eval_deck_field",
            "--dry-run",
        ],
    )
    monkeypatch.setattr(
        submit,
        "_build_command",
        lambda *args: (seen.append(args[-1]), ["sbatch", "worker"])[1],
    )

    submit.main()

    assert seen == ["src.eval_deck_field"]
