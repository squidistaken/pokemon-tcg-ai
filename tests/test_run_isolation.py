"""
Guards the one-run-one-snapshot-directory invariant.

Two runs sharing a snapshot directory write identically-named snapshots over
each other and load each other's policies as self-play opponents, which is
silent: the worker-side league treats a failed load as a mid-write retry, and
same-architecture arms cross-load without error at all. Both halves of the
defence are pinned here — the unique run directory, and the exclusive create
that catches whatever slips past it.
"""

from pathlib import Path
from unittest import mock

import pytest
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.hydra_resolvers import run_uid
from src.train import _resolve_checkpoint_dir

CONF_ROOT = Path(__file__).parents[1] / "conf"


def _cfg(checkpoint_dir: str) -> DictConfig:
    """
    Build the minimal config ``_resolve_checkpoint_dir`` reads.

    :param checkpoint_dir: Value for ``train.checkpoint_dir``.
    :return: A config with only that key populated.
    """
    return OmegaConf.create({"train": {"checkpoint_dir": checkpoint_dir}})


@pytest.fixture
def hydra_output_dir(tmp_path: Path):
    """
    Point ``HydraConfig.get().runtime.output_dir`` at a temporary run directory.

    :param tmp_path: Pytest temporary directory.
    :return: The run directory relative paths resolve against.
    """
    run_dir = tmp_path / "2026-08-02" / "15-56-47-30429451"
    run_dir.mkdir(parents=True)
    runtime = OmegaConf.create({"runtime": {"output_dir": str(run_dir)}})
    with mock.patch.object(HydraConfig, "get", return_value=runtime):
        yield run_dir


class TestRunUid:
    """The identifier that separates same-second runs."""

    @staticmethod
    def test_prefers_slurm_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
        """The job id is what distinguishes sweep arms released together."""
        monkeypatch.setenv("SLURM_JOB_ID", "30429451")
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
        assert run_uid() == "30429451"

    @staticmethod
    def test_array_tasks_stay_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
        """Array elements share a job id, so the task id has to be in the path."""
        monkeypatch.setenv("SLURM_JOB_ID", "30429451")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
        assert run_uid() == "30429451_3"

    @staticmethod
    def test_falls_back_to_pid(monkeypatch: pytest.MonkeyPatch) -> None:
        """Unscheduled runs still need to be told apart from each other."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
        import os

        assert run_uid() == f"p{os.getpid()}"


class TestRunDirTemplate:
    """The configured run directory has to carry the uid, not just a clock."""

    @staticmethod
    @pytest.mark.parametrize("config_name", ["config", "experiment/debug"])
    def test_run_dir_is_uid_stamped(config_name: str) -> None:
        """A timestamp alone collides when Slurm starts arms in one second."""
        raw = yaml.safe_load((CONF_ROOT / f"{config_name}.yaml").read_text())
        run_dir = raw["hydra"]["run"]["dir"]
        assert "${run_uid:}" in run_dir, (
            f"{config_name}.yaml run dir {run_dir!r} is not uid-stamped; "
            f"concurrently started runs would share one checkpoints/"
        )


class TestResolveCheckpointDir:
    """The exclusive-create tripwire behind the unique run directory."""

    @staticmethod
    def test_creates_directory_under_run_dir(hydra_output_dir: Path) -> None:
        """A relative path is anchored to the run dir and created up front."""
        resolved = _resolve_checkpoint_dir(_cfg("checkpoints"))
        assert resolved == hydra_output_dir / "checkpoints"
        assert resolved.is_dir()

    @staticmethod
    @pytest.mark.usefixtures("hydra_output_dir")
    def test_rejects_a_directory_another_run_owns() -> None:
        """The second run into one directory fails instead of overwriting."""
        _resolve_checkpoint_dir(_cfg("checkpoints"))
        with pytest.raises(RuntimeError, match="already exists"):
            _resolve_checkpoint_dir(_cfg("checkpoints"))

    @staticmethod
    def test_absolute_path_is_left_alone(tmp_path: Path) -> None:
        """An absolute path is deliberate operator intent, not a collision."""
        existing = tmp_path / "league"
        existing.mkdir()
        assert _resolve_checkpoint_dir(_cfg(str(existing))) == existing
