"""Tests for the repository-local completed-checkpoint registry."""

from __future__ import annotations

import csv
import multiprocessing
from pathlib import Path
from typing import Any

import pytest

from src.checkpoint_registry import (
    CHECKPOINT_REGISTRY_FIELDS,
    CheckpointRegistryError,
    append_checkpoint_record,
    find_checkpoint_record,
    latest_checkpoint_record,
    read_checkpoint_records,
    resolve_record_path,
    sha256_file,
    validate_checkpoint_record,
)


def _append_in_process(
    start_barrier: Any,
    registry_value: str,
    checkpoint_value: str,
    repo_root_value: str,
    frames: int,
) -> None:
    """Append from a child process to exercise the POSIX file lock."""
    registry = Path(registry_value)
    checkpoint = Path(checkpoint_value)
    start_barrier.wait(timeout=20)
    append_checkpoint_record(
        registry,
        checkpoint,
        digest=sha256_file(checkpoint),
        frames=frames,
        repo_root=Path(repo_root_value),
    )


def _write_checkpoint(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def test_append_creates_one_header_and_latest_is_bottom_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Append order, not mtime, defines the newest completed checkpoint."""
    registry = tmp_path / "logs" / "checkpoint_keys.csv"
    first = _write_checkpoint(tmp_path / "checkpoints" / "first.pt", b"first")
    second = _write_checkpoint(tmp_path / "checkpoints" / "second.pt", b"second")
    config = _write_checkpoint(tmp_path / "config.yaml", b"model: {}\n")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    first_record = append_checkpoint_record(
        registry,
        first,
        digest=sha256_file(first),
        frames=100,
        repo_root=tmp_path,
        config_path=config,
    )
    second_record = append_checkpoint_record(
        registry,
        second,
        digest=sha256_file(second),
        frames=200,
        repo_root=tmp_path,
    )

    with registry.open(encoding="utf-8", newline="") as registry_file:
        rows = list(csv.reader(registry_file))
    assert rows[0] == list(CHECKPOINT_REGISTRY_FIELDS)
    assert len(rows) == 3
    assert first_record.checkpoint_path == "checkpoints/first.pt"
    assert first_record.config_path == "config.yaml"
    assert first_record.slurm_job_id == "12345"
    assert second_record.slurm_job_id == "12345"
    assert latest_checkpoint_record(registry) == second_record
    assert (
        resolve_record_path(first_record.checkpoint_path, tmp_path) == first.resolve()
    )
    assert validate_checkpoint_record(first_record, tmp_path) == first.resolve()


def test_find_uses_digest_prefix_and_rejects_ambiguous_keys(tmp_path: Path) -> None:
    """Hash lookup returns the last duplicate and reports ambiguous prefixes."""
    registry = tmp_path / "checkpoint_keys.csv"
    first = _write_checkpoint(tmp_path / "first.pt", b"first")
    second = _write_checkpoint(tmp_path / "second.pt", b"second")
    first_digest = sha256_file(first)
    second_digest = sha256_file(second)
    first_record = append_checkpoint_record(
        registry,
        first,
        digest=first_digest,
        frames=1,
        repo_root=tmp_path,
    )
    append_checkpoint_record(
        registry,
        first,
        digest=first_digest,
        frames=2,
        repo_root=tmp_path,
    )
    append_checkpoint_record(
        registry,
        second,
        digest=second_digest,
        frames=3,
        repo_root=tmp_path,
    )

    found = find_checkpoint_record(registry, first_record.key)
    assert found.frames == 2
    common_prefix = ""
    for first_char, second_char in zip(first_digest, second_digest, strict=True):
        if first_char != second_char:
            break
        common_prefix += first_char
    if common_prefix:
        with pytest.raises(CheckpointRegistryError, match="ambiguous"):
            find_checkpoint_record(registry, common_prefix)


@pytest.mark.parametrize(
    "contents",
    [
        "not,the,registry\n",
        ",,,,,,,,\n",
    ],
)
def test_malformed_registry_is_actionable(tmp_path: Path, contents: str) -> None:
    """Malformed headers and rows never silently change latest selection."""
    registry = tmp_path / "checkpoint_keys.csv"
    registry.write_text(contents, encoding="utf-8")

    with pytest.raises(CheckpointRegistryError, match="Malformed|no records"):
        read_checkpoint_records(registry)


@pytest.mark.parametrize("row_suffix", [",unexpected", ""])
def test_registry_rejects_rows_with_extra_or_missing_columns(
    tmp_path: Path,
    row_suffix: str,
) -> None:
    """A shifted/truncated CSV row cannot silently become the latest checkpoint."""
    registry = tmp_path / "checkpoint_keys.csv"
    values = [
        "2026-07-31T12:00:00+00:00",
        "a" * 12,
        "a" * 64,
        "1",
        "checkpoint.pt",
        "",
        "training",
        "",
        "host",
    ]
    if not row_suffix:
        values.pop()
    registry.write_text(
        ",".join(CHECKPOINT_REGISTRY_FIELDS)
        + "\n"
        + ",".join(values)
        + row_suffix
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CheckpointRegistryError, match="Malformed.*row 2"):
        read_checkpoint_records(registry)


def test_validation_detects_missing_and_changed_checkpoint(tmp_path: Path) -> None:
    """A stale path or content hash blocks packaging."""
    checkpoint = _write_checkpoint(tmp_path / "checkpoint.pt", b"original")
    registry = tmp_path / "checkpoint_keys.csv"
    record = append_checkpoint_record(
        registry,
        checkpoint,
        digest=sha256_file(checkpoint),
        frames=1,
        repo_root=tmp_path,
    )

    checkpoint.write_bytes(b"changed")
    with pytest.raises(CheckpointRegistryError, match="hash mismatch"):
        validate_checkpoint_record(record, tmp_path)
    checkpoint.unlink()
    with pytest.raises(CheckpointRegistryError, match="no longer exists"):
        validate_checkpoint_record(record, tmp_path)


def test_concurrent_processes_write_complete_rows_and_one_header(
    tmp_path: Path,
) -> None:
    """Concurrent Slurm-style processes cannot interleave headers or CSV rows."""
    registry = tmp_path / "logs" / "checkpoint_keys.csv"
    checkpoints = [
        _write_checkpoint(
            tmp_path / "checkpoints" / f"checkpoint-{index}.pt", bytes([index])
        )
        for index in range(8)
    ]
    context = multiprocessing.get_context("spawn")
    start_barrier = context.Barrier(len(checkpoints))
    processes = [
        context.Process(
            target=_append_in_process,
            args=(
                start_barrier,
                str(registry),
                str(checkpoint),
                str(tmp_path),
                index,
            ),
        )
        for index, checkpoint in enumerate(checkpoints)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    records = read_checkpoint_records(registry)
    assert len(records) == len(processes)
    assert {record.frames for record in records} == set(range(8))
    assert registry.read_text(encoding="utf-8").count("completed_at_utc") == 1
