"""Append-only registry for completed training checkpoints."""

from __future__ import annotations

import csv
import hashlib
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - Slurm and CI are POSIX.
    fcntl = None

CHECKPOINT_KEY_LENGTH = 12
CHECKPOINT_REGISTRY_FIELDS = (
    "completed_at_utc",
    "key",
    "sha256",
    "frames",
    "checkpoint_path",
    "config_path",
    "source",
    "slurm_job_id",
    "hostname",
)
_HASH_CHUNK_SIZE = 1024 * 1024
_CHECKPOINT_SOURCES = {"training", "manual_import"}


class CheckpointRegistryError(RuntimeError):
    """Checkpoint registry is missing, malformed, stale, or ambiguous."""


class CheckpointRegistryLookupError(CheckpointRegistryError):
    """A well-formed registry contains no requested checkpoint key."""


@dataclass(frozen=True)
class CheckpointRecord:
    """One completed checkpoint, serialized as one CSV row."""

    completed_at_utc: str
    key: str
    sha256: str
    frames: int
    checkpoint_path: str
    config_path: str
    source: str
    slurm_job_id: str
    hostname: str


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path, repo_root: Path) -> str:
    """Store repository files relatively and external/scratch files absolutely."""
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(repo_root.expanduser().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def resolve_record_path(value: str, repo_root: Path) -> Path:
    """Resolve a path stored by :func:`portable_path`."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


@contextmanager
def registry_lock(path: Path, *, exclusive: bool) -> Iterator[None]:
    """Lock a sidecar file while reading or appending an append-only registry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / f".{path.name}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(lock_file.fileno(), operation)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def append_checkpoint_record(
    registry_path: Path,
    checkpoint_path: Path,
    *,
    digest: str,
    frames: int,
    repo_root: Path,
    config_path: Path | None = None,
    source: str = "training",
) -> CheckpointRecord:
    """Append one completed checkpoint to the bottom of the registry CSV."""
    normalized_digest = digest.lower()
    if len(normalized_digest) != 64 or any(
        character not in "0123456789abcdef" for character in normalized_digest
    ):
        raise ValueError("Checkpoint digest must be a 64-character SHA-256 value.")
    if frames < 0:
        raise ValueError("Checkpoint frame count cannot be negative.")
    if source not in _CHECKPOINT_SOURCES:
        raise ValueError(
            f"Checkpoint source must be one of {', '.join(sorted(_CHECKPOINT_SOURCES))}."
        )
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    actual_digest = sha256_file(checkpoint_path)
    if actual_digest != normalized_digest:
        raise ValueError(
            f"Checkpoint digest mismatch: expected {normalized_digest}, got {actual_digest}."
        )
    if config_path is not None and not config_path.expanduser().resolve().is_file():
        raise FileNotFoundError(config_path)

    record = CheckpointRecord(
        completed_at_utc=datetime.now(UTC).isoformat(),
        key=normalized_digest[:CHECKPOINT_KEY_LENGTH],
        sha256=normalized_digest,
        frames=frames,
        checkpoint_path=portable_path(checkpoint_path, repo_root),
        config_path=(
            portable_path(config_path, repo_root) if config_path is not None else ""
        ),
        source=source,
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
        hostname=socket.gethostname(),
    )

    registry_path = registry_path.expanduser().resolve()
    with registry_lock(registry_path, exclusive=True):
        needs_header = not registry_path.exists() or registry_path.stat().st_size == 0
        with registry_path.open("a", encoding="utf-8", newline="") as registry_file:
            writer = csv.DictWriter(
                registry_file,
                fieldnames=CHECKPOINT_REGISTRY_FIELDS,
                lineterminator="\n",
            )
            if needs_header:
                writer.writeheader()
            writer.writerow(asdict(record))
            registry_file.flush()
            os.fsync(registry_file.fileno())
    return record


def read_checkpoint_records(registry_path: Path) -> list[CheckpointRecord]:
    """Read and validate all checkpoint rows in append order."""
    registry_path = registry_path.expanduser().resolve()
    if not registry_path.is_file():
        raise CheckpointRegistryError(
            f"Checkpoint registry does not exist: {registry_path}"
        )
    with (
        registry_lock(registry_path, exclusive=False),
        registry_path.open(encoding="utf-8", newline="") as registry_file,
    ):
        reader = csv.DictReader(registry_file)
        if reader.fieldnames != list(CHECKPOINT_REGISTRY_FIELDS):
            raise CheckpointRegistryError(
                f"Malformed checkpoint registry header in {registry_path}."
            )
        records: list[CheckpointRecord] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                if None in row or any(
                    not isinstance(row.get(field), str)
                    for field in CHECKPOINT_REGISTRY_FIELDS
                ):
                    raise ValueError
                completed_at_utc = row["completed_at_utc"]
                digest = row["sha256"].lower()
                key = row["key"].lower()
                frames = int(row["frames"])
                checkpoint_path = row["checkpoint_path"]
                config_path = row["config_path"]
                source = row["source"]
                slurm_job_id = row["slurm_job_id"]
                hostname = row["hostname"]
                completed = datetime.fromisoformat(completed_at_utc)
                if (
                    completed.utcoffset() != timedelta(0)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                    or key != digest[:CHECKPOINT_KEY_LENGTH]
                    or frames < 0
                    or not checkpoint_path
                    or source not in _CHECKPOINT_SOURCES
                    or not hostname
                ):
                    raise ValueError
                records.append(
                    CheckpointRecord(
                        completed_at_utc=completed_at_utc,
                        key=key,
                        sha256=digest,
                        frames=frames,
                        checkpoint_path=checkpoint_path,
                        config_path=config_path,
                        source=source,
                        slurm_job_id=slurm_job_id,
                        hostname=hostname,
                    )
                )
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                raise CheckpointRegistryError(
                    f"Malformed checkpoint registry row {row_number} in {registry_path}."
                ) from error
    if not records:
        raise CheckpointRegistryError(
            f"Checkpoint registry has no records: {registry_path}"
        )
    return records


def latest_checkpoint_record(registry_path: Path) -> CheckpointRecord:
    """Return the bottom (newest completed) checkpoint row."""
    return read_checkpoint_records(registry_path)[-1]


def find_checkpoint_record(registry_path: Path, digest_prefix: str) -> CheckpointRecord:
    """Resolve a unique digest prefix, preferring the last duplicate record."""
    prefix = digest_prefix.lower()
    matching = [
        record
        for record in read_checkpoint_records(registry_path)
        if record.sha256.startswith(prefix)
    ]
    if not matching:
        raise CheckpointRegistryLookupError(
            f"No checkpoint registry record matches SHA-256 prefix {digest_prefix!r}."
        )
    distinct_digests = {record.sha256 for record in matching}
    if len(distinct_digests) > 1:
        raise CheckpointRegistryError(
            f"Checkpoint key {digest_prefix!r} is ambiguous; provide more characters."
        )
    return matching[-1]


def validate_checkpoint_record(record: CheckpointRecord, repo_root: Path) -> Path:
    """Resolve a registry checkpoint and verify its recorded content hash."""
    checkpoint = resolve_record_path(record.checkpoint_path, repo_root)
    if not checkpoint.is_file():
        raise CheckpointRegistryError(
            f"Registered checkpoint no longer exists: {checkpoint}"
        )
    actual_digest = sha256_file(checkpoint)
    if actual_digest != record.sha256:
        raise CheckpointRegistryError(
            f"Registered checkpoint hash mismatch for {checkpoint}: "
            f"expected {record.sha256}, got {actual_digest}."
        )
    return checkpoint


# Kaggle live-rating history.
_RATING_HISTORY_FIELDS = (
    "fetched_at_utc",
    "kaggle_ref",
    "label",
    "status",
    "public_score",
    "leaderboard_rank",
)


def _migrate_rating_history_header(path: Path) -> None:
    """Backfill older history."""
    with path.open(encoding="utf-8", newline="") as history_file:
        reader = csv.DictReader(history_file)
        rows = list(reader)
    with path.open("w", encoding="utf-8", newline="") as history_file:
        writer = csv.writer(history_file)
        writer.writerow(_RATING_HISTORY_FIELDS)
        for row in rows:
            writer.writerow([row.get(field, "") for field in _RATING_HISTORY_FIELDS])


def append_rating_history(
    path: Path,
    *,
    kaggle_ref: int,
    label: str,
    status: str,
    public_score: float | None,
    leaderboard_rank: int | None = None,
) -> None:
    """Append one submission's live-rating snapshot to the history CSV."""
    with registry_lock(path, exclusive=True):
        needs_header = not path.exists() or path.stat().st_size == 0
        if not needs_header:
            with path.open(encoding="utf-8", newline="") as history_file:
                header = next(csv.reader(history_file), [])
            if header and "leaderboard_rank" not in header:
                _migrate_rating_history_header(path)
        with path.open("a", encoding="utf-8", newline="") as history_file:
            writer = csv.writer(history_file)
            if needs_header:
                writer.writerow(_RATING_HISTORY_FIELDS)
            writer.writerow(
                [
                    datetime.now(UTC).isoformat(),
                    kaggle_ref,
                    label,
                    status,
                    "" if public_score is None else public_score,
                    "" if leaderboard_rank is None else leaderboard_rank,
                ]
            )
