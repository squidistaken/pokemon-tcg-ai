import pickle
from pathlib import Path

import pytest

from src.env.opponents.external_snapshot_opponent_pool import (
    ExternalSnapshotOpponentPool,
)
from src.env.opponents.opponent_pool import OpponentPool
from src.env.opponents.random_opponent import RandomOpponent


def _fake_loader(path: Path) -> RandomOpponent:
    """Build a cheap, picklable stand-in keyed by the snapshot frame."""
    frame = int(path.stem.removeprefix("snapshot_"))
    return RandomOpponent(seed=frame)


def _write_snapshots(directory: Path, frames: list[int]) -> list[Path]:
    """Create placeholder files with production snapshot names."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for frame in frames:
        path = directory / f"snapshot_{frame:012d}.pt"
        path.write_bytes(b"checkpoint")
        paths.append(path.resolve())
    return paths


def test_frozen_pool_loads_latest_ten_uniformly_and_never_rescans(tmp_path) -> None:
    """Frozen membership has no random anchor and ignores files added later."""
    baseline = tmp_path / "baseline"
    paths = _write_snapshots(baseline, list(range(1, 13)))
    pool = ExternalSnapshotOpponentPool(
        [baseline], _fake_loader, pool_size=10, refresh=False, seed=0
    )

    assert pool.snapshot_paths == tuple(paths[-10:])
    assert pool.weights == (1.0,) * 10
    assert len(pool.opponents) == 10

    _write_snapshots(baseline, [13])
    for _ in range(20):
        pool.on_reset()
    assert pool.snapshot_paths == tuple(paths[-10:])


def test_refresh_starts_from_baseline_then_admits_newest_learner(tmp_path) -> None:
    """Refresh globally ranks both sources and evicts the oldest baseline member."""
    baseline = tmp_path / "baseline"
    learner = tmp_path / "learner"
    initial = _write_snapshots(baseline, list(range(1, 11)))
    learner.mkdir()
    pool = ExternalSnapshotOpponentPool(
        [baseline, learner], _fake_loader, pool_size=10, refresh=True, seed=0
    )
    assert pool.snapshot_paths == tuple(initial)

    added = _write_snapshots(learner, [11, 12])
    pool.on_reset()

    assert pool.snapshot_paths == tuple(initial[2:] + added)
    assert len(pool.opponents) == 10
    assert pool.weights == (1.0,) * 10


def test_external_pool_validates_sources_and_initial_size(tmp_path) -> None:
    """Bad source configurations fail before an environment begins collecting."""
    baseline = tmp_path / "baseline"
    _write_snapshots(baseline, [1, 2])

    with pytest.raises(ValueError, match="expected at least 3"):
        ExternalSnapshotOpponentPool([baseline], _fake_loader, pool_size=3)
    with pytest.raises(ValueError, match="must be distinct"):
        ExternalSnapshotOpponentPool(
            [baseline, baseline / ".." / "baseline"], _fake_loader, pool_size=2
        )
    with pytest.raises(ValueError, match="positive"):
        ExternalSnapshotOpponentPool([baseline], _fake_loader, pool_size=0)


def test_external_pool_is_picklable(tmp_path) -> None:
    """A constructed population can cross a multiprocessing pickle boundary."""
    baseline = tmp_path / "baseline"
    _write_snapshots(baseline, list(range(10)))
    pool = ExternalSnapshotOpponentPool(
        [baseline], _fake_loader, pool_size=10, refresh=False, seed=0
    )

    restored = pickle.loads(pickle.dumps(pool))
    assert restored.snapshot_paths == pool.snapshot_paths
    restored.on_reset()


def test_policy_sampling_uses_a_reproducible_independent_seed_stream(tmp_path) -> None:
    """Policy draws are deterministic without sharing the deck seed stream."""
    baseline = tmp_path / "baseline"
    _write_snapshots(baseline, list(range(10)))
    first = ExternalSnapshotOpponentPool(
        [baseline], _fake_loader, pool_size=10, seed=42
    )
    second = ExternalSnapshotOpponentPool(
        [baseline], _fake_loader, pool_size=10, seed=42
    )
    unseparated = OpponentPool(list(first.opponents), seed=42)

    def draws(pool: OpponentPool) -> list[int]:
        indices = []
        for _ in range(20):
            pool.on_reset()
            indices.append(
                next(
                    i
                    for i, member in enumerate(pool.opponents)
                    if member is pool.active
                )
            )
        return indices

    first_draws = draws(first)
    assert first_draws == draws(second)
    assert first_draws != draws(unseparated)

    first.seed(42)
    assert first_draws == draws(first)
