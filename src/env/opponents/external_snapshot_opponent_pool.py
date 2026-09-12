import random
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from cg.api import Observation
from src.env.opponents.opponent_pool import OpponentPool

_SNAPSHOT_PATTERN = re.compile(r"^snapshot_(\d+)\.pt$")


class ExternalSnapshotOpponentPool(OpponentPool):
    """
    Uniform pool backed by pre-existing checkpoint directories.

    Unlike :class:`SnapshotOpponentPool`, this pool has no random warmup
    member and does not assume its source is the learner's output directory.
    It is deliberately separate so the established self-play path keeps its
    exact construction, membership and PFSP behavior.

    With ``refresh=False`` the membership is loaded once and remains fixed.
    With ``refresh=True`` all source directories are rescanned at episode
    boundaries and the globally newest ``pool_size`` snapshots are retained.
    """

    def __init__(
        self,
        checkpoint_dirs: Sequence[str | Path],
        load_snapshot: Callable[[Path], Callable[[Observation], list[int]]],
        pool_size: int,
        refresh: bool = False,
        seed: int | None = None,
    ) -> None:
        """
        :param checkpoint_dirs: Ordered, distinct directories to scan. The
            first is the fixed baseline source; later directories may contain
            learner snapshots.
        :param load_snapshot: Builds one callable policy from a checkpoint.
        :param pool_size: Exact number of newest policies kept in the pool.
        :param refresh: Rescan sources before every episode when true.
        :param seed: Seed for uniform member sampling.
        :raises ValueError: If sources are duplicated, the pool size is not
            positive, or fewer than ``pool_size`` snapshots are initially
            available.
        """
        if pool_size <= 0:
            raise ValueError(
                "ExternalSnapshotOpponentPool pool_size must be positive, "
                f"got {pool_size}."
            )
        directories = tuple(Path(path).resolve() for path in checkpoint_dirs)
        if not directories:
            raise ValueError(
                "ExternalSnapshotOpponentPool needs a checkpoint directory."
            )
        if len(set(directories)) != len(directories):
            raise ValueError(
                "External snapshot checkpoint directories must be distinct."
            )

        self._checkpoint_dirs = directories
        self._baseline_dir = directories[0]
        self._load_snapshot = load_snapshot
        self._pool_size = pool_size
        self._refresh_enabled = bool(refresh)
        self._loaded: dict[Path, Callable[[Observation], list[int]]] = {}
        self._paths: list[Path] = []
        self._active_path: Path | None = None

        initial = self._candidate_paths()
        if len(initial) < pool_size:
            raise ValueError(
                f"External opponent pool found {len(initial)} snapshot(s) across "
                f"{', '.join(map(str, directories))}; expected at least {pool_size}."
            )
        keep = initial[-pool_size:]
        for path in keep:
            self._loaded[path] = self._load_snapshot(path)
        self._paths = keep
        # Keep policy-member draws on a deterministic stream distinct from the
        # deck sampler's stream. TCGEnv deliberately reseeds both components
        # with its worker seed, so seed() applies the same derivation.
        pool_seed = _derive_policy_seed(seed)
        super().__init__([self._loaded[path] for path in keep], seed=pool_seed)
        self._active_path = keep[0]

    @property
    def snapshot_paths(self) -> tuple[Path, ...]:
        """Return current members in sampling order."""
        return tuple(self._paths)

    @property
    def snapshot_count(self) -> int:
        """Return the number of checkpoint policies currently loaded."""
        return len(self._paths)

    @property
    def refresh_enabled(self) -> bool:
        """Whether source directories are rescanned between episodes."""
        return self._refresh_enabled

    @property
    def active_is_anchor(self) -> bool:
        """Whether the active policy came from the fixed baseline directory."""
        return (
            self._active_path is not None
            and self._active_path.parent.resolve() == self._baseline_dir
        )

    def on_reset(self) -> None:
        """Optionally refresh membership, then draw the episode opponent."""
        if self._refresh_enabled:
            self._refresh()
        super().on_reset()
        self._active_path = self._path_of_active()

    def seed(self, seed: int) -> None:
        """Reseed policy sampling independently of the episode's deck draw."""
        derived_seed = _derive_policy_seed(seed)
        assert derived_seed is not None
        super().seed(derived_seed)

    def _refresh(self) -> None:
        """Replace membership with the globally newest available snapshots."""
        candidates = self._candidate_paths()
        if len(candidates) < self._pool_size:
            raise RuntimeError(
                f"External opponent pool shrank to {len(candidates)} snapshot(s); "
                f"expected at least {self._pool_size}."
            )
        keep = candidates[-self._pool_size :]
        for path in keep:
            if path not in self._loaded:
                self._loaded[path] = self._load_snapshot(path)
        keep_set = set(keep)
        for path in list(self._loaded):
            if path not in keep_set:
                del self._loaded[path]
        self._paths = keep
        self.set_opponents([self._loaded[path] for path in keep])

    def _candidate_paths(self) -> list[Path]:
        """Return uniquely resolved snapshot paths ordered by frame count."""
        paths: set[Path] = set()
        for directory in self._checkpoint_dirs:
            if not directory.is_dir():
                continue
            for path in directory.glob("snapshot_*.pt"):
                resolved = path.resolve()
                if _snapshot_frame(resolved) is not None:
                    paths.add(resolved)
        return sorted(paths, key=_snapshot_sort_key)

    def _path_of_active(self) -> Path | None:
        """Resolve the active callable back to its checkpoint path."""
        active = self.active
        for path in self._paths:
            if self._loaded[path] is active:
                return path
        return None


def valid_snapshot_paths(directory: Path) -> list[Path]:
    """Return valid production-named snapshots from one directory."""
    if not directory.is_dir():
        return []
    paths = [
        path.resolve()
        for path in directory.glob("snapshot_*.pt")
        if _snapshot_frame(path) is not None
    ]
    return sorted(set(paths), key=_snapshot_sort_key)


def _snapshot_frame(path: Path) -> int | None:
    """Extract the numeric frame counter from a snapshot filename."""
    match = _SNAPSHOT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match is not None else None


def _snapshot_sort_key(path: Path) -> tuple[int, str]:
    """Order a path after the filename has passed snapshot validation."""
    frames = _snapshot_frame(path)
    assert frames is not None
    return frames, str(path)


def _derive_policy_seed(seed: int | None) -> int | None:
    """Derive a reproducible policy-selection stream from a worker seed."""
    if seed is None:
        return None
    return random.Random(seed).getrandbits(64)
