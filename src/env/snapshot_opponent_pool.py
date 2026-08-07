import logging
from collections.abc import Callable
from pathlib import Path

from cg.api import Observation

from .opponent_pool import OpponentPool

logger = logging.getLogger(__name__)

#: Filename suffix of the learner snapshots written during training.
SNAPSHOT_SUFFIX = ".pt"


class SnapshotOpponentPool(OpponentPool):
    """
    Self-play league that discovers new learner snapshots from disk.

    Under ``ParallelEnv`` the environments live in separate worker processes,
    so a snapshot added to a pool in the trainer process would be invisible to
    them. This pool therefore treats a directory as the shared channel: the
    trainer writes snapshots into it (see
    :class:`~src.training.callbacks.snapshot_callback.SnapshotCallback`) and
    every worker's pool rescans it at each episode boundary, loading files it
    has not seen before.

    The warmup opponents (in practice a
    :class:`~src.env.random_opponent.RandomOpponent`) are permanent members, so
    the league always retains a fixed reference point and never degenerates
    into training purely against near-copies of itself. Beyond them only the
    ``pool_size`` newest snapshots are kept; older ones are evicted.

    A snapshot is picked up at the next episode boundary rather than
    immediately, and a file that fails to load (e.g. caught mid-write) is
    skipped and retried on the next scan.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        load_snapshot: Callable[[Path], Callable[[Observation], list[int]]],
        warmup_opponents: list[Callable[[Observation], list[int]]],
        pool_size: int = 5,
        seed: int | None = None,
    ) -> None:
        """
        :param checkpoint_dir: Directory scanned for ``*.pt`` snapshots. It need
            not exist yet; the first snapshots usually appear well into the run.
        :param load_snapshot: Turns a snapshot path into an opponent callable
            (in practice :func:`~src.policies.greedy_policy_opponent.
            load_greedy_opponent` bound to the run's config and specs).
        :param warmup_opponents: Permanent members faced before any snapshot
            exists, and retained alongside the snapshots afterwards.
        :param pool_size: Number of most-recent snapshots kept, on top of the
            warmup opponents.
        :param seed: Seed for the member sampler.
        :raises ValueError: If ``warmup_opponents`` is empty or ``pool_size``
            is negative.
        """
        if not warmup_opponents:
            raise ValueError("SnapshotOpponentPool needs at least one warmup opponent.")
        if pool_size < 0:
            raise ValueError(f"pool_size must be non-negative, got {pool_size}.")
        super().__init__(list(warmup_opponents), seed=seed)
        self._checkpoint_dir = Path(checkpoint_dir)
        self._load_snapshot = load_snapshot
        self._warmup_opponents = list(warmup_opponents)
        self._warmup_keys = [
            f"warmup:{index}" for index in range(len(warmup_opponents))
        ]
        self._pool_size = pool_size
        self._loaded: dict[Path, Callable[[Observation], list[int]]] = {}

    @property
    def snapshot_count(self) -> int:
        """
        Number of snapshots currently loaded into the league.

        :return: Count of loaded snapshot members, excluding warmup opponents.
        """
        return len(self._loaded)

    @property
    def active_is_anchor(self) -> bool:
        """
        Whether the member playing this episode is a fixed reference opponent.

        :return: True if the drawn member is a warmup opponent.
        """
        active = self.active
        return any(member is active for member in self._warmup_opponents)

    def on_reset(self) -> None:
        """
        Rescan for new snapshots, then draw the member for the next episode.
        """
        self._refresh()
        super().on_reset()

    def _refresh(self) -> None:
        """
        Load snapshots that appeared since the last scan and evict stale ones.

        Ordering is by filename, which the snapshot writer keeps
        lexicographically increasing in frame count, so "newest" is well
        defined without stat-ing every file.
        """
        if not self._checkpoint_dir.is_dir():
            return
        paths = sorted(self._checkpoint_dir.glob(f"*{SNAPSHOT_SUFFIX}"))
        if not paths:
            return
        keep = paths[-self._pool_size :] if self._pool_size else []
        for path in keep:
            if path in self._loaded:
                continue
            try:
                self._loaded[path] = self._load_snapshot(path)
            except Exception:
                # Most likely the trainer is mid-write. Retried next episode.
                logger.debug(
                    "Could not load snapshot %s yet; retrying later.",
                    path,
                    exc_info=True,
                )
        keep_set = set(keep)
        for path in list(self._loaded):
            if path not in keep_set:
                del self._loaded[path]
        live = [path for path in keep if path in self._loaded]
        members = self._warmup_opponents + [self._loaded[path] for path in live]
        keys = self._warmup_keys + [str(path) for path in live]
        if len(members) != len(self._opponents):
            logger.info(
                "Self-play league now has %d member(s): %d warmup + %d snapshot(s).",
                len(members),
                len(self._warmup_opponents),
                len(members) - len(self._warmup_opponents),
            )
        self.set_opponents(members, self._member_weights(keys))

    # Instance method (not static) so subclasses can weight from instance state;
    # the base returns None, leaving the pool uniform exactly as before.
    # noinspection PyMethodMayBeStatic
    def _member_weights(self, keys: list[str]) -> list[float] | None:  # noqa: ARG002, PLR6301
        """
        Sampling weight for each league member, aligned with ``keys``.

        Returns None in this class, which leaves the pool sampling uniformly.
        :class:`~src.env.pfsp_opponent_pool.PFSPOpponentPool` overrides it to
        weight members by the learner's win rate against them.

        :param keys: Stable identifier per member, in member order: the warmup
            opponents first, then one snapshot path per loaded snapshot.
        :return: One weight per member, or None for uniform sampling.
        """
        return None
