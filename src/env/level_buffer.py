from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LevelEntry:
    """
    One scored matchup in the level buffer.

    :param pair_id: Matchup identifier from
        :meth:`~src.env.archetype_index.ArchetypeIndex.pair_id`.
    :param mean_residual: Running mean of the per-episode mean critic residual.
        Kept **signed**; the score takes its magnitude only after averaging.
    :param visits: Completed episodes folded into ``mean_residual``.
    :param last_visit: Value of the buffer's episode counter at the last visit,
        from which staleness is derived.
    :param wins: Learner wins on anchor episodes, a draw counting as half.
    :param games: Anchor episodes played, the denominator for ``wins``.
    """

    pair_id: int
    mean_residual: float = 0.0
    visits: int = 0
    last_visit: int = 0
    wins: float = 0.0
    games: float = 0.0

    @property
    def win_rate(self) -> float | None:
        """
        Learner win rate on this matchup against the anchor opponent.

        :return: The rate, or None if no anchor episode has been played.
        """
        return self.wins / self.games if self.games > 0 else None


@dataclass
class _Stats:
    """
    Counters describing how the buffer is behaving, for logging.
    """

    commits: int = 0
    orphan_commits: int = 0
    evictions: int = 0
    anchor_games: int = 0
    matured: int = 0
    field_names: tuple[str, ...] = field(default=(), repr=False)


class LevelBuffer:
    """
    Prioritized Level Replay buffer over deck-archetype matchups.

    Follows Jiang et al. (2021): each level carries a score estimating how much
    the policy still has to learn from it, and levels are drawn by a mixture of
    a rank-based score distribution and a staleness distribution, the latter
    forcing periodic revisits so no score silently stops describing the current
    policy.

    Two things differ from the reference implementation, both forced by this
    environment:

    * **The score is** ``|mean residual|``, **not** ``mean |residual|``. Reward
      here is terminal-only ±1 over long episodes, so the residual is dominated
      by shuffle and coin-flip variance. Taking the magnitude per step and then
      averaging would score a genuine coin-flip matchup highly forever, since
      the critic correctly predicts 0 and the outcome is ±1 every time.
      Averaging the *signed* residual first cancels that zero-mean noise and
      leaves systematic critic bias, which does decay as the critic learns.
    * **Entries below** ``min_visits`` **are exempt from eviction** and score at
      the maximum. A handful of ±1 outcomes says almost nothing, so evicting on
      such a score would churn levels out before they were ever measured, and
      the optimistic score makes unvisited levels get picked up early without
      needing a separate explore/replay branch.

    That optimistic-score trick is what :meth:`prefill` relies on for a corpus
    that fits entirely within ``capacity``: every level starts unmeasured
    together, so the buffer sweeps the whole space once and then prioritizes.
    It does not work for a corpus *larger* than ``capacity`` (see
    :meth:`commit`): a level discovered lazily after that initial sweep would
    sit at ``score() == inf`` next to an already-prioritizing buffer and pull
    :meth:`distribution` back into full-coverage mode -- for as long as
    discovery keeps running, i.e. permanently. Levels discovered after
    construction instead mature in a side table (:meth:`commit`'s "probation"),
    invisible to :meth:`distribution`, and only enter the scored buffer once
    trustworthy, evicting the weakest entry already there.

    The buffer also tallies win/loss on episodes played against a fixed anchor
    opponent. Those counters take no part in the curriculum; they accumulate
    the matchup matrix that deck selection reads after training.
    """

    def __init__(
            self,
            capacity: int,
            score_temperature: float = 0.9,
            staleness_coefficient: float = 0.4,
            min_visits: int = 5,
            seed: int | None = None,
    ) -> None:
        """
        :param capacity: Maximum number of matchups held at once.
        :param score_temperature: Rank-distribution temperature; lower is
            greedier toward the top-scoring levels.
        :param staleness_coefficient: Weight of the staleness distribution in
            the mixture, in [0, 1]. Higher than the paper's default is
            appropriate here, because self-play moves the league underneath a
            level and ages its score for reasons unrelated to learning.
        :param min_visits: Episodes a level must accumulate before its score is
            trusted and it becomes eligible for eviction.
        :param seed: Unused today; accepted so a future sampling path inside
            the buffer does not change the signature.
        :raises ValueError: If an argument is out of range.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if score_temperature <= 0.0:
            raise ValueError(
                f"score_temperature must be positive, got {score_temperature}"
            )
        if not 0.0 <= staleness_coefficient <= 1.0:
            raise ValueError(
                f"staleness_coefficient must lie in [0, 1], got {staleness_coefficient}"
            )
        if min_visits < 0:
            raise ValueError(f"min_visits must be non-negative, got {min_visits}")
        self._capacity = capacity
        self._score_temperature = score_temperature
        self._staleness_coefficient = staleness_coefficient
        self._min_visits = min_visits
        self._seed = seed
        self._entries: list[LevelEntry] = []
        self._slots: dict[int, int] = {}
        # Levels seen via commit() but not yet in _entries/_slots: maturing
        # outside the scored buffer so they never fool distribution() into
        # full-coverage mode. See the class docstring and commit().
        self._probation: dict[int, LevelEntry] = {}
        self._episodes = 0
        self._stats = _Stats()

    @property
    def capacity(self) -> int:
        """
        :return: Maximum number of matchups the buffer holds.
        """
        return self._capacity

    @property
    def size(self) -> int:
        """
        :return: Number of matchups currently held.
        """
        return len(self._entries)

    @property
    def entries(self) -> tuple[LevelEntry, ...]:
        """
        :return: The held matchups, as an immutable view.
        """
        return tuple(self._entries)

    def prefill(self, pair_ids: Iterable[int]) -> None:
        """
        Seed the buffer with matchups, all unvisited.

        With a capacity above the number of reachable matchups this is the
        whole of "exploration": every level starts at the maximum score, so the
        curriculum works through them all early and then concentrates on
        whichever keep scoring highly. No explore/replay branch is needed.

        :param pair_ids: Matchup identifiers to insert.
        :raises ValueError: If more identifiers are supplied than capacity.
        """
        pair_ids = list(pair_ids)
        if len(pair_ids) > self._capacity:
            raise ValueError(
                f"cannot prefill {len(pair_ids)} levels into a buffer of capacity "
                f"{self._capacity}; raise curriculum.capacity or use coarser archetypes"
            )
        for pair_id in pair_ids:
            self.insert(pair_id)

    def insert(self, pair_id: int) -> int | None:
        """
        Add a matchup, evicting the weakest matured entry if the buffer is full.

        :param pair_id: Matchup identifier to add.
        :return: The slot it occupies, or None if the buffer was full and every
            entry was still below ``min_visits`` (in which case nothing is
            evicted, since no score is yet trustworthy enough to act on).
        """
        if pair_id in self._slots:
            return self._slots[pair_id]
        if self.size < self._capacity:
            self._entries.append(LevelEntry(pair_id=pair_id, last_visit=self._episodes))
            self._slots[pair_id] = self.size - 1
            return self.size - 1
        victim = self._eviction_candidate()
        if victim is None:
            return None
        del self._slots[self._entries[victim].pair_id]
        self._entries[victim] = LevelEntry(pair_id=pair_id, last_visit=self._episodes)
        self._slots[pair_id] = victim
        self._stats.evictions += 1
        return victim

    def commit(
            self,
            pair_id: int,
            mean_residual: float,
            outcome: float | None = None,
    ) -> bool:
        """
        Fold one completed episode into a matchup's statistics.

        A ``pair_id`` the buffer has never registered is only possible when the
        corpus is larger than ``capacity`` and levels are discovered lazily
        rather than prefilled (see :class:`~src.training.curriculum.Curriculum`
        and this class's docstring). Such a level accumulates in a probation
        table -- invisible to :meth:`distribution` -- until it reaches
        ``min_visits`` and is promoted into the scored buffer via
        :meth:`_promote`, evicting the weakest entry already there.

        :param pair_id: Matchup the episode was played on.
        :param mean_residual: Mean signed critic residual over the episode.
        :param outcome: Result of an anchor episode from the learner's
            perspective (1 win, 0.5 draw, 0 loss), or None when the opponent was
            not the anchor and the game should not enter the matchup matrix.
        :return: True if the episode was recorded, whether into the scored
            buffer or into probation. False only if a matured probation entry
            failed to promote because every scored entry was somehow still
            immature -- kept as a defensive fallback; it cannot happen once the
            buffer holds any entry, since only already-matured entries ever
            enter it under lazy discovery.
        """
        self._episodes += 1
        slot = self._slots.get(pair_id)
        if slot is None:
            return self._commit_probation(pair_id, mean_residual, outcome)
        entry = self._entries[slot]
        alpha = 1.0 / (entry.visits + 1)
        entry.mean_residual = (1.0 - alpha) * entry.mean_residual + alpha * mean_residual
        entry.visits += 1
        entry.last_visit = self._episodes
        if outcome is not None:
            entry.wins += outcome
            entry.games += 1.0
            self._stats.anchor_games += 1
        self._stats.commits += 1
        return True

    def _commit_probation(
            self,
            pair_id: int,
            mean_residual: float,
            outcome: float | None,
    ) -> bool:
        """
        Fold an episode into a not-yet-scored level's probation record.

        :param pair_id: Matchup identifier not currently held in the buffer.
        :param mean_residual: Mean signed critic residual over the episode.
        :param outcome: Anchor-episode outcome, or None.
        :return: True once recorded; False if maturing it could not be
            promoted (see :meth:`commit`).
        """
        entry = self._probation.setdefault(pair_id, LevelEntry(pair_id=pair_id))
        alpha = 1.0 / (entry.visits + 1)
        entry.mean_residual = (1.0 - alpha) * entry.mean_residual + alpha * mean_residual
        entry.visits += 1
        if outcome is not None:
            entry.wins += outcome
            entry.games += 1.0
            self._stats.anchor_games += 1
        self._stats.commits += 1
        if entry.visits < self._min_visits:
            return True
        del self._probation[pair_id]
        if self._promote(entry):
            return True
        self._stats.orphan_commits += 1
        return False

    def _promote(self, entry: LevelEntry) -> bool:
        """
        Insert a matured probation entry into the scored buffer.

        Unlike :meth:`insert`, this never places a level at zero visits: it is
        only ever called with an entry that already cleared ``min_visits`` in
        probation, so it is immediately eligible to be an eviction victim
        itself once matured further.

        :param entry: Matured entry to insert.
        :return: True once placed; False if the buffer was full and every held
            entry was somehow still immature (see :meth:`commit`).
        """
        entry.last_visit = self._episodes
        if self.size < self._capacity:
            self._entries.append(entry)
            self._slots[entry.pair_id] = self.size - 1
            return True
        victim = self._eviction_candidate()
        if victim is None:
            return False
        del self._slots[self._entries[victim].pair_id]
        self._entries[victim] = entry
        self._slots[entry.pair_id] = victim
        self._stats.evictions += 1
        return True

    def score(self, entry: LevelEntry) -> float:
        """
        Learning-potential score of one entry.

        :param entry: Entry to score.
        :return: ``|mean_residual|`` once the entry has enough visits to be
            trusted, else the maximum score so it is visited early.
        """
        if entry.visits < self._min_visits:
            return float("inf")
        return abs(entry.mean_residual)

    def distribution(self) -> np.ndarray:
        """
        Sampling probability for each held matchup.

        Runs in two regimes. While any level is still below ``min_visits`` the
        buffer is *measuring*: it has no trustworthy score for those levels, so
        all mass goes uniformly to the least-visited of them (see
        :meth:`_coverage_distribution`). Once every level is measured it
        switches to *prioritizing*, mixing a rank-based score distribution with
        a staleness distribution as in Jiang et al. (2021). Ranking rather than
        using raw scores keeps selection scale-invariant, which matters because
        residual magnitudes shrink over training.

        :return: Probabilities aligned with :attr:`entries`, summing to one.
            Empty if the buffer is empty.
        """
        count = self.size
        if count == 0:
            return np.empty(0, dtype=np.float64)

        coverage = self._coverage_distribution()
        if coverage is not None:
            return coverage

        scores = np.array([self.score(entry) for entry in self._entries])
        # Rank 1 is the highest score; ties resolve by position, which is
        # arbitrary but stable. `inf` scores (unvisited levels) sort first, so
        # they are drawn before anything already measured.
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(count, dtype=np.float64)
        ranks[order] = np.arange(1, count + 1)
        score_dist = (1.0 / ranks) ** (1.0 / self._score_temperature)
        score_dist /= score_dist.sum()

        staleness = np.array(
            [float(self._episodes - entry.last_visit) for entry in self._entries]
        )
        total_staleness = staleness.sum()
        staleness_dist = (
            staleness / total_staleness
            if total_staleness > 0
            else np.full(count, 1.0 / count)
        )

        mixed = (
            1.0 - self._staleness_coefficient
        ) * score_dist + self._staleness_coefficient * staleness_dist
        return mixed / mixed.sum()

    def _coverage_distribution(self) -> np.ndarray | None:
        """
        Sweep the least-measured levels while any still lack a trusted score.

        A level below ``min_visits`` has no score to rank on, so until it is
        measured the buffer is doing survey work rather than prioritization.
        Folding those levels into the rank distribution instead -- by scoring
        them ``inf`` so they sort first -- makes that survey random: the top
        rank takes a large share of the mass, so the same few unmeasured levels
        are drawn repeatedly while others wait. That turns coverage into a
        coupon-collector problem costing on the order of ``n log n`` episodes
        rather than ``n``, which in practice consumed most of a run's budget
        before any prioritization began.

        Restricting the mass to the *joint-least-visited* unmeasured levels and
        spreading it uniformly over them sweeps the space instead: every level
        in the current tier is measured once before the tier below it is
        touched. It also removes a tie-break-by-buffer-position artefact, since
        equally-visited levels now receive equal mass rather than being ordered
        by index.

        :return: The coverage distribution, or None when every level is
            measured and normal prioritization should take over.
        """
        visits = np.array([entry.visits for entry in self._entries])
        unmeasured = visits < self._min_visits
        if not unmeasured.any():
            return None
        tier = unmeasured & (visits == visits[unmeasured].min())
        return tier / tier.sum()

    def pair_ids(self) -> np.ndarray:
        """
        :return: Matchup identifiers aligned with :meth:`distribution`.
        """
        return np.array([entry.pair_id for entry in self._entries], dtype=np.int64)

    def win_rate_matrix(self, archetype_count: int) -> np.ndarray:
        """
        Anchor-opponent win rate per matchup, for deck selection.

        :param archetype_count: Number of archetypes, ``K``.
        :return: ``(K, K)`` array of win rates, NaN where no anchor episode has
            been played. Row is the agent's archetype, column the opponent's.
        """
        matrix = np.full((archetype_count, archetype_count), np.nan)
        for entry in self._entries:
            if entry.games <= 0:
                continue
            agent, opponent = divmod(entry.pair_id, archetype_count)
            matrix[agent, opponent] = entry.wins / entry.games
        return matrix

    def stats(self) -> dict[str, float]:
        """
        Summary counters for logging.

        :return: Metrics describing buffer occupancy, maturity and score spread.
        """
        matured = [entry for entry in self._entries if entry.visits >= self._min_visits]
        summary: dict[str, float] = {
            "curriculum/size": float(self.size),
            "curriculum/matured": float(len(matured)),
            "curriculum/episodes": float(self._episodes),
            "curriculum/orphan_commits": float(self._stats.orphan_commits),
            "curriculum/evictions": float(self._stats.evictions),
            "curriculum/anchor_games": float(self._stats.anchor_games),
            "curriculum/probation_size": float(len(self._probation)),
        }
        if matured:
            scores = np.array([abs(entry.mean_residual) for entry in matured])
            summary.update(
                {
                    "curriculum/score_mean": float(scores.mean()),
                    "curriculum/score_max": float(scores.max()),
                    "curriculum/score_std": float(scores.std()),
                }
            )
        visits = np.array([entry.visits for entry in self._entries], dtype=np.float64)
        if visits.size:
            summary["curriculum/visits_mean"] = float(visits.mean())
            summary["curriculum/visits_min"] = float(visits.min())
        return summary

    def state_dict(self) -> dict[str, list]:
        """
        Serializable snapshot of every entry.

        Deck selection needs the win/loss tallies over a *window* of training
        (the design calls for the last fifth), which a single cumulative total
        cannot provide. Dumping this periodically lets two dumps be differenced
        into any window afterwards.

        :return: Column-oriented entry data plus the episode counter.
        """
        return {
            "episodes": [self._episodes],
            "pair_id": [entry.pair_id for entry in self._entries],
            "mean_residual": [entry.mean_residual for entry in self._entries],
            "visits": [entry.visits for entry in self._entries],
            "last_visit": [entry.last_visit for entry in self._entries],
            "wins": [entry.wins for entry in self._entries],
            "games": [entry.games for entry in self._entries],
        }

    def load_state_dict(self, state: dict[str, Sequence]) -> None:
        """
        Restore entries from a :meth:`state_dict` snapshot.

        :param state: Snapshot produced by :meth:`state_dict`.
        """
        self._entries = [
            LevelEntry(
                pair_id=int(pair_id),
                mean_residual=float(mean_residual),
                visits=int(visits),
                last_visit=int(last_visit),
                wins=float(wins),
                games=float(games),
            )
            for pair_id, mean_residual, visits, last_visit, wins, games in zip(
                state["pair_id"],
                state["mean_residual"],
                state["visits"],
                state["last_visit"],
                state["wins"],
                state["games"],
                strict=True,
            )
        ]
        self._slots = {entry.pair_id: slot for slot, entry in enumerate(self._entries)}
        self._episodes = int(state["episodes"][0])

    def _eviction_candidate(self) -> int | None:
        """
        Slot holding the weakest entry that is safe to discard.

        :return: The slot, or None if no entry has reached ``min_visits``.
        """
        matured = [
            (self.score(entry), slot)
            for slot, entry in enumerate(self._entries)
            if entry.visits >= self._min_visits
        ]
        if not matured:
            return None
        return min(matured)[1]
