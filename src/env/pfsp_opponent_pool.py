import logging
from collections.abc import Callable
from pathlib import Path

from cg.api import Observation

from .snapshot_opponent_pool import SnapshotOpponentPool

logger = logging.getLogger(__name__)

#: Weighting schemes selectable by name; see :class:`PFSPOpponentPool`.
WEIGHTINGS = ("hard", "even")


class PFSPOpponentPool(SnapshotOpponentPool):
    """
    Self-play league sampled by Prioritized Fictitious Self-Play.

    A uniform league spends an ever-larger share of episodes on members the
    learner already beats comfortably, whose games are effectively decided at
    reset. PFSP (Vinyals et al., 2019) instead weights each member by a
    function of ``p``, the learner's running win rate against it, so the budget
    concentrates on members that still contest the game:

    * ``hard`` -- ``(1 - p) ** exponent``, favouring members that beat the
      learner. The default, and the right choice while the league is a ladder
      of strictly improving snapshots.
    * ``even`` -- ``p * (1 - p)``, favouring members the learner is level with.

    Both are offset by ``min_weight`` so no member is ever unreachable: a
    member the learner always beats still needs occasional games, or its win
    rate stops tracking the current policy and the league silently loses a
    reference point.

    Win rates are estimated per worker from the episodes that worker played,
    which needs no inter-process communication: the outcome is the terminal
    reward, which the environment already computes in-process and reports
    through :meth:`record_outcome`. This is unlike the level curriculum, whose
    score is a critic residual the environment cannot see.

    Records are keyed by snapshot path, so a member evicted from the pool and
    later re-loaded resumes its history rather than restarting at the prior.
    """

    def __init__(
            self,
            checkpoint_dir: str | Path,
            load_snapshot: Callable[[Path], Callable[[Observation], list[int]]],
            warmup_opponents: list[Callable[[Observation], list[int]]],
            pool_size: int = 5,
            seed: int | None = None,
            weighting: str = "hard",
            exponent: float = 2.0,
            min_weight: float = 0.05,
            prior_games: float = 2.0,
    ) -> None:
        """
        :param checkpoint_dir: Directory scanned for ``*.pt`` snapshots.
        :param load_snapshot: Turns a snapshot path into an opponent callable.
        :param warmup_opponents: Permanent members faced before any snapshot
            exists, and retained alongside the snapshots afterwards.
        :param pool_size: Number of most-recent snapshots kept.
        :param seed: Seed for the member sampler.
        :param weighting: ``"hard"`` or ``"even"`` (see the class docstring).
        :param exponent: Exponent of the ``hard`` weighting; larger values
            concentrate harder on members that beat the learner. Ignored under
            ``even``.
        :param min_weight: Floor added to every member's weight, so a member
            the learner always beats keeps a small share of episodes.
        :param prior_games: Strength of the ``p = 0.5`` prior, in games. Keeps
            the first few episodes against a new snapshot from pinning its
            weight to an extreme.
        :raises ValueError: If ``weighting`` is unknown or a numeric argument
            is out of range.
        """
        if weighting not in WEIGHTINGS:
            raise ValueError(
                f"unknown weighting {weighting!r}; expected one of {WEIGHTINGS}"
            )
        if min_weight <= 0.0:
            raise ValueError(f"min_weight must be positive, got {min_weight}")
        if prior_games <= 0.0:
            raise ValueError(f"prior_games must be positive, got {prior_games}")
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            load_snapshot=load_snapshot,
            warmup_opponents=warmup_opponents,
            pool_size=pool_size,
            seed=seed,
        )
        self._weighting = weighting
        self._exponent = exponent
        self._min_weight = min_weight
        self._prior_games = prior_games
        # key -> (learner wins, games played), a draw counting as half a win.
        self._records: dict[str, tuple[float, float]] = {}
        self._keys: list[str] = list(self._warmup_keys)
        self._active_key: str | None = self._keys[0] if self._keys else None

    @property
    def win_rates(self) -> dict[str, float]:
        """
        Smoothed learner win rate against every member seen so far.

        :return: Mapping from member key to its posterior win rate.
        """
        return {key: self._win_rate(key) for key in self._records}

    @property
    def member_keys(self) -> tuple[str, ...]:
        """
        Stable identifier of each current member, aligned with
        :attr:`~src.env.opponent_pool.OpponentPool.opponents`.

        :return: The keys as an immutable view.
        """
        return tuple(self._keys)

    @property
    def active_key(self) -> str | None:
        """
        Identifier of the member drawn for the current episode.

        :return: The active member's key, or None before the first reset.
        """
        return self._active_key

    @property
    def active_is_anchor(self) -> bool:
        """
        Whether the member playing this episode is a fixed reference opponent.

        The warmup members never change over a run, which makes them the only
        stationary yardstick in a league that is otherwise chasing the learner.
        The curriculum tallies win rates on anchor episodes alone, so those
        rates stay comparable across training instead of drifting as the league
        strengthens.

        :return: True if the drawn member is a warmup opponent.
        """
        return self._active_key in self._warmup_keys

    @property
    def records(self) -> dict[str, tuple[float, float]]:
        """
        Raw ``(wins, games)`` tally per member, a draw counting as half a win.

        Retained for members no longer in the pool, so a snapshot that is
        evicted and later re-loaded resumes its history.

        :return: A copy of the tally, keyed by member.
        """
        return dict(self._records)

    def on_reset(self) -> None:
        """
        Rescan for snapshots, reweight the league, and draw the next member.
        """
        super().on_reset()
        self._active_key = self._key_of_active()

    def record_outcome(self, reward: float) -> None:
        """
        Record the result of an episode against the member that just played.

        Called by :class:`~src.env.tcg_env.TCGEnv` when a battle terminates.
        Truncated episodes are not reported, since a run cut off by the
        selection cap says nothing about either player's strength.

        A draw scores 0.5, the standard convention for win-rate estimation
        (chess/Elo, Bradley-Terry, and :mod:`src.training.cross_play` alike).
        It has to be the neutral value here rather than a stylistic choice:
        PFSP weights each member by how much it still contests the game, so
        counting a draw as a loss would make drawish members read as hard and
        soak up sampling budget, while counting it as a win would starve them.
        Only 0.5 leaves a drawn series weighting the member exactly as an
        evenly-split one.

        :param reward: Terminal reward from the agent's perspective: positive
            for a win, negative for a loss, zero for a draw.
        """
        if self._active_key is None:
            return
        wins, games = self._records.get(self._active_key, (0.0, 0.0))
        if reward > 0.0:
            score = 1.0
        elif reward < 0.0:
            score = 0.0
        else:
            score = 0.5
        self._records[self._active_key] = (wins + score, games + 1.0)

    def _member_weights(self, keys: list[str]) -> list[float]:
        """
        PFSP weight per league member.

        Also records the key ordering, which :meth:`record_outcome` needs to
        attribute an episode's result to the member that played it.

        :param keys: Stable identifier per member, in member order.
        :return: One weight per member.
        """
        self._keys = list(keys)
        return [self._weight_for(key) for key in keys]

    def _weight_for(self, key: str) -> float:
        """
        Sampling weight of one member under the configured scheme.

        :param key: Member identifier.
        :return: Weight, always at least ``min_weight``.
        """
        win_rate = self._win_rate(key)
        if self._weighting == "hard":
            shaped = (1.0 - win_rate) ** self._exponent
        else:
            shaped = win_rate * (1.0 - win_rate)
        return self._min_weight + shaped

    def _win_rate(self, key: str) -> float:
        """
        Learner win rate against a member, smoothed toward 0.5.

        :param key: Member identifier.
        :return: Posterior win rate in [0, 1].
        """
        wins, games = self._records.get(key, (0.0, 0.0))
        return (wins + 0.5 * self._prior_games) / (games + self._prior_games)

    def _key_of_active(self) -> str | None:
        """
        Identifier of the member the last draw selected.

        Matching is by identity rather than equality because opponents are
        unhashable callables, mirroring
        :meth:`~src.env.opponent_pool.OpponentPool.set_opponents`.

        :return: The active member's key, or None if it cannot be resolved.
        """
        active = self.active
        for key, opponent in zip(self._keys, self._opponents, strict=False):
            if opponent is active:
                return key
        return None
