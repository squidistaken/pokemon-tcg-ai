import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

#: Normal quantile for a two-sided 95% interval.
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class EpisodeRecord:
    """
    Outcome of one played game, scored from the tracked policy's seat.

    :param score: 1.0 win, 0.5 draw, 0.0 loss for the policy in the agent seat.
    :param terminated: False when the episode was truncated instead of decided;
        such episodes carry no outcome and are excluded from every rate.
    :param agent_seat: Engine seat the tracked policy occupied, 0 or 1. Seat 0
        moves first, so this is the covariate the design has to balance.
    :param deck: Archetype label of the deck the tracked policy piloted.
    :param opponent_deck: Archetype label the opponent piloted. Equal to
        ``deck`` under a mirror matchup.
    :param steps: Agent decisions taken, a rough proxy for game length.
    """

    score: float
    terminated: bool
    agent_seat: int
    deck: str
    opponent_deck: str
    steps: int


@dataclass(frozen=True)
class Interval:
    """
    A point estimate with a two-sided confidence interval.

    :param point: The estimate.
    :param low: Lower confidence limit.
    :param high: Upper confidence limit.
    """

    point: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.low:.4f}, {self.high:.4f}]"


class MatchStatistics:
    """
    Turns per-episode head-to-head outcomes into a defensible verdict.

    Three things make the naive "wins over games" number misleading here, and
    each has a method:

    * **Draws.** They are outcomes, not missing data, so they score 0.5 and the
      estimate is a mean score rather than a win proportion. The Wilson interval
      that suits a proportion is kept for the decided-games-only view.
    * **Seat.** Seat 0 moves first. An unbalanced seat split shifts the estimate
      by the size of that advantage, which :meth:`seat_split` exposes and which
      the two-way design cancels.
    * **Deck.** Games on the same archetype are correlated: a deck the policy
      pilots badly loses many games together, so treating every episode as an
      independent Bernoulli draw understates the variance. :meth:`cluster_interval`
      resamples whole archetype blocks instead of individual games.

    The null hypothesis is 0.5 exactly. A mirror matchup gives both seats the
    same deck, so under equal skill the only asymmetry left is the seat, and the
    two-way design removes that too.
    """

    def __init__(self, records: Sequence[EpisodeRecord], seed: int = 0) -> None:
        """
        :param records: Episodes scored from the tracked policy's seat, pooled
            across both directions of the match.
        :param seed: Seed for the bootstrap resampler, so a report is repeatable.
        """
        self._records = [record for record in records if record.terminated]
        self._unfinished = len(records) - len(self._records)
        self._seed = seed

    @property
    def games(self) -> int:
        """
        :return: Decided games backing every estimate below.
        """
        return len(self._records)

    @property
    def unfinished(self) -> int:
        """
        :return: Episodes that never terminated and were dropped.
        """
        return self._unfinished

    @property
    def wins(self) -> int:
        """
        :return: Outright wins, draws excluded.
        """
        return sum(1 for record in self._records if record.score > 0.5)

    @property
    def draws(self) -> int:
        """
        :return: Drawn games.
        """
        return sum(1 for record in self._records if record.score == 0.5)

    @property
    def score(self) -> float:
        """
        :return: Mean score with a draw worth half a win, or 0.5 with no games.
        """
        if not self._records:
            return 0.5
        return sum(record.score for record in self._records) / len(self._records)

    def naive_interval(self) -> Interval:
        """
        Normal-approximation interval treating every game as independent.

        Reported as the optimistic bound, and the one to distrust: it ignores
        the archetype clustering that :meth:`cluster_interval` prices in.

        :return: Mean score with its 95% interval.
        """
        n = len(self._records)
        if n < 2:
            return Interval(self.score, 0.0, 1.0)
        mean = self.score
        variance = sum((record.score - mean) ** 2 for record in self._records) / (n - 1)
        half_width = Z_95 * math.sqrt(variance / n)
        return Interval(mean, mean - half_width, mean + half_width)

    def cluster_interval(self, resamples: int = 10000) -> Interval:
        """
        Two-stage bootstrap interval over archetype blocks.

        Each block is one (piloted deck, opposing deck) cell. A replicate draws
        blocks with replacement, then draws games with replacement inside each
        block. Both stages are needed: resampling blocks alone reproduces the
        correlation between games that share a matchup but collapses to zero
        width when every block happens to have the same mean, and resampling
        games alone is the episode-level interval this method exists to replace.

        :param resamples: Bootstrap replicates.
        :return: Mean score with its percentile 95% interval.
        """
        blocks = list(self._blocks().values())
        if len(blocks) < 2:
            return self.naive_interval()
        rng = random.Random(self._seed)
        means: list[float] = []
        for _ in range(resamples):
            total = 0.0
            count = 0
            for _ in range(len(blocks)):
                scores = blocks[rng.randrange(len(blocks))]
                total += sum(scores[rng.randrange(len(scores))] for _ in scores)
                count += len(scores)
            if count:
                means.append(total / count)
        means.sort()
        return Interval(
            self.score,
            means[int(0.025 * len(means))],
            means[min(int(0.975 * len(means)), len(means) - 1)],
        )

    def binomial_p_value(self) -> float:
        """
        Exact two-sided binomial test of the win share against 0.5.

        Runs on decided-and-non-drawn games, which is the only subset on which
        "wins out of trials" is a proper Bernoulli sample.

        :return: The p-value; 1.0 when there are no non-drawn games.
        """
        trials = self.games - self.draws
        if trials == 0:
            return 1.0
        successes = self.wins
        log_half = math.log(0.5) * trials
        observed = math.lgamma(trials + 1) - (
            math.lgamma(successes + 1) + math.lgamma(trials - successes + 1)
        )
        tolerance = 1e-9
        total = 0.0
        for k in range(trials + 1):
            log_prob = math.lgamma(trials + 1) - (
                math.lgamma(k + 1) + math.lgamma(trials - k + 1)
            )
            if log_prob <= observed + tolerance:
                total += math.exp(log_prob + log_half)
        return min(1.0, total)

    def elo_interval(self, interval: Interval) -> Interval:
        """
        Convert a score interval into an Elo difference.

        :param interval: Score estimate on the 0-1 scale.
        :return: Elo points the tracked policy is ahead by, same coverage.
        """
        return Interval(
            self._to_elo(interval.point),
            self._to_elo(interval.low),
            self._to_elo(interval.high),
        )

    def seat_split(self) -> dict[int, tuple[int, float]]:
        """
        Score broken down by the seat the tracked policy occupied.

        :return: Seat -> (games, mean score). A gap between seat 0 and seat 1 is
            the first-player advantage, not a property of either policy.
        """
        by_seat: dict[int, list[float]] = defaultdict(list)
        for record in self._records:
            by_seat[record.agent_seat].append(record.score)
        return {
            seat: (len(scores), sum(scores) / len(scores))
            for seat, scores in sorted(by_seat.items())
        }

    def archetype_split(self) -> dict[str, tuple[int, float]]:
        """
        Score broken down by the archetype the tracked policy piloted.

        :return: Archetype -> (games, mean score), worst first. The tail is
            where a policy that looks level on average is actually losing.
        """
        by_deck: dict[str, list[float]] = defaultdict(list)
        for record in self._records:
            by_deck[record.deck].append(record.score)
        rates = {
            deck: (len(scores), sum(scores) / len(scores))
            for deck, scores in by_deck.items()
        }
        return dict(sorted(rates.items(), key=lambda item: item[1][1]))

    def games_for_precision(self, half_width: float) -> int:
        """
        Games needed for a naive interval of the requested half-width.

        Uses the observed score variance, so it answers "how much longer" from
        data already collected rather than from a worst-case assumption.

        :param half_width: Target 95% half-width on the 0-1 score scale.
        :return: Total decided games required.
        """
        n = len(self._records)
        if n < 2 or half_width <= 0:
            return 0
        mean = self.score
        variance = sum((record.score - mean) ** 2 for record in self._records) / (n - 1)
        return math.ceil(variance * (Z_95 / half_width) ** 2)

    def _blocks(self) -> dict[tuple[str, str], list[float]]:
        """
        Group episode scores by matchup cell.

        :return: (piloted deck, opposing deck) -> scores in that cell.
        """
        blocks: dict[tuple[str, str], list[float]] = defaultdict(list)
        for record in self._records:
            blocks[(record.deck, record.opponent_deck)].append(record.score)
        return blocks

    @staticmethod
    def _to_elo(score: float) -> float:
        """
        :param score: Expected score on the 0-1 scale.
        :return: Equivalent Elo difference, clamped at +/-800 for degenerate ends.
        """
        score = min(max(score, 1e-6), 1 - 1e-6)
        return max(-800.0, min(800.0, -400.0 * math.log10(1.0 / score - 1.0)))
