"""Aggregate parsed replay episodes into deck-refinement statistics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from cg.api import CardType
from submission_analysis.loading import CardIndex, attack_name, card_name
from submission_analysis.parser import AttackUsage, GameEndReason, ParsedEpisode


@dataclass(frozen=True)
class CardStat:
    """One card's play/impact record across the analyzed episodes."""

    card_id: int
    name: str
    is_evolution: bool
    is_pokemon: bool
    games_seen: int
    games_played: int
    games_stuck: int
    wins_when_played: int
    losses_when_played: int
    games_prized: int
    times_knocked_out: int
    games_attacked_with: int

    @property
    def play_rate(self) -> float:
        """Fraction of games it was seen in where it also got played."""
        return self.games_played / self.games_seen if self.games_seen else 0.0

    @property
    def attack_utilization(self) -> float:
        """Fraction of games played in which it landed at least one attack."""
        return self.games_attacked_with / self.games_played if self.games_played else 0.0

    @property
    def win_rate_when_played(self) -> float:
        """Win rate among decided (non-draw) games where it was played."""
        decided = self.wins_when_played + self.losses_when_played
        return self.wins_when_played / decided if decided else 0.0

    @property
    def ko_rate_when_played(self) -> float:
        """Fraction of games played in which it ended up knocked out."""
        return self.times_knocked_out / self.games_played if self.games_played else 0.0

    @property
    def prize_rate(self) -> float:
        """Fraction of games seen where it ended up revealed from our prizes.

        A card can only be revealed from prizes by drawing it out of them, 
        which always adds it to ``cards_seen`` too, so ``games_prized <= games_seen``.
        """
        return self.games_prized / self.games_seen if self.games_seen else 0.0


@dataclass(frozen=True)
class AttackStat:
    """One attack's usage/impact record across the analyzed episodes."""

    attack_id: int
    name: str
    card_id: int
    card_name: str
    uses: int
    total_damage: int
    kos: int

    @property
    def average_damage(self) -> float:
        return self.total_damage / self.uses if self.uses else 0.0


@dataclass(frozen=True)
class EvolutionStat:
    """How often a deployed pre-evolution actually gets evolved."""

    evo_card_id: int
    evo_name: str
    pre_evo_card_id: int
    pre_evo_name: str
    games_pre_evo_played: int
    games_evolved: int

    @property
    def conversion_rate(self) -> float:
        """Fraction of games the pre-evolution was in play that we evolved it."""
        if not self.games_pre_evo_played:
            return 0.0
        return min(1.0, self.games_evolved / self.games_pre_evo_played)


@dataclass(frozen=True)
class LossPostmortem:
    """A port-mortem review for one lost episode."""

    episode_id: int
    opponent_team: str
    no_basic_pokemon: bool
    never_attacked: bool
    ko_deficit: int
    evolution_stalled: bool
    decked_out: bool
    wiped_out: bool
    ended_by_card_effect: bool
    prize_deficit: int

    @property
    def lost_prize_trade(self) -> bool:
        """True when the raw KO count doesn't explain the loss
        (``ko_deficit <= 0``) but the prize-card value of what was knocked
        out still favored the opponent."""
        return self.ko_deficit <= 0 and self.prize_deficit > 0

    @property
    def unexplained(self) -> bool:
        """True when none of the other tags account for this loss.

        A loss can clear every named cause and still happen.
        """
        return not (
            self.no_basic_pokemon
            or self.never_attacked
            or self.evolution_stalled
            or self.ko_deficit > 0
            or self.decked_out
            or self.wiped_out
            or self.ended_by_card_effect
            or self.lost_prize_trade
        )


@dataclass(frozen=True)
class DeckReport:
    """A complete deck-refinement report over a set of parsed episodes."""

    episodes_analyzed: int
    wins: int
    losses: int
    draws: int
    no_basic_pokemon_rate: float
    average_ko_margin: float
    average_prize_margin: float
    average_first_attack_turn_wins: float | None
    average_first_attack_turn_losses: float | None
    first_attack_turns: list[tuple[str, int]]
    ko_margins: list[tuple[str, int]]
    game_lengths: list[tuple[str, int]]
    card_stats: list[CardStat]
    attack_stats: list[AttackStat]
    opponent_attack_stats: list[AttackStat]
    evolution_stats: list[EvolutionStat]
    loss_postmortems: list[LossPostmortem]

    @property
    def win_rate(self) -> float:
        """Win rate among decided (non-draw) games."""
        decided = self.wins + self.losses
        return self.wins / decided if decided else 0.0

    @property
    def average_first_attack_turn(self) -> float | None:
        """Mean turn of first attack across every episode that landed one,
        regardless of result."""
        turns = [turn for _, turn in self.first_attack_turns]
        return sum(turns) / len(turns) if turns else None


class SubmissionSummary(NamedTuple):
    """One submission's headline stats."""

    label: str
    win_rate: float
    average_ko_margin: float
    average_first_attack_turn: float | None


class Recommendation(NamedTuple):
    """One synthesized, actionable note from the report's numbers."""

    category: str
    text: str


def generate_recommendations(report: DeckReport, *, min_games: int = 3) -> list[Recommendation]:
    """
    Turn the report's numbers into a short, prioritized list of what to check.

    :param report: Aggregated deck report to synthesize recommendations from.
    :param min_games: Minimum sample size before a card/evolution-level
        finding is flagged.
    :return: Prioritized recommendations, grouped loosely by category.
    """
    notes: list[Recommendation] = []

    losses = report.loss_postmortems
    if losses:
        total = len(losses)
        causes = [
            ("no Basic Pokemon in opener", sum(1 for loss in losses if loss.no_basic_pokemon)),
            ("never landed an attack", sum(1 for loss in losses if loss.never_attacked)),
            ("evolution stalled", sum(1 for loss in losses if loss.evolution_stalled)),
            ("lost the KO trade", sum(1 for loss in losses if loss.ko_deficit > 0)),
            ("lost the prize trade", sum(1 for loss in losses if loss.lost_prize_trade)),
            ("decked out", sum(1 for loss in losses if loss.decked_out)),
            ("wiped out", sum(1 for loss in losses if loss.wiped_out)),
            ("ended by a card effect", sum(1 for loss in losses if loss.ended_by_card_effect)),
        ]
        top_cause, top_count = max(causes, key=lambda cause: cause[1])
        if top_count and top_count / total >= 0.3:
            notes.append(
                Recommendation(
                    "loss cause",
                    f"'{top_cause}' is behind {top_count}/{total} losses "
                    f"({top_count / total:.0%}) - the single biggest lever for "
                    "improving the win rate.",
                )
            )
        unexplained = sum(1 for loss in losses if loss.unexplained)
        if unexplained and unexplained / total >= 0.2:
            notes.append(
                Recommendation(
                    "loss cause",
                    f"{unexplained}/{total} losses ({unexplained / total:.0%}) don't "
                    "match any tracked cause - worth a manual replay review.",
                )
            )

    wins_turn = report.average_first_attack_turn_wins
    losses_turn = report.average_first_attack_turn_losses
    if wins_turn is not None and losses_turn is not None and losses_turn - wins_turn >= 2:
        notes.append(
            Recommendation(
                "deck speed",
                f"Losses land their first attack ~{losses_turn - wins_turn:.1f} turns "
                f"later than wins (turn {losses_turn:.1f} vs {wins_turn:.1f}) - a slow "
                "opener is costing games.",
            )
        )

    win_lengths = [turn for result, turn in report.game_lengths if result == "win"]
    loss_lengths = [turn for result, turn in report.game_lengths if result == "loss"]
    if win_lengths and loss_lengths:
        avg_win_length = sum(win_lengths) / len(win_lengths)
        avg_loss_length = sum(loss_lengths) / len(loss_lengths)
        if avg_loss_length - avg_win_length >= 5:
            notes.append(
                Recommendation(
                    "game length",
                    f"Losses run ~{avg_loss_length - avg_win_length:.0f} turns longer "
                    f"than wins on average ({avg_loss_length:.0f} vs {avg_win_length:.0f}) "
                    "- the deck gets outgrinded rather than rushed.",
                )
            )
        elif avg_win_length - avg_loss_length >= 5:
            notes.append(
                Recommendation(
                    "game length",
                    f"Losses end ~{avg_win_length - avg_loss_length:.0f} turns faster "
                    f"than wins ({avg_loss_length:.0f} vs {avg_win_length:.0f}) - the "
                    "deck is getting rushed down.",
                )
            )

    stuck = sorted(
        (
            stat
            for stat in report.card_stats
            if stat.games_seen >= min_games and stat.play_rate < 0.6
        ),
        key=lambda stat: stat.play_rate,
    )
    for stat in stuck[:3]:
        notes.append(
            Recommendation(
                "card",
                f"{stat.name} is stuck in hand {1 - stat.play_rate:.0%} of the time it's "
                f"seen ({stat.games_stuck}/{stat.games_seen} games) - a cut candidate.",
            )
        )

    baseline = report.win_rate
    underperformers = sorted(
        (
            stat
            for stat in report.card_stats
            if stat.games_played >= min_games
            and baseline - stat.win_rate_when_played >= 0.15
        ),
        key=lambda stat: stat.win_rate_when_played,
    )
    for stat in underperformers[:3]:
        notes.append(
            Recommendation(
                "card",
                f"{stat.name} wins only {stat.win_rate_when_played:.0%} of games it's "
                f"played, vs a {baseline:.0%} deck average ({stat.games_played} games) "
                "- worth investigating.",
            )
        )

    fragile = sorted(
        (
            stat
            for stat in report.card_stats
            if stat.games_played >= min_games and stat.ko_rate_when_played >= 0.5
        ),
        key=lambda stat: stat.ko_rate_when_played,
        reverse=True,
    )
    for stat in fragile[:3]:
        notes.append(
            Recommendation(
                "card",
                f"{stat.name} is knocked out in {stat.ko_rate_when_played:.0%} of games "
                f"it's played ({stat.times_knocked_out}/{stat.games_played}) - consider "
                "more protection or a bulkier alternative.",
            )
        )

    bench_warmers = sorted(
        (
            stat
            for stat in report.card_stats
            if stat.is_pokemon
            and stat.games_played >= min_games
            and stat.attack_utilization < 0.4
        ),
        key=lambda stat: stat.attack_utilization,
    )
    for stat in bench_warmers[:3]:
        notes.append(
            Recommendation(
                "card",
                f"{stat.name} is played but only attacks in {stat.attack_utilization:.0%} "
                f"of those games ({stat.games_attacked_with}/{stat.games_played}) - check "
                "whether it's dying before its turn or just not contributing.",
            )
        )

    poor_conversion = sorted(
        (
            stat
            for stat in report.evolution_stats
            if stat.games_pre_evo_played >= min_games and stat.conversion_rate < 0.6
        ),
        key=lambda stat: stat.conversion_rate,
    )
    for stat in poor_conversion[:2]:
        notes.append(
            Recommendation(
                "evolution",
                f"{stat.pre_evo_name} only evolves into {stat.evo_name} "
                f"{stat.conversion_rate:.0%} of the games it's in play "
                f"({stat.games_evolved}/{stat.games_pre_evo_played}) - a stalled "
                "evolution line.",
            )
        )

    threats = sorted(report.opponent_attack_stats, key=lambda stat: stat.kos, reverse=True)
    if threats and threats[0].kos >= min_games:
        top_threat = threats[0]
        notes.append(
            Recommendation(
                "threat",
                f"{top_threat.card_name}'s {top_threat.name} is the biggest threat "
                f"across these games, responsible for {top_threat.kos} of our KOs - "
                "consider a tech answer or faster removal.",
            )
        )

    return notes


def _card_stats(
    episodes: Sequence[ParsedEpisode],
    decklist: Sequence[int] | None,
    card_index: CardIndex,
) -> list[CardStat]:
    """
    Aggregate per-card play/impact stats across the given episodes.

    :param episodes: Parsed episodes to aggregate.
    :param decklist: Our deck's card IDs (one entry per physical copy), or
        None; ensures decklist cards never seen still get a zeroed row.
    :param card_index: Card/attack lookup tables for name/type resolution.
    :return: One stat per card seen and/or in the decklist, name-sorted.
    """
    seen: Counter[int] = Counter()
    played: Counter[int] = Counter()
    stuck: Counter[int] = Counter()
    wins_when_played: Counter[int] = Counter()
    losses_when_played: Counter[int] = Counter()
    prized: Counter[int] = Counter()
    knocked_out: Counter[int] = Counter()
    attacked_with: Counter[int] = Counter()

    for episode in episodes:
        for card_id in episode.cards_seen:
            seen[card_id] += 1
            if card_id not in episode.cards_played:
                stuck[card_id] += 1
        for card_id in episode.cards_played:
            played[card_id] += 1
            if episode.result == "win":
                wins_when_played[card_id] += 1
            elif episode.result == "loss":
                losses_when_played[card_id] += 1
        for card_id in episode.prized_cards:
            prized[card_id] += 1
        for card_id in episode.our_pokemon_lost:
            knocked_out[card_id] += 1
        for card_id in {attack.card_id for attack in episode.attacks}:
            attacked_with[card_id] += 1

    card_ids = set(seen) | set(decklist or [])
    stats = []
    for card_id in card_ids:
        card = card_index.cards.get(card_id)
        stats.append(
            CardStat(
                card_id=card_id,
                name=card_name(card_index, card_id),
                is_evolution=bool(card and (card.stage1 or card.stage2)),
                is_pokemon=bool(card and card.cardType == CardType.POKEMON),
                games_seen=seen[card_id],
                games_played=played[card_id],
                games_stuck=stuck[card_id],
                games_attacked_with=attacked_with[card_id],
                wins_when_played=wins_when_played[card_id],
                losses_when_played=losses_when_played[card_id],
                games_prized=prized[card_id],
                times_knocked_out=knocked_out[card_id],
            )
        )
    return sorted(stats, key=lambda stat: stat.name)


def _attack_stats(
    episodes: Sequence[ParsedEpisode],
    card_index: CardIndex,
    *,
    select: Callable[[ParsedEpisode], list[AttackUsage]],
) -> list[AttackStat]:
    """
    Aggregate a per-episode attack list (``select``) into per-attack stats.

    :param episodes: Parsed episodes to aggregate.
    :param card_index: Card/attack lookup tables for name resolution.
    :param select: Extracts the attack-usage list to aggregate from one
        episode (our own attacks, or the opponent's).
    :return: One stat per attack used, most-used first.
    """
    uses: Counter[int] = Counter()
    damage: Counter[int] = Counter()
    kos: Counter[int] = Counter()
    card_for_attack: dict[int, int] = {}
    for episode in episodes:
        for attack in select(episode):
            uses[attack.attack_id] += 1
            damage[attack.attack_id] += attack.damage
            kos[attack.attack_id] += 1 if attack.knocked_out else 0
            card_for_attack[attack.attack_id] = attack.card_id

    stats = [
        AttackStat(
            attack_id=attack_id,
            name=attack_name(card_index, attack_id),
            card_id=card_for_attack[attack_id],
            card_name=card_name(card_index, card_for_attack[attack_id]),
            uses=count,
            total_damage=damage[attack_id],
            kos=kos[attack_id],
        )
        for attack_id, count in uses.items()
    ]
    return sorted(stats, key=lambda stat: stat.uses, reverse=True)


def _evolution_stats(
    episodes: Sequence[ParsedEpisode],
    decklist: Sequence[int] | None,
    card_index: CardIndex,
) -> list[EvolutionStat]:
    """
    Compute how often each deployed pre-evolution actually gets evolved.

    :param episodes: Parsed episodes to aggregate.
    :param decklist: Our deck's card IDs (one entry per physical copy);
        returns an empty list when None, since evolution lines are only
        known from the decklist's own card names.
    :param card_index: Card/attack lookup tables for name/evolution
        resolution.
    :return: One stat per evolution card in the decklist with a resolvable
        pre-evolution, sorted by conversion rate ascending.
    """
    if not decklist:
        return []
    name_to_id = {
        card_index.cards[card_id].name: card_id
        for card_id in decklist
        if card_id in card_index.cards
    }

    stats = []
    for card_id in set(decklist):  # decklist repeats an id once per physical copy
        card = card_index.cards.get(card_id)
        if card is None or not card.evolvesFrom:
            continue
        pre_evo_id = name_to_id.get(card.evolvesFrom)
        if pre_evo_id is None:
            continue
        games_evolved = sum(1 for episode in episodes if card_id in episode.evolutions_made)
        games_pre_evo_played = sum(
            1 for episode in episodes if pre_evo_id in episode.cards_played
        )
        stats.append(
            EvolutionStat(
                evo_card_id=card_id,
                evo_name=card.name,
                pre_evo_card_id=pre_evo_id,
                pre_evo_name=card.evolvesFrom,
                games_pre_evo_played=games_pre_evo_played,
                games_evolved=games_evolved,
            )
        )
    return sorted(stats, key=lambda stat: stat.conversion_rate)


def _prize_value(card_id: int, card_index: CardIndex) -> int:
    """
    Prize cards given up when this card is knocked out.

    :param card_id: The knocked-out card's ID.
    :param card_index: Card/attack lookup tables for ex/megaEx resolution.
    :return: 3 for a Mega Evolution Pokemon ex, 2 for any other Pokemon ex,
        1 otherwise (including when the card can't be resolved).
    """
    card = card_index.cards.get(card_id)
    if card is None:
        return 1
    if card.megaEx:
        return 3
    if card.ex:
        return 2
    return 1


def _prize_value_sum(card_ids: Sequence[int], card_index: CardIndex) -> int:
    """
    Total prize cards given up across a list of knocked-out cards.

    :param card_ids: Knocked-out card IDs, e.g. :attr:`ParsedEpisode.our_kos_cards`.
    :param card_index: Card/attack lookup tables for ex/megaEx resolution.
    :return: The summed prize value.
    """
    return sum(_prize_value(card_id, card_index) for card_id in card_ids)


def _loss_postmortems(
    episodes: Sequence[ParsedEpisode], evolution_card_ids: set[int], card_index: CardIndex
) -> list[LossPostmortem]:
    """
    Tag each lost episode with a quick "why did we lose" cause set.

    :param episodes: Parsed episodes to filter to losses and tag.
    :param evolution_card_ids: Card IDs of evolution-stage cards, for
        detecting a stalled evolution line.
    :param card_index: Card/attack lookup tables for prize-value resolution.
    :return: One postmortem per lost episode.
    """
    postmortems = []
    for episode in episodes:
        if episode.result != "loss":
            continue
        seen_evolutions = episode.cards_seen & evolution_card_ids
        played_evolutions = episode.cards_played & evolution_card_ids
        postmortems.append(
            LossPostmortem(
                episode_id=episode.episode_id,
                opponent_team=episode.opponent_team,
                no_basic_pokemon=episode.had_basic_pokemon is False,
                never_attacked=len(episode.attacks) == 0,
                ko_deficit=episode.our_kos - episode.opponent_kos,
                evolution_stalled=bool(seen_evolutions) and not played_evolutions,
                decked_out=episode.game_end_reason == GameEndReason.DECKED_OUT,
                wiped_out=episode.game_end_reason == GameEndReason.NO_POKEMON_LEFT,
                ended_by_card_effect=episode.game_end_reason == GameEndReason.CARD_EFFECT,
                prize_deficit=(
                    _prize_value_sum(episode.our_kos_cards, card_index)
                    - _prize_value_sum(episode.opponent_kos_cards, card_index)
                ),
            )
        )
    return postmortems


def build_report(
    episodes: Sequence[ParsedEpisode],
    *,
    decklist: Sequence[int] | None,
    card_index: CardIndex,
) -> DeckReport:
    """
    Aggregate parsed episodes into a full deck-refinement report.

    :param episodes: Parsed episodes to aggregate.
    :param decklist: Our deck's card IDs (one entry per physical copy), or
        None to skip never-drawn-card and evolution coverage.
    :param card_index: Card/attack lookup tables for name/type resolution.
    :return: The aggregated report.
    """
    wins = sum(1 for episode in episodes if episode.result == "win")
    losses = sum(1 for episode in episodes if episode.result == "loss")
    draws = sum(1 for episode in episodes if episode.result == "draw")

    no_basic_games = [episode for episode in episodes if episode.had_basic_pokemon is not None]
    no_basic_pokemon_rate = (
        sum(1 for episode in no_basic_games if episode.had_basic_pokemon is False)
        / len(no_basic_games)
        if no_basic_games
        else 0.0
    )
    ko_margin_values = [episode.opponent_kos - episode.our_kos for episode in episodes]
    average_ko_margin = (
        sum(ko_margin_values) / len(ko_margin_values) if ko_margin_values else 0.0
    )
    prize_margin_values = [
        _prize_value_sum(episode.opponent_kos_cards, card_index)
        - _prize_value_sum(episode.our_kos_cards, card_index)
        for episode in episodes
    ]
    average_prize_margin = (
        sum(prize_margin_values) / len(prize_margin_values) if prize_margin_values else 0.0
    )

    def _average_first_attack_turn(result: str) -> float | None:
        turns = [
            episode.first_attack_turn
            for episode in episodes
            if episode.result == result and episode.first_attack_turn is not None
        ]
        return sum(turns) / len(turns) if turns else None

    evolution_card_ids = {
        card_id
        for card_id in set(decklist or []) | {c for e in episodes for c in e.cards_seen}
        if (card := card_index.cards.get(card_id)) and (card.stage1 or card.stage2)
    }

    return DeckReport(
        episodes_analyzed=len(episodes),
        wins=wins,
        losses=losses,
        draws=draws,
        no_basic_pokemon_rate=no_basic_pokemon_rate,
        average_ko_margin=average_ko_margin,
        average_prize_margin=average_prize_margin,
        average_first_attack_turn_wins=_average_first_attack_turn("win"),
        average_first_attack_turn_losses=_average_first_attack_turn("loss"),
        first_attack_turns=[
            (episode.result, episode.first_attack_turn)
            for episode in episodes
            if episode.first_attack_turn is not None
        ],
        ko_margins=[
            (episode.result, episode.opponent_kos - episode.our_kos) for episode in episodes
        ],
        game_lengths=[
            (episode.result, episode.final_turn)
            for episode in episodes
            if episode.final_turn is not None
        ],
        card_stats=_card_stats(episodes, decklist, card_index),
        attack_stats=_attack_stats(episodes, card_index, select=lambda e: e.attacks),
        opponent_attack_stats=_attack_stats(
            episodes, card_index, select=lambda e: e.opponent_attacks
        ),
        evolution_stats=_evolution_stats(episodes, decklist, card_index),
        loss_postmortems=_loss_postmortems(episodes, evolution_card_ids, card_index),
    )
