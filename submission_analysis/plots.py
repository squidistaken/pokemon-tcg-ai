from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from submission_analysis.console import console
from submission_analysis.deck_report import (
    AttackStat,
    CardStat,
    DeckReport,
    EvolutionStat,
    SubmissionSummary,
)
from submission_analysis.loading import RatingHistoryRow

_BLUE = "#4c72b0"
_GREEN = "#55a868"
_RED = "#c44e52"
_ORANGE = "#dd8452"


def plot_card_play_rate(card_stats: list[CardStat], out_path: Path, *, top: int = 30) -> None:
    """
    Save a horizontal bar chart of play rate, lowest first (most-stuck cards).

    :param card_stats: Per-card stats to chart.
    :param out_path: PNG path to write.
    :param top: Maximum cards shown.
    :return: None.
    """
    seen = sorted(
        (stat for stat in card_stats if stat.games_seen > 0), key=lambda stat: stat.play_rate
    )[:top]
    if not seen:
        return
    labels = [stat.name for stat in seen]
    values = [stat.play_rate for stat in seen]
    # Below this, a card is stuck often enough to be a real cut candidate;
    # above it, most of the bars in a top-N-by-play-rate view are actually
    # fine and shouldn't visually compete with the ones that aren't.
    colors = [_RED if value < 0.6 else _ORANGE for value in values]
    plt.figure(figsize=(9, max(4.0, len(labels) * 0.28)))
    plt.barh(range(len(labels)), values, color=colors)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.gca().invert_yaxis()
    plt.xlabel("play rate (played / seen)")
    plt.xlim(0, 1)
    plt.title("Cards most often stuck in hand (lowest play rate first)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_win_rate_when_played(
    card_stats: list[CardStat], out_path: Path, *, top: int = 30, win_rate_baseline: float = 0.5
) -> None:
    """
    Save a horizontal bar chart of win rate when played, lowest first.

    Colored relative to ``win_rate_baseline`` (pass the submission's own
    overall win rate), not a fixed 0.4/0.5: for a submission that's losing
    most of its games, every card sits below a fixed threshold and the
    chart turns uniformly red, which says "this submission is losing" (already
    obvious from the overview) rather than "these specific cards are the
    problem" (the actual point of this chart).

    :param card_stats: Per-card stats to chart.
    :param out_path: PNG path to write.
    :param top: Maximum cards shown.
    :param win_rate_baseline: Reference line and color threshold; pass the
        submission's own overall win rate.
    :return: None.
    """
    played = sorted(
        (stat for stat in card_stats if stat.games_played > 0),
        key=lambda stat: stat.win_rate_when_played,
    )[:top]
    if not played:
        return
    labels = [stat.name for stat in played]
    values = [stat.win_rate_when_played for stat in played]
    colors = [_RED if value < win_rate_baseline else _GREEN for value in values]
    plt.figure(figsize=(9, max(4.0, len(labels) * 0.28)))
    plt.barh(range(len(labels)), values, color=colors)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.gca().invert_yaxis()
    plt.axvline(win_rate_baseline, color="black", linewidth=0.6, linestyle="--")
    plt.xlabel("win rate in games it was played")
    plt.xlim(0, 1)
    plt.title("Win rate when played (lowest first)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_ko_rate(card_stats: list[CardStat], out_path: Path, *, top: int = 30) -> None:
    """
    Save a horizontal bar chart of KO rate when played, highest (most fragile) first.

    :param card_stats: Per-card stats to chart.
    :param out_path: PNG path to write.
    :param top: Maximum cards shown.
    :return: None.
    """
    fragile = sorted(
        (stat for stat in card_stats if stat.times_knocked_out > 0),
        key=lambda stat: stat.ko_rate_when_played,
        reverse=True,
    )[:top]
    if not fragile:
        return
    labels = [stat.name for stat in fragile]
    values = [stat.ko_rate_when_played for stat in fragile]
    plt.figure(figsize=(9, max(4.0, len(labels) * 0.28)))
    plt.barh(range(len(labels)), values, color=_RED)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.gca().invert_yaxis()
    plt.xlabel("knocked out / played")
    plt.xlim(0, 1)
    plt.title("Cards most often knocked out (highest KO rate first)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_attack_utilization(card_stats: list[CardStat], out_path: Path, *, top: int = 20) -> None:
    """
    Save a horizontal bar chart of attack utilization, lowest first.

    Filtered to Pokemon only (``is_pokemon``): Trainer/Energy cards
    structurally never attack, so they'd always show 0% for a reason
    unrelated to deck quality - the same filter ``plot_ko_rate_vs_win_rate``
    applies for the same reason.

    :param card_stats: Per-card stats to chart.
    :param out_path: PNG path to write.
    :param top: Maximum cards shown.
    :return: None.
    """
    played = sorted(
        (stat for stat in card_stats if stat.is_pokemon and stat.games_played > 0),
        key=lambda stat: stat.attack_utilization,
    )[:top]
    if not played:
        return
    labels = [stat.name for stat in played]
    values = [stat.attack_utilization for stat in played]
    colors = [_RED if value < 0.4 else _ORANGE for value in values]
    plt.figure(figsize=(9, max(4.0, len(labels) * 0.28)))
    plt.barh(range(len(labels)), values, color=colors)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.gca().invert_yaxis()
    plt.xlabel("games attacked with / games played")
    plt.xlim(0, 1)
    plt.title("Played but rarely attacks with (lowest utilization first)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def _annotate_outliers(
    points: list[tuple[float, float, str]],
    *,
    is_outlier: Callable[[float, float], bool],
) -> None:
    """
    Label only the points worth calling out, not every point.

    A card list is dense enough (20-30 cards) that labeling every point
    makes the "boring middle" cluster an unreadable smear of overlapping
    text - the labels that matter are the ones away from the reference
    lines, which is exactly what makes a point interesting here.

    :param points: ``(x, y, label)`` for every point on the scatter.
    :param is_outlier: Predicate deciding whether a point's ``(x, y)``
        is worth labeling.
    :return: None.
    """
    for x, y, name in points:
        if is_outlier(x, y):
            plt.annotate(
                name, (x, y), fontsize=6, xytext=(4, 4), textcoords="offset points"
            )


def plot_play_rate_vs_win_rate(
    card_stats: list[CardStat], out_path: Path, *, win_rate_baseline: float = 0.5
) -> None:
    """
    Save a scatter of play rate vs win rate when played, one point per card.

    Neither single-metric bar chart (``plot_card_play_rate``,
    ``plot_win_rate_when_played``) shows this: a card that's played often
    but loses when it is (top-left) is a worse problem than one that's just
    stuck in hand a lot but wins when it lands (bottom-right, more of a
    draw/consistency issue than a card-quality one). ``win_rate_baseline``
    (pass the submission's own overall win rate) is what the reference line
    and outlier labeling are measured against, not a fixed 0.5 - for a
    submission that's losing most of its games, every card sits well below
    a fixed 0.5 and the labels would smear into an unreadable cluster.

    :param card_stats: Per-card stats to chart.
    :param out_path: PNG path to write.
    :param win_rate_baseline: Reference line and outlier threshold; pass the
        submission's own overall win rate.
    :return: None.
    """
    played = [stat for stat in card_stats if stat.games_played > 0]
    if not played:
        return
    x = [stat.play_rate for stat in played]
    y = [stat.win_rate_when_played for stat in played]
    sizes = [30 + stat.games_seen * 6 for stat in played]
    plt.figure(figsize=(9, 7))
    plt.scatter(x, y, s=sizes, color=_BLUE, alpha=0.7, edgecolors="white", linewidths=0.5)
    _annotate_outliers(
        [(stat.play_rate, stat.win_rate_when_played, stat.name) for stat in played],
        is_outlier=lambda px, py: px < 0.6 or abs(py - win_rate_baseline) > 0.15,
    )
    plt.axhline(win_rate_baseline, color="black", linewidth=0.6, linestyle="--")
    plt.axvline(0.5, color="black", linewidth=0.6, linestyle="--")
    plt.xlabel("play rate (played / seen)")
    plt.ylabel("win rate when played")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.title("Play rate vs win rate (bubble size = games seen)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_ko_rate_vs_win_rate(
    card_stats: list[CardStat], out_path: Path, *, win_rate_baseline: float = 0.5
) -> None:
    """
    Save a scatter of KO rate vs win rate when played, one point per card.

    Distinguishes a card that dies a lot but still carries its games
    (top-right - probably fine, it's doing its job before going down) from
    one that dies a lot AND loses when it does (bottom-right - the clearer
    cut candidate). Filtered to ``times_knocked_out > 0`` (matches
    ``plot_ko_rate``): Trainer/Energy cards can never be knocked out, so
    without this filter every non-Pokemon card in the deck piles up at
    x=0 and swamps the cards this plot is actually about. Reference line and
    outlier labeling are relative to ``win_rate_baseline`` (the submission's
    own overall win rate) for the same reason as ``plot_play_rate_vs_win_rate``.

    :param card_stats: Per-card stats to chart.
    :param out_path: PNG path to write.
    :param win_rate_baseline: Reference line and outlier threshold; pass the
        submission's own overall win rate.
    :return: None.
    """
    fragile = [stat for stat in card_stats if stat.times_knocked_out > 0]
    if not fragile:
        return
    x = [stat.ko_rate_when_played for stat in fragile]
    y = [stat.win_rate_when_played for stat in fragile]
    sizes = [30 + stat.games_played * 6 for stat in fragile]
    plt.figure(figsize=(9, 7))
    plt.scatter(x, y, s=sizes, color=_RED, alpha=0.7, edgecolors="white", linewidths=0.5)
    _annotate_outliers(
        [(stat.ko_rate_when_played, stat.win_rate_when_played, stat.name) for stat in fragile],
        is_outlier=lambda px, py: px > 0.4 or abs(py - win_rate_baseline) > 0.15,
    )
    plt.axhline(win_rate_baseline, color="black", linewidth=0.6, linestyle="--")
    plt.xlabel("KO rate when played (fragility)")
    plt.ylabel("win rate when played")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.title("Fragility vs win rate (bubble size = games played)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_evolution_conversion_rate(
    evolution_stats: list[EvolutionStat], out_path: Path, *, top: int = 20
) -> None:
    """
    Save a horizontal bar chart of evolution conversion rate, lowest first.

    Only lines where the pre-evolution was actually deployed at least once
    are shown - a line that never got that far is a draw/consistency issue
    (already covered by ``plot_card_play_rate``), not a conversion one.

    :param evolution_stats: Per-evolution-line stats to chart.
    :param out_path: PNG path to write.
    :param top: Maximum lines shown.
    :return: None.
    """
    deployed = sorted(
        (stat for stat in evolution_stats if stat.games_pre_evo_played > 0),
        key=lambda stat: stat.conversion_rate,
    )[:top]
    if not deployed:
        return
    labels = [f"{stat.pre_evo_name} -> {stat.evo_name}" for stat in deployed]
    values = [stat.conversion_rate for stat in deployed]
    colors = [_RED if value < 0.6 else _GREEN for value in values]
    plt.figure(figsize=(9, max(4.0, len(labels) * 0.28)))
    plt.barh(range(len(labels)), values, color=colors)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.gca().invert_yaxis()
    plt.xlabel("evolved / games pre-evolution was played")
    plt.xlim(0, 1)
    plt.title("Evolution conversion rate (lowest first)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_attack_usage(attack_stats: list[AttackStat], out_path: Path, *, top: int = 15) -> None:
    """
    Save a bar chart of attack use counts, annotated with average damage.

    :param attack_stats: Per-attack stats to chart.
    :param out_path: PNG path to write.
    :param top: Maximum attacks shown.
    :return: None.
    """
    stats = attack_stats[:top]
    if not stats:
        return
    labels = [f"{stat.name}\n({stat.card_name})" for stat in stats]
    uses = [stat.uses for stat in stats]
    plt.figure(figsize=(max(8.0, len(labels) * 0.6), 6))
    bars = plt.bar(range(len(labels)), uses, color=_BLUE)
    for bar, stat in zip(bars, stats, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{stat.average_damage:.0f} dmg",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    plt.xticks(range(len(labels)), labels, fontsize=7, rotation=45, ha="right")
    plt.ylabel("uses")
    plt.title("Attack usage (bar height = uses, label = average damage)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_damage_by_card(attack_stats: list[AttackStat], out_path: Path, *, top: int = 15) -> None:
    """
    Save a bar chart of total damage dealt, summed across each card's attacks.

    Complements ``plot_attack_usage`` (which ranks individual attacks): a
    Pokemon with two mediocre attacks can out-damage one flashy hard-hitter
    once everything it knows is added up.

    :param attack_stats: Per-attack stats to sum by card and chart.
    :param out_path: PNG path to write.
    :param top: Maximum cards shown.
    :return: None.
    """
    if not attack_stats:
        return
    damage_by_card: dict[int, int] = {}
    name_by_card: dict[int, str] = {}
    for stat in attack_stats:
        damage_by_card[stat.card_id] = damage_by_card.get(stat.card_id, 0) + stat.total_damage
        name_by_card[stat.card_id] = stat.card_name
    ordered = sorted(damage_by_card.items(), key=lambda item: item[1], reverse=True)[:top]
    labels = [name_by_card[card_id] for card_id, _ in ordered]
    values = [damage for _, damage in ordered]
    plt.figure(figsize=(max(8.0, len(labels) * 0.5), 6))
    plt.bar(range(len(labels)), values, color=_ORANGE)
    plt.xticks(range(len(labels)), labels, fontsize=8, rotation=45, ha="right")
    plt.ylabel("total damage dealt")
    plt.title("Total damage dealt by card (summed across its attacks)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_opponent_attack_usage(
    opponent_attack_stats: list[AttackStat], out_path: Path, *, top: int = 15
) -> None:
    """
    Save a bar chart of the opponent attacks/cards most responsible for our KOs.

    Threat-intel counterpart to ``plot_attack_usage``: sorted by KOs caused
    (the direct "how dangerous is this" signal), not uses - a frequently
    used but weak attack matters less than a rare but lethal one.

    :param opponent_attack_stats: Per-opponent-attack stats to chart.
    :param out_path: PNG path to write.
    :param top: Maximum attacks shown.
    :return: None.
    """
    stats = sorted(opponent_attack_stats, key=lambda stat: stat.kos, reverse=True)[:top]
    if not stats:
        return
    labels = [f"{stat.name}\n({stat.card_name})" for stat in stats]
    kos = [stat.kos for stat in stats]
    plt.figure(figsize=(max(8.0, len(labels) * 0.6), 6))
    bars = plt.bar(range(len(labels)), kos, color=_RED)
    for bar, stat in zip(bars, stats, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{stat.average_damage:.0f} dmg",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    plt.xticks(range(len(labels)), labels, fontsize=7, rotation=45, ha="right")
    plt.ylabel("our Pokemon knocked out")
    plt.title("Threats: opponent attacks against us (label = average damage)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_loss_causes(report: DeckReport, out_path: Path) -> None:
    """
    Save a bar chart of why the analyzed losses happened.

    :param report: Aggregated deck report to draw loss postmortems from.
    :param out_path: PNG path to write.
    :return: None.
    """
    losses = report.loss_postmortems
    if not losses:
        return
    total = len(losses)
    causes = {
        "no basic\npokemon": sum(1 for loss in losses if loss.no_basic_pokemon),
        "never\nattacked": sum(1 for loss in losses if loss.never_attacked),
        "evolution\nstalled": sum(1 for loss in losses if loss.evolution_stalled),
        "lost KO\ntrade": sum(1 for loss in losses if loss.ko_deficit > 0),
        "lost prize\ntrade": sum(1 for loss in losses if loss.lost_prize_trade),
        "decked\nout": sum(1 for loss in losses if loss.decked_out),
        "wiped\nout": sum(1 for loss in losses if loss.wiped_out),
        "card\neffect": sum(1 for loss in losses if loss.ended_by_card_effect),
        "other": sum(1 for loss in losses if loss.unexplained),
    }
    labels = list(causes.keys())
    values = [causes[label] / total for label in labels]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color=_RED)
    for bar, value in zip(bars, values, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.xticks(fontsize=8)
    plt.ylabel("share of losses")
    plt.ylim(0, 1)
    plt.title(f"Why we lost ({total} losses analyzed)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_first_attack_turn_distribution(
    first_attack_turns: list[tuple[str, int]], out_path: Path, *, cap: int = 15
) -> None:
    """
    Save a grouped bar chart of the turn we land our first attack, wins vs losses.

    A single mean (as shown in the overview table) can't distinguish "we
    always attack by turn 3" from "half our games attack turn 2, half turn
    8" - this shows the actual shape. Plotted as explicit per-turn counts
    (not ``plt.hist``, whose grouped bars don't land on integer turns) and
    capped at ``cap`` turns (folded into a "<cap>+" bucket): one rare
    slow game would otherwise stretch the whole axis and compress every
    normal-length game into a sliver.

    :param first_attack_turns: ``(result, turn)`` per episode that landed a
        first attack.
    :param out_path: PNG path to write.
    :param cap: Turns beyond this are folded into a single overflow bucket.
    :return: None.
    """
    if not first_attack_turns:
        return

    def _bucket(turn: int) -> int:
        return min(turn, cap)

    wins = Counter(_bucket(turn) for result, turn in first_attack_turns if result == "win")
    losses = Counter(_bucket(turn) for result, turn in first_attack_turns if result == "loss")
    turns = sorted(set(wins) | set(losses))
    labels = [str(turn) if turn < cap else f"{cap}+" for turn in turns]
    positions = range(len(turns))
    width = 0.4
    plt.figure(figsize=(max(8.0, len(turns) * 0.5), 5))
    plt.bar(
        [p - width / 2 for p in positions],
        [wins.get(turn, 0) for turn in turns],
        width=width,
        color=_GREEN,
        label="wins",
    )
    plt.bar(
        [p + width / 2 for p in positions],
        [losses.get(turn, 0) for turn in turns],
        width=width,
        color=_RED,
        label="losses",
    )
    plt.xticks(list(positions), labels)
    plt.xlabel("turn of first attack")
    plt.ylabel("games")
    plt.legend()
    plt.title("When we land our first attack (wins vs losses)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_game_length_distribution(
    game_lengths: list[tuple[str, int]], out_path: Path, *, cap: int = 30
) -> None:
    """
    Save a grouped bar chart of how many turns games ran, by result.

    Are our losses fast blowouts (opponent snowballs early) or long grinds
    (we keep the game close but can't close it out)? Those call for very
    different fixes. Same capped-bucket technique as
    ``plot_first_attack_turn_distribution``, for the same reason: one very
    long game would otherwise swamp the axis.

    :param game_lengths: ``(result, final_turn)`` per episode with turn data.
    :param out_path: PNG path to write.
    :param cap: Turns beyond this are folded into a single overflow bucket.
    :return: None.
    """
    if not game_lengths:
        return

    def _bucket(turn: int) -> int:
        return min(turn, cap)

    wins = Counter(_bucket(turn) for result, turn in game_lengths if result == "win")
    losses = Counter(_bucket(turn) for result, turn in game_lengths if result == "loss")
    draws = Counter(_bucket(turn) for result, turn in game_lengths if result == "draw")
    turns = sorted(set(wins) | set(losses) | set(draws))
    labels = [str(turn) if turn < cap else f"{cap}+" for turn in turns]
    positions = range(len(turns))
    width = 0.27
    plt.figure(figsize=(max(8.0, len(turns) * 0.5), 5))
    plt.bar(
        [p - width for p in positions],
        [wins.get(turn, 0) for turn in turns],
        width=width,
        color=_GREEN,
        label="wins",
    )
    plt.bar(
        [p for p in positions],
        [losses.get(turn, 0) for turn in turns],
        width=width,
        color=_RED,
        label="losses",
    )
    plt.bar(
        [p + width for p in positions],
        [draws.get(turn, 0) for turn in turns],
        width=width,
        color=_BLUE,
        label="draws",
    )
    plt.xticks(list(positions), labels)
    plt.xlabel("final turn")
    plt.ylabel("games")
    plt.legend()
    plt.title("Game length (final turn reached), by result")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_ko_margin_distribution(ko_margins: list[tuple[str, int]], out_path: Path) -> None:
    """
    Save a stacked histogram of KO margin (opponent KOs - our KOs) by result.

    Complements ``average_ko_margin`` (a single blended number) by showing
    whether wins tend to be blowouts, losses close, or the reverse.

    :param ko_margins: ``(result, opponent_kos - our_kos)`` per episode.
    :param out_path: PNG path to write.
    :return: None.
    """
    if not ko_margins:
        return
    wins = [margin for result, margin in ko_margins if result == "win"]
    losses = [margin for result, margin in ko_margins if result == "loss"]
    draws = [margin for result, margin in ko_margins if result == "draw"]
    lo = min(margin for _, margin in ko_margins)
    hi = max(margin for _, margin in ko_margins)
    bins = range(lo, hi + 2)
    plt.figure(figsize=(8, 5))
    plt.hist(
        [wins, losses, draws],
        bins=bins,
        label=["wins", "losses", "draws"],
        color=[_GREEN, _RED, _BLUE],
        stacked=True,
    )
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("KO margin (opponent KOs - our KOs)")
    plt.ylabel("games")
    plt.legend()
    plt.title("KO margin distribution by game result")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_rating_history(history: list[RatingHistoryRow], out_path: Path) -> None:
    """
    Save a line plot of each submission label's rating over time.

    Kaggle only ever reports a submission's *current* rating; this plot only
    has data for the invocations of ``submission_analysis.submissions``' the
    ``status`` command that happened to run while a submission was live.

    :param history: Rating-history rows to chart.
    :param out_path: PNG path to write.
    :return: None.
    """
    scored = [row for row in history if row.public_score is not None]
    if not scored:
        return
    by_label: dict[str, list[RatingHistoryRow]] = {}
    for row in scored:
        by_label.setdefault(row.label, []).append(row)

    plt.figure(figsize=(9, 5.5))
    for label, rows in sorted(by_label.items()):
        rows = sorted(rows, key=lambda row: row.fetched_at_utc)
        times = [datetime.fromisoformat(row.fetched_at_utc) for row in rows]
        scores = [row.public_score for row in rows]
        plt.plot(times, scores, marker="o", markersize=3, label=label, linewidth=1.2)
    plt.xlabel("fetched at")
    plt.ylabel("live skill rating")
    plt.title("Kaggle live skill rating over time, by submission")
    plt.legend(fontsize=6, loc="best")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_leaderboard_rank_over_time(history: list[RatingHistoryRow], out_path: Path) -> None:
    """
    Save a line plot of our team's leaderboard rank over time.

    Unlike rating, rank is team-wide (one leaderboard row per team, not per
    submission) so every row from the same ``status`` run carries the same
    value - this dedupes by ``fetched_at_utc`` down to one point per run.
    Lower is better, so the y-axis is inverted: an upward line reads as
    "climbing the leaderboard", matching the reader's intuition.

    :param history: Rating-history rows to chart.
    :param out_path: PNG path to write.
    :return: None.
    """
    ranked = [row for row in history if row.leaderboard_rank is not None]
    if not ranked:
        return
    latest_per_run: dict[str, int] = {}
    for row in ranked:
        latest_per_run[row.fetched_at_utc] = row.leaderboard_rank
    points = sorted(latest_per_run.items())
    times = [datetime.fromisoformat(fetched_at) for fetched_at, _ in points]
    ranks = [rank for _, rank in points]
    plt.figure(figsize=(9, 5.5))
    plt.plot(times, ranks, marker="o", markersize=3, color=_BLUE, linewidth=1.2)
    plt.gca().invert_yaxis()
    plt.xlabel("fetched at")
    plt.ylabel("leaderboard rank")
    plt.title("Our leaderboard rank over time")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_win_rate_by_submission(summaries: list[SubmissionSummary], out_path: Path) -> None:
    """
    Save a bar chart comparing win rate across analyzed submissions.

    Our own local record from downloaded replays, not Kaggle's live rating -
    a different (denser, less noisy, but Kaggle-rating-independent) signal
    of which submission is actually performing best.

    :param summaries: Per-submission summaries to chart.
    :param out_path: PNG path to write.
    :return: None.
    """
    if not summaries:
        return
    ordered = sorted(summaries, key=lambda s: s.win_rate, reverse=True)
    labels = [s.label for s in ordered]
    values = [s.win_rate for s in ordered]
    colors = [_RED if value < 0.4 else _GREEN for value in values]
    plt.figure(figsize=(max(8.0, len(labels) * 0.6), 6))
    plt.bar(range(len(labels)), values, color=colors)
    plt.xticks(range(len(labels)), labels, fontsize=8, rotation=30, ha="right")
    plt.ylabel("win rate (decided games)")
    plt.ylim(0, 1)
    plt.title("Win rate by submission")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_average_ko_margin_by_submission(
    summaries: list[SubmissionSummary], out_path: Path
) -> None:
    """
    Save a bar chart comparing average KO margin across analyzed submissions.

    :param summaries: Per-submission summaries to chart.
    :param out_path: PNG path to write.
    :return: None.
    """
    if not summaries:
        return
    ordered = sorted(summaries, key=lambda s: s.average_ko_margin, reverse=True)
    labels = [s.label for s in ordered]
    values = [s.average_ko_margin for s in ordered]
    colors = [_GREEN if value >= 0 else _RED for value in values]
    plt.figure(figsize=(max(8.0, len(labels) * 0.6), 6))
    plt.bar(range(len(labels)), values, color=colors)
    plt.axhline(0, color="black", linewidth=0.6, linestyle="--")
    plt.xticks(range(len(labels)), labels, fontsize=8, rotation=30, ha="right")
    plt.ylabel("average KO margin (opponent KOs - our KOs)")
    plt.title("Average KO margin by submission")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_average_first_attack_turn_by_submission(
    summaries: list[SubmissionSummary], out_path: Path
) -> None:
    """
    Save a bar chart comparing deck speed (avg turn of first attack) across submissions.

    :param summaries: Per-submission summaries to chart.
    :param out_path: PNG path to write.
    :return: None.
    """
    timed = [s for s in summaries if s.average_first_attack_turn is not None]
    if not timed:
        return
    ordered = sorted(timed, key=lambda s: s.average_first_attack_turn)
    labels = [s.label for s in ordered]
    values = [s.average_first_attack_turn for s in ordered]
    plt.figure(figsize=(max(8.0, len(labels) * 0.6), 6))
    plt.bar(range(len(labels)), values, color=_ORANGE)
    plt.xticks(range(len(labels)), labels, fontsize=8, rotation=30, ha="right")
    plt.ylabel("average turn of first attack")
    plt.title("Deck speed by submission (lower = faster)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")
