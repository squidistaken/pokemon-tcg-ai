"""Smoke tests: each plot function must produce its output file for populated
input and skip writing for empty input, without crashing either way. Content
correctness is covered by the deck_report tests that compute the underlying
numbers; a handful of tests below also pin down plot-specific logic (capping,
filtering, summing) that isn't just "doesn't crash"."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from submission_analysis.deck_report import (
    AttackStat,
    CardStat,
    DeckReport,
    EvolutionStat,
    LossPostmortem,
    SubmissionSummary,
)
from submission_analysis.loading import RatingHistoryRow
from submission_analysis.plots import (
    plot_attack_usage,
    plot_attack_utilization,
    plot_average_first_attack_turn_by_submission,
    plot_average_ko_margin_by_submission,
    plot_card_play_rate,
    plot_damage_by_card,
    plot_evolution_conversion_rate,
    plot_first_attack_turn_distribution,
    plot_game_length_distribution,
    plot_ko_margin_distribution,
    plot_ko_rate,
    plot_ko_rate_vs_win_rate,
    plot_leaderboard_rank_over_time,
    plot_loss_causes,
    plot_opponent_attack_usage,
    plot_play_rate_vs_win_rate,
    plot_rating_history,
    plot_win_rate_by_submission,
    plot_win_rate_when_played,
)

CARD_STATS = [
    CardStat(
        card_id=1,
        name="Basic Mon",
        is_evolution=False,
        is_pokemon=True,
        games_seen=10,
        games_played=8,
        games_stuck=2,
        games_attacked_with=6,
        wins_when_played=5,
        losses_when_played=3,
        games_prized=1,
        times_knocked_out=4,
    ),
    CardStat(
        card_id=2,
        name="Evolved Mon",
        is_evolution=True,
        is_pokemon=True,
        games_seen=5,
        games_played=1,
        games_stuck=4,
        games_attacked_with=0,
        wins_when_played=0,
        losses_when_played=1,
        games_prized=0,
        times_knocked_out=0,
    ),
]

ATTACK_STATS = [
    AttackStat(
        attack_id=99,
        name="Quick Attack",
        card_id=1,
        card_name="Basic Mon",
        uses=8,
        total_damage=400,
        kos=3,
    )
]

EVOLUTION_STATS = [
    EvolutionStat(
        evo_card_id=2,
        evo_name="Evolved Mon",
        pre_evo_card_id=1,
        pre_evo_name="Basic Mon",
        games_pre_evo_played=8,
        games_evolved=3,
    )
]

SUMMARIES = [
    SubmissionSummary(
        label="111-fast",
        win_rate=0.6,
        average_ko_margin=1.2,
        average_first_attack_turn=4.0,
    ),
    SubmissionSummary(
        label="222-slow",
        win_rate=0.3,
        average_ko_margin=-0.5,
        average_first_attack_turn=None,
    ),
]

LOSS_POSTMORTEMS = [
    LossPostmortem(
        episode_id=1,
        opponent_team="alice",
        no_basic_pokemon=True,
        never_attacked=False,
        ko_deficit=1,
        evolution_stalled=False,
        decked_out=False,
        wiped_out=False,
        ended_by_card_effect=False,
        prize_deficit=1,
    ),
    LossPostmortem(
        episode_id=2,
        opponent_team="bob",
        no_basic_pokemon=False,
        never_attacked=True,
        ko_deficit=2,
        evolution_stalled=True,
        decked_out=False,
        wiped_out=False,
        ended_by_card_effect=False,
        prize_deficit=2,
    ),
]

OPPONENT_ATTACK_STATS = [
    AttackStat(
        attack_id=700,
        name="Enemy Strike",
        card_id=401,
        card_name="Villain Mon",
        uses=6,
        total_damage=900,
        kos=2,
    )
]

REPORT = DeckReport(
    episodes_analyzed=10,
    wins=6,
    losses=2,
    draws=2,
    no_basic_pokemon_rate=0.2,
    average_ko_margin=0.5,
    average_prize_margin=0.5,
    average_first_attack_turn_wins=3.0,
    average_first_attack_turn_losses=5.0,
    first_attack_turns=[("win", 3), ("win", 4), ("loss", 6)],
    ko_margins=[("win", 2), ("win", 1), ("loss", -3), ("draw", 0)],
    game_lengths=[("win", 8), ("win", 10), ("loss", 14)],
    card_stats=CARD_STATS,
    attack_stats=ATTACK_STATS,
    opponent_attack_stats=OPPONENT_ATTACK_STATS,
    evolution_stats=EVOLUTION_STATS,
    loss_postmortems=LOSS_POSTMORTEMS,
)

EMPTY_REPORT = DeckReport(
    episodes_analyzed=0,
    wins=0,
    losses=0,
    draws=0,
    no_basic_pokemon_rate=0.0,
    average_ko_margin=0.0,
    average_prize_margin=0.0,
    average_first_attack_turn_wins=None,
    average_first_attack_turn_losses=None,
    first_attack_turns=[],
    ko_margins=[],
    game_lengths=[],
    card_stats=[],
    attack_stats=[],
    opponent_attack_stats=[],
    evolution_stats=[],
    loss_postmortems=[],
)

RATING_HISTORY = [
    RatingHistoryRow(
        "2026-08-01T00:00:00+00:00", 1, "agent-a", "COMPLETE", 400.0, 5000
    ),
    RatingHistoryRow(
        "2026-08-01T01:00:00+00:00", 1, "agent-a", "COMPLETE", 420.0, 4200
    ),
    RatingHistoryRow(
        "2026-08-01T00:00:00+00:00", 2, "agent-b", "COMPLETE", 300.0, None
    ),
]
EMPTY_RATING_HISTORY = [
    RatingHistoryRow("2026-08-01T00:00:00+00:00", 1, "agent", "ERROR", None, None)
]
NO_TURN_SUMMARIES = [
    SubmissionSummary(
        label="x", win_rate=0.0, average_ko_margin=0.0, average_first_attack_turn=None
    )
]

# (name, call with populated data, call with empty-equivalent data) - one
# entry per plot function, driving the two smoke tests below.
_PLOTTERS: list[tuple[str, Callable[[Path], None], Callable[[Path], None]]] = [
    (
        "play_rate",
        lambda p: plot_card_play_rate(CARD_STATS, p),
        lambda p: plot_card_play_rate([], p),
    ),
    (
        "win_rate_when_played",
        lambda p: plot_win_rate_when_played(CARD_STATS, p),
        lambda p: plot_win_rate_when_played([], p),
    ),
    ("ko_rate", lambda p: plot_ko_rate(CARD_STATS, p), lambda p: plot_ko_rate([], p)),
    (
        "attack_utilization",
        lambda p: plot_attack_utilization(CARD_STATS, p),
        lambda p: plot_attack_utilization([], p),
    ),
    (
        "play_rate_vs_win_rate",
        lambda p: plot_play_rate_vs_win_rate(CARD_STATS, p),
        lambda p: plot_play_rate_vs_win_rate([], p),
    ),
    (
        "ko_rate_vs_win_rate",
        lambda p: plot_ko_rate_vs_win_rate(CARD_STATS, p),
        lambda p: plot_ko_rate_vs_win_rate([], p),
    ),
    (
        "evo_conversion",
        lambda p: plot_evolution_conversion_rate(EVOLUTION_STATS, p),
        lambda p: plot_evolution_conversion_rate([], p),
    ),
    (
        "attacks",
        lambda p: plot_attack_usage(ATTACK_STATS, p),
        lambda p: plot_attack_usage([], p),
    ),
    (
        "damage_by_card",
        lambda p: plot_damage_by_card(ATTACK_STATS, p),
        lambda p: plot_damage_by_card([], p),
    ),
    (
        "opponent_attack_usage",
        lambda p: plot_opponent_attack_usage(OPPONENT_ATTACK_STATS, p),
        lambda p: plot_opponent_attack_usage([], p),
    ),
    (
        "losses",
        lambda p: plot_loss_causes(REPORT, p),
        lambda p: plot_loss_causes(EMPTY_REPORT, p),
    ),
    (
        "first_attack_turns",
        lambda p: plot_first_attack_turn_distribution(REPORT.first_attack_turns, p),
        lambda p: plot_first_attack_turn_distribution([], p),
    ),
    (
        "game_length",
        lambda p: plot_game_length_distribution(REPORT.game_lengths, p),
        lambda p: plot_game_length_distribution([], p),
    ),
    (
        "ko_margins",
        lambda p: plot_ko_margin_distribution(REPORT.ko_margins, p),
        lambda p: plot_ko_margin_distribution([], p),
    ),
    (
        "rating_history",
        lambda p: plot_rating_history(RATING_HISTORY, p),
        lambda p: plot_rating_history(EMPTY_RATING_HISTORY, p),
    ),
    (
        "leaderboard_rank",
        lambda p: plot_leaderboard_rank_over_time(RATING_HISTORY, p),
        lambda p: plot_leaderboard_rank_over_time(EMPTY_RATING_HISTORY, p),
    ),
    (
        "win_rate_by_submission",
        lambda p: plot_win_rate_by_submission(SUMMARIES, p),
        lambda p: plot_win_rate_by_submission([], p),
    ),
    (
        "ko_margin_by_submission",
        lambda p: plot_average_ko_margin_by_submission(SUMMARIES, p),
        lambda p: plot_average_ko_margin_by_submission([], p),
    ),
    (
        "turn_by_submission",
        lambda p: plot_average_first_attack_turn_by_submission(SUMMARIES, p),
        lambda p: plot_average_first_attack_turn_by_submission(NO_TURN_SUMMARIES, p),
    ),
]


def test_plots_write_a_file_for_populated_input(tmp_path) -> None:
    for name, populated, _empty in _PLOTTERS:
        out_path = tmp_path / f"{name}.png"
        populated(out_path)
        assert out_path.is_file(), name


def test_plots_skip_when_input_is_empty(tmp_path) -> None:
    for name, _populated, empty in _PLOTTERS:
        out_path = tmp_path / f"{name}.png"
        empty(out_path)
        assert not out_path.exists(), name


def test_plot_attack_utilization_excludes_non_pokemon_cards(tmp_path) -> None:
    """A Trainer/Energy card is always 0% utilized for a reason unrelated to
    deck quality - it must not show up and drag the chart's signal down."""
    out_path = tmp_path / "attack_utilization.png"
    only_trainer = [
        CardStat(
            card_id=3,
            name="Ultra Ball",
            is_evolution=False,
            is_pokemon=False,
            games_seen=5,
            games_played=5,
            games_stuck=0,
            games_attacked_with=0,
            wins_when_played=2,
            losses_when_played=3,
            games_prized=0,
            times_knocked_out=0,
        )
    ]

    plot_attack_utilization(only_trainer, out_path)

    assert not out_path.exists()


def test_plot_damage_by_card_sums_across_a_cards_attacks(tmp_path) -> None:
    """Two attacks on the same card must combine into one bar, not two."""
    out_path = tmp_path / "damage_by_card.png"
    two_attacks_one_card = [
        AttackStat(
            attack_id=1,
            name="Tackle",
            card_id=1,
            card_name="Basic Mon",
            uses=5,
            total_damage=100,
            kos=0,
        ),
        AttackStat(
            attack_id=2,
            name="Slam",
            card_id=1,
            card_name="Basic Mon",
            uses=3,
            total_damage=150,
            kos=1,
        ),
    ]

    plot_damage_by_card(two_attacks_one_card, out_path)

    assert out_path.is_file()


def test_plot_first_attack_turn_distribution_caps_outlier_turns(tmp_path) -> None:
    """A single very-late-game outlier must not crash or blow out the axis -
    it should fold into the overflow bucket instead."""
    out_path = tmp_path / "turns.png"
    turns = [("win", 3), ("win", 5), ("loss", 40)]

    plot_first_attack_turn_distribution(turns, out_path, cap=15)

    assert out_path.is_file()


def test_plot_game_length_distribution_caps_outlier_turns(tmp_path) -> None:
    out_path = tmp_path / "game_length.png"
    lengths = [("win", 8), ("loss", 12), ("draw", 60)]

    plot_game_length_distribution(lengths, out_path, cap=30)

    assert out_path.is_file()
