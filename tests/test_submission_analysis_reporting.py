"""save_report is the only function here with a real contract to assert on
(a written file, in the right format); report_* just print Rich tables, so
they're exercised once via report_all to catch a hard crash, not per-function."""

from __future__ import annotations

from submission_analysis.deck_report import (
    AttackStat,
    CardStat,
    DeckReport,
    EvolutionStat,
    LossPostmortem,
)
from submission_analysis.reporting import report_all, report_overview, save_report

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
    first_attack_turns=[("win", 3), ("loss", 5)],
    ko_margins=[("win", 2), ("loss", -1)],
    game_lengths=[("win", 8), ("loss", 12)],
    card_stats=[
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
            name="Never Drawn Tech",
            is_evolution=False,
            is_pokemon=False,
            games_seen=0,
            games_played=0,
            games_stuck=0,
            games_attacked_with=0,
            wins_when_played=0,
            losses_when_played=0,
            games_prized=0,
            times_knocked_out=0,
        ),
    ],
    attack_stats=[
        AttackStat(
            attack_id=99,
            name="Quick Attack",
            card_id=1,
            card_name="Basic Mon",
            uses=8,
            total_damage=400,
            kos=3,
        )
    ],
    opponent_attack_stats=[
        AttackStat(
            attack_id=700,
            name="Enemy Strike",
            card_id=401,
            card_name="Villain Mon",
            uses=6,
            total_damage=900,
            kos=2,
        )
    ],
    evolution_stats=[
        EvolutionStat(
            evo_card_id=2,
            evo_name="Evolved Mon",
            pre_evo_card_id=1,
            pre_evo_name="Basic Mon",
            games_pre_evo_played=8,
            games_evolved=3,
        )
    ],
    loss_postmortems=[
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
        )
    ],
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


def test_report_all_runs_on_populated_and_empty_reports() -> None:
    report_all(REPORT)
    report_all(EMPTY_REPORT)


def test_save_report_writes_text(tmp_path) -> None:
    report_overview(REPORT)
    out_path = tmp_path / "report.txt"

    save_report(out_path)

    assert out_path.is_file()
    assert out_path.read_text(encoding="utf-8")
