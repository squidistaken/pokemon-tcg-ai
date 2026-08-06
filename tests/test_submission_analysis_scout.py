from __future__ import annotations

from submission_analysis.loading import CardIndex
from submission_analysis.scout import (
    ScoutedTeam,
    card_frequency,
    gap_analysis,
    plot_card_frequency,
    plot_gap_analysis,
)

CARD_INDEX = CardIndex(
    cards={}, attacks={}
)  # names fall back to "card <id>"; fine for plots


def _team(team_id: int, cards: set[int]) -> ScoutedTeam:
    return ScoutedTeam(
        team_id=team_id,
        team_name=f"team-{team_id}",
        score=1000.0,
        submission_id=team_id * 10,
        episodes_sampled=3,
        cards=frozenset(cards),
    )


def test_card_frequency_counts_teams_not_occurrences() -> None:
    teams = [
        _team(1, {100, 200}),
        _team(2, {100}),
        _team(3, {100, 200, 300}),
    ]

    frequency = card_frequency(teams)

    assert frequency[100] == 3
    assert frequency[200] == 2
    assert frequency[300] == 1


def test_card_frequency_empty_without_teams() -> None:
    assert card_frequency([]) == {}


def test_gap_analysis_excludes_cards_we_already_run() -> None:
    teams = [_team(1, {100, 200}), _team(2, {100, 200})]

    gaps = gap_analysis(teams, our_deck=[100], min_teams=1)

    assert gaps == [(200, 2)]


def test_gap_analysis_respects_min_teams_threshold() -> None:
    teams = [_team(1, {100, 200}), _team(2, {100})]

    gaps = gap_analysis(teams, our_deck=[], min_teams=2)

    assert gaps == [(100, 2)]


def test_gap_analysis_sorted_by_frequency_descending() -> None:
    teams = [_team(1, {100, 200}), _team(2, {200}), _team(3, {200})]

    gaps = gap_analysis(teams, our_deck=[], min_teams=1)

    assert gaps == [(200, 3), (100, 1)]


def test_gap_analysis_empty_when_our_deck_covers_everything() -> None:
    teams = [_team(1, {100})]

    gaps = gap_analysis(teams, our_deck=[100], min_teams=1)

    assert gaps == []


def test_plot_card_frequency_writes_a_file(tmp_path) -> None:
    teams = [_team(1, {100, 200}), _team(2, {100})]
    out_path = tmp_path / "card_frequency.png"

    plot_card_frequency(teams, CARD_INDEX, out_path)

    assert out_path.is_file()


def test_plot_card_frequency_skips_when_no_teams(tmp_path) -> None:
    out_path = tmp_path / "card_frequency.png"

    plot_card_frequency([], CARD_INDEX, out_path)

    assert not out_path.exists()


def test_plot_gap_analysis_writes_a_file(tmp_path) -> None:
    teams = [_team(1, {100, 200}), _team(2, {100, 200})]
    out_path = tmp_path / "gap_analysis.png"

    plot_gap_analysis(teams, [100], CARD_INDEX, out_path, min_teams=1)

    assert out_path.is_file()


def test_plot_gap_analysis_skips_when_no_gaps(tmp_path) -> None:
    teams = [_team(1, {100})]
    out_path = tmp_path / "gap_analysis.png"

    plot_gap_analysis(teams, [100], CARD_INDEX, out_path, min_teams=1)

    assert not out_path.exists()
