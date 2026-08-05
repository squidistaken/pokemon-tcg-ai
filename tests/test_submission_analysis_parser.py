"""Tests for submission_analysis.parser against the real replay JSON schema.

Log/event shapes here are copied from real downloaded replays (see
submission_analysis/parser.py's module docstring for where that schema comes
from), not invented, so a schema drift in the engine would show up here.
"""

from __future__ import annotations

from submission_analysis.parser import AttackUsage, ParsedEpisode, parse_replay

OUR_INDEX = 1
OTHER_INDEX = 0


def _placeholder_agent() -> dict:
    """A minimal opponent-side step entry; parse_replay never reads it."""
    return {"action": [], "observation": {}, "status": "INACTIVE"}


def _our_step(
    logs: list[dict],
    *,
    hand: list[dict] | None = None,
    turn: int | None = None,
    status: str = "ACTIVE",
) -> list[dict]:
    """Build one [opponent, ours] step pair with our own observation populated."""
    current = {"players": [{}, {}]}
    if hand is not None:
        current["players"][OUR_INDEX] = {"hand": hand}
    if turn is not None:
        current["turn"] = turn
    step = [_placeholder_agent(), _placeholder_agent()]
    step[OUR_INDEX] = {
        "action": [0],
        "status": status,
        "observation": {"logs": logs, "current": current},
    }
    return step


def _replay(steps: list[list[dict]]) -> dict:
    return {"steps": steps}


def _parse(steps: list[list[dict]], **kwargs) -> ParsedEpisode:
    defaults = {
        "episode_id": 1,
        "our_index": OUR_INDEX,
        "result": "win",
        "opponent_team": "opponent",
    }
    defaults.update(kwargs)
    return parse_replay(_replay(steps), **defaults)


def test_parse_replay_records_has_basic_pokemon_for_our_side_only() -> None:
    steps = [
        _our_step(
            [
                {"type": 1, "playerIndex": OUR_INDEX, "hasBasicPokemon": True},
                {"type": 1, "playerIndex": OTHER_INDEX, "hasBasicPokemon": False},
            ]
        )
    ]

    episode = _parse(steps)

    assert episode.had_basic_pokemon is True


def test_parse_replay_no_basic_pokemon_in_opener() -> None:
    steps = [_our_step([{"type": 1, "playerIndex": OUR_INDEX, "hasBasicPokemon": False}])]

    episode = _parse(steps)

    assert episode.had_basic_pokemon is False


def test_parse_replay_tracks_played_cards() -> None:
    steps = [
        _our_step([{"type": 10, "playerIndex": OUR_INDEX, "cardId": 434, "serial": 71}]),
        _our_step([{"type": 10, "playerIndex": OTHER_INDEX, "cardId": 999, "serial": 1}]),
    ]

    episode = _parse(steps)

    assert episode.cards_played == {434}
    assert episode.cards_seen == {434}


def test_parse_replay_tracks_attached_energy_and_tool_cards_as_played() -> None:
    """Energy/Tool cards are ATTACH events, not PLAY - both count as played."""
    steps = [
        _our_step(
            [
                {
                    "type": 11,
                    "playerIndex": OUR_INDEX,
                    "cardId": 500,
                    "serial": 3,
                    "cardIdTarget": 354,
                    "serialTarget": 11,
                },
                {"type": 11, "playerIndex": OTHER_INDEX, "cardId": 999, "serial": 1},
            ]
        )
    ]

    episode = _parse(steps)

    assert episode.cards_played == {500}
    assert episode.cards_seen == {500}


def test_parse_replay_tracks_hand_snapshots_as_cards_seen() -> None:
    steps = [
        _our_step([], hand=[{"id": 100, "serial": 1}, {"id": 200, "serial": 2}]),
    ]

    episode = _parse(steps)

    assert episode.cards_seen == {100, 200}
    assert episode.cards_played == set()


def test_parse_replay_evolution_records_both_sides() -> None:
    steps = [
        _our_step(
            [
                {
                    "type": 12,
                    "playerIndex": OUR_INDEX,
                    "cardId": 354,
                    "serial": 11,
                    "cardIdTarget": 353,
                    "serialTarget": 8,
                }
            ]
        )
    ]

    episode = _parse(steps)

    assert episode.evolutions_made == {354}
    assert episode.pre_evolutions_available == {353}
    assert episode.cards_played == {354}


def test_parse_replay_attributes_damage_and_ko_to_the_attack_that_caused_them() -> None:
    """Mirrors a real batch: ATTACK, then HP_CHANGE, then the KO'd MOVE_CARD,
    all for the same decision point."""
    steps = [
        _our_step(
            [
                {"type": 15, "playerIndex": OUR_INDEX, "cardId": 354, "serial": 11, "attackId": 490},
                {
                    "type": 16,
                    "playerIndex": OTHER_INDEX,
                    "cardId": 401,
                    "serial": 67,
                    "value": -560,
                    "putDamageCounter": False,
                },
                {
                    "type": 6,
                    "playerIndex": OTHER_INDEX,
                    "cardId": 401,
                    "serial": 67,
                    "fromArea": 4,
                    "toArea": 3,
                },
            ]
        )
    ]

    episode = _parse(steps)

    assert episode.attacks == [
        AttackUsage(attack_id=490, card_id=354, damage=560, knocked_out=True)
    ]
    assert episode.opponent_kos == 1
    assert episode.our_kos == 0


def test_parse_replay_attributes_damage_and_ko_of_our_pokemon_to_the_opponents_attack() -> None:
    """Mirrors test_..._attributes_damage_and_ko_to_the_attack_that_caused_them,
    but for the opponent attacking us - the data needed to know which enemy
    cards/attacks are actually dangerous to our deck."""
    steps = [
        _our_step(
            [
                {
                    "type": 15,
                    "playerIndex": OTHER_INDEX,
                    "cardId": 401,
                    "serial": 67,
                    "attackId": 700,
                },
                {
                    "type": 16,
                    "playerIndex": OUR_INDEX,
                    "cardId": 354,
                    "serial": 11,
                    "value": -300,
                    "putDamageCounter": False,
                },
                {
                    "type": 6,
                    "playerIndex": OUR_INDEX,
                    "cardId": 354,
                    "serial": 11,
                    "fromArea": 4,
                    "toArea": 3,
                },
            ]
        )
    ]

    episode = _parse(steps)

    assert episode.opponent_attacks == [
        AttackUsage(attack_id=700, card_id=401, damage=300, knocked_out=True)
    ]
    assert episode.our_kos == 1
    assert episode.opponent_kos == 0


def test_parse_replay_does_not_attribute_opponent_damage_across_step_boundaries() -> None:
    steps = [
        _our_step(
            [{"type": 15, "playerIndex": OTHER_INDEX, "cardId": 401, "serial": 67, "attackId": 700}]
        ),
        _our_step(
            [{"type": 16, "playerIndex": OUR_INDEX, "cardId": 354, "serial": 11, "value": -100}]
        ),
    ]

    episode = _parse(steps)

    assert episode.opponent_attacks == [AttackUsage(attack_id=700, card_id=401, damage=0)]


def test_parse_replay_final_turn_tracks_the_last_observed_turn() -> None:
    steps = [
        _our_step([], turn=1),
        _our_step([], turn=5),
        _our_step([], turn=12),
    ]

    episode = _parse(steps)

    assert episode.final_turn == 12


def test_parse_replay_final_turn_is_none_without_any_turn_data() -> None:
    episode = _parse([_our_step([])])

    assert episode.final_turn is None


def test_parse_replay_ignores_logs_echoed_by_inactive_steps() -> None:
    """kaggle_environments repeats the last-delivered batch verbatim on every
    step where it's not our turn (status INACTIVE), instead of an empty
    delta - those echoes must not be recounted as new events."""
    ko_log = [
        {
            "type": 6,
            "playerIndex": OTHER_INDEX,
            "cardId": 401,
            "serial": 67,
            "fromArea": 4,
            "toArea": 3,
        }
    ]
    steps = [
        _our_step(ko_log, status="ACTIVE"),
        *[_our_step(ko_log, status="INACTIVE") for _ in range(10)],
    ]

    episode = _parse(steps)

    assert episode.opponent_kos == 1


def test_parse_replay_ignores_a_done_step_that_echoes_the_last_active_batch() -> None:
    ko_log = [
        {
            "type": 6,
            "playerIndex": OTHER_INDEX,
            "cardId": 401,
            "serial": 67,
            "fromArea": 4,
            "toArea": 3,
        }
    ]
    steps = [
        _our_step(ko_log, status="ACTIVE"),
        _our_step(ko_log, status="DONE"),
    ]

    episode = _parse(steps)

    assert episode.opponent_kos == 1


def test_parse_replay_processes_genuinely_new_logs_on_a_done_step() -> None:
    first_ko = [
        {
            "type": 6,
            "playerIndex": OTHER_INDEX,
            "cardId": 401,
            "serial": 67,
            "fromArea": 4,
            "toArea": 3,
        }
    ]
    final_ko = [
        {
            "type": 6,
            "playerIndex": OTHER_INDEX,
            "cardId": 402,
            "serial": 68,
            "fromArea": 5,
            "toArea": 3,
        }
    ]
    steps = [
        _our_step(first_ko, status="ACTIVE"),
        *[_our_step(first_ko, status="INACTIVE") for _ in range(5)],
        _our_step(final_ko, status="DONE"),
    ]

    episode = _parse(steps)

    assert episode.opponent_kos == 2


def test_parse_replay_records_the_turn_of_our_first_attack() -> None:
    steps = [
        _our_step([], turn=3),
        _our_step(
            [{"type": 15, "playerIndex": OUR_INDEX, "cardId": 354, "serial": 11, "attackId": 490}],
            turn=5,
        ),
        _our_step(
            [{"type": 15, "playerIndex": OUR_INDEX, "cardId": 354, "serial": 11, "attackId": 490}],
            turn=7,
        ),
    ]

    episode = _parse(steps)

    assert episode.first_attack_turn == 5


def test_parse_replay_first_attack_turn_is_none_when_we_never_attack() -> None:
    steps = [_our_step([], turn=3)]

    episode = _parse(steps)

    assert episode.first_attack_turn is None


def test_parse_replay_does_not_attribute_damage_across_step_boundaries() -> None:
    """A pending attack from one decision point must not absorb HP_CHANGE
    entries that show up in a later, unrelated step."""
    steps = [
        _our_step(
            [{"type": 15, "playerIndex": OUR_INDEX, "cardId": 354, "serial": 11, "attackId": 490}]
        ),
        _our_step(
            [
                {
                    "type": 16,
                    "playerIndex": OTHER_INDEX,
                    "cardId": 401,
                    "serial": 67,
                    "value": -100,
                }
            ]
        ),
    ]

    episode = _parse(steps)

    assert episode.attacks == [AttackUsage(attack_id=490, card_id=354, damage=0)]


def test_parse_replay_own_pokemon_ko_counts_separately_from_opponents() -> None:
    steps = [
        _our_step(
            [
                {
                    "type": 6,
                    "playerIndex": OUR_INDEX,
                    "cardId": 414,
                    "serial": 75,
                    "fromArea": 4,
                    "toArea": 3,
                }
            ]
        )
    ]

    episode = _parse(steps)

    assert episode.our_kos == 1
    assert episode.opponent_kos == 0
    assert episode.our_pokemon_lost == {414}


def test_parse_replay_tracks_which_of_our_pokemon_are_knocked_out() -> None:
    """A second copy of the same card ID knocked out doesn't double the
    per-episode set - it stays comparable to cards_played (also per-episode)."""
    steps = [
        _our_step(
            [{"type": 6, "playerIndex": OUR_INDEX, "cardId": 414, "serial": 75, "fromArea": 4, "toArea": 3}]
        ),
        _our_step(
            [{"type": 6, "playerIndex": OUR_INDEX, "cardId": 414, "serial": 12, "fromArea": 5, "toArea": 3}]
        ),
        _our_step(
            [{"type": 6, "playerIndex": OUR_INDEX, "cardId": 500, "serial": 3, "fromArea": 4, "toArea": 3}]
        ),
    ]

    episode = _parse(steps)

    assert episode.our_pokemon_lost == {414, 500}
    assert episode.our_kos == 3


def test_parse_replay_our_kos_cards_preserves_multiplicity() -> None:
    """Unlike our_pokemon_lost's per-episode set, our_kos_cards keeps one
    entry per KO - needed to sum prize value correctly when the same card
    is knocked out more than once."""
    steps = [
        _our_step(
            [{"type": 6, "playerIndex": OUR_INDEX, "cardId": 414, "serial": 75, "fromArea": 4, "toArea": 3}]
        ),
        _our_step(
            [{"type": 6, "playerIndex": OUR_INDEX, "cardId": 414, "serial": 12, "fromArea": 5, "toArea": 3}]
        ),
    ]

    episode = _parse(steps)

    assert episode.our_kos_cards == [414, 414]


def test_parse_replay_tracks_opponent_kos_cards() -> None:
    steps = [
        _our_step(
            [{"type": 6, "playerIndex": OTHER_INDEX, "cardId": 401, "serial": 67, "fromArea": 4, "toArea": 3}]
        )
    ]

    episode = _parse(steps)

    assert episode.opponent_kos_cards == [401]
    assert episode.our_kos_cards == []


def test_parse_replay_records_game_end_reason() -> None:
    steps = [_our_step([{"type": 23, "result": OUR_INDEX, "reason": 2}])]

    episode = _parse(steps)

    assert episode.game_end_reason == 2


def test_parse_replay_game_end_reason_is_none_without_a_result_log() -> None:
    episode = _parse([_our_step([])])

    assert episode.game_end_reason is None


def test_parse_replay_prize_reveal_only_when_ours_and_visible() -> None:
    steps = [
        _our_step(
            [
                {
                    "type": 6,
                    "playerIndex": OUR_INDEX,
                    "cardId": 119,
                    "serial": 14,
                    "fromArea": 6,
                    "toArea": 2,
                },
                # Opponent's own prize reveal is anonymous (type 7, no cardId) - ignored.
                {"type": 7, "playerIndex": OTHER_INDEX, "fromArea": 6, "toArea": 2},
            ]
        )
    ]

    episode = _parse(steps)

    assert episode.prized_cards == {119}


def test_parse_replay_ignores_steps_shorter_than_our_index() -> None:
    """A malformed/truncated step must not crash the parser."""
    episode = _parse([[_placeholder_agent()]])

    assert episode.cards_seen == set()


def test_parse_replay_carries_through_identity_fields() -> None:
    episode = _parse([], episode_id=42, result="loss", opponent_team="Kumajiro")

    assert episode.episode_id == 42
    assert episode.result == "loss"
    assert episode.opponent_team == "Kumajiro"
