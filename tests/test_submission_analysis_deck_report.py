from __future__ import annotations

from cg.api import Attack, CardData, CardType
from submission_analysis.deck_report import (
    CardStat,
    EvolutionStat,
    build_report,
    generate_recommendations,
)
from submission_analysis.loading import CardIndex
from submission_analysis.parser import AttackUsage, GameEndReason, ParsedEpisode


def _card(
    card_id: int,
    name: str,
    *,
    basic: bool = True,
    stage1: bool = False,
    evolves_from: str | None = None,
    card_type: CardType = CardType.POKEMON,
    ex: bool = False,
    mega_ex: bool = False,
) -> CardData:
    return CardData(
        cardId=card_id,
        name=name,
        cardType=card_type,
        retreatCost=1,
        hp=100,
        weakness=None,
        resistance=None,
        energyType=0,
        basic=basic,
        stage1=stage1,
        stage2=False,
        ex=ex,
        megaEx=mega_ex,
        tera=False,
        aceSpec=False,
        evolvesFrom=evolves_from,
        skills=[],
        attacks=[],
    )


def _attack(attack_id: int, name: str) -> Attack:
    return Attack(attackId=attack_id, name=name, text="", damage=0, energies=[])


CARD_INDEX = CardIndex(
    cards={
        100: _card(100, "Basic Mon"),
        200: _card(200, "Evolved Mon", basic=False, stage1=True, evolves_from="Basic Mon"),
        300: _card(300, "Ultra Ball", card_type=CardType.ITEM),
        400: _card(400, "Basic ex", ex=True),
        500: _card(500, "Mega ex", ex=True, mega_ex=True),
    },
    attacks={999: _attack(999, "Quick Attack"), 700: _attack(700, "Enemy Strike")},
)


def _episode(
    *,
    episode_id: int,
    result: str,
    cards_seen: set[int] | None = None,
    cards_played: set[int] | None = None,
    evolutions_made: set[int] | None = None,
    attacks: list[AttackUsage] | None = None,
    opponent_attacks: list[AttackUsage] | None = None,
    our_kos: int = 0,
    opponent_kos: int = 0,
    our_pokemon_lost: set[int] | None = None,
    our_kos_cards: list[int] | None = None,
    opponent_kos_cards: list[int] | None = None,
    prized_cards: set[int] | None = None,
    had_basic_pokemon: bool | None = None,
    first_attack_turn: int | None = None,
    final_turn: int | None = None,
    game_end_reason: int | None = None,
) -> ParsedEpisode:
    return ParsedEpisode(
        episode_id=episode_id,
        result=result,
        opponent_team="opponent",
        had_basic_pokemon=had_basic_pokemon,
        cards_seen=cards_seen or set(),
        cards_played=cards_played or set(),
        evolutions_made=evolutions_made or set(),
        attacks=attacks or [],
        opponent_attacks=opponent_attacks or [],
        our_kos=our_kos,
        opponent_kos=opponent_kos,
        our_pokemon_lost=our_pokemon_lost or set(),
        our_kos_cards=our_kos_cards or [],
        opponent_kos_cards=opponent_kos_cards or [],
        prized_cards=prized_cards or set(),
        first_attack_turn=first_attack_turn,
        final_turn=final_turn,
        game_end_reason=game_end_reason,
    )


def test_build_report_overall_win_loss_draw_counts() -> None:
    episodes = [
        _episode(episode_id=1, result="win"),
        _episode(episode_id=2, result="loss"),
        _episode(episode_id=3, result="draw"),
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    assert (report.wins, report.losses, report.draws) == (1, 1, 1)
    assert report.episodes_analyzed == 3


def test_build_report_no_basic_pokemon_rate_ignores_unknown_episodes() -> None:
    episodes = [
        _episode(episode_id=1, result="loss", had_basic_pokemon=False),
        _episode(episode_id=2, result="win", had_basic_pokemon=True),
        _episode(episode_id=3, result="win", had_basic_pokemon=None),
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    assert report.no_basic_pokemon_rate == 0.5  # 1 of the 2 episodes with known info


def test_build_report_average_ko_margin() -> None:
    episodes = [
        _episode(episode_id=1, result="win", our_kos=1, opponent_kos=3),
        _episode(episode_id=2, result="loss", our_kos=4, opponent_kos=1),
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    # (3-1) + (1-4) = -1, over 2 episodes = -0.5
    assert report.average_ko_margin == -0.5


def test_build_report_average_prize_margin_weights_ex_pokemon() -> None:
    """A single ex Pokemon knocked out is worth 2 raw KOs' worth of prizes,
    so a submission can win the KO count and still lose the prize count."""
    episodes = [
        # We KO'd two Basics (2 prizes); the opponent KO'd our one ex (2
        # prizes) - even KO count (2-1... wait, opponent_kos=2, our_kos=1
        # cards) but even prize value.
        _episode(
            episode_id=1,
            result="loss",
            our_kos=1,
            opponent_kos=2,
            our_kos_cards=[400],  # we lost an ex: 2 prizes to the opponent
            opponent_kos_cards=[100, 100],  # we KO'd two basics: 2 prizes to us
        )
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    # KO count favors us (opponent_kos=2 > our_kos=1), but prize value is even.
    assert report.average_ko_margin == 1.0
    assert report.average_prize_margin == 0.0


def test_build_report_first_attack_turn_split_by_result() -> None:
    episodes = [
        _episode(episode_id=1, result="win", first_attack_turn=3),
        _episode(episode_id=2, result="win", first_attack_turn=5),
        _episode(episode_id=3, result="loss", first_attack_turn=8),
        _episode(episode_id=4, result="loss", first_attack_turn=None),  # never attacked
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    assert report.average_first_attack_turn_wins == 4.0
    assert report.average_first_attack_turn_losses == 8.0


def test_build_report_first_attack_turn_none_when_never_attacked() -> None:
    episodes = [_episode(episode_id=1, result="loss", first_attack_turn=None)]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    assert report.average_first_attack_turn_wins is None
    assert report.average_first_attack_turn_losses is None


def test_build_report_first_attack_turns_excludes_episodes_that_never_attacked() -> None:
    episodes = [
        _episode(episode_id=1, result="win", first_attack_turn=3),
        _episode(episode_id=2, result="loss", first_attack_turn=None),
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    assert report.first_attack_turns == [("win", 3)]


def test_build_report_ko_margins_include_every_episode_regardless_of_attacks() -> None:
    episodes = [
        _episode(episode_id=1, result="win", our_kos=1, opponent_kos=3),
        _episode(episode_id=2, result="loss", our_kos=4, opponent_kos=1),
        _episode(episode_id=3, result="draw", our_kos=2, opponent_kos=2),
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    assert report.ko_margins == [("win", 2), ("loss", -3), ("draw", 0)]


def test_deck_report_win_rate_ignores_draws() -> None:
    episodes = [
        _episode(episode_id=1, result="win"),
        _episode(episode_id=2, result="win"),
        _episode(episode_id=3, result="loss"),
        _episode(episode_id=4, result="draw"),
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    assert report.win_rate == 2 / 3


def test_deck_report_win_rate_is_zero_with_no_decided_games() -> None:
    report = build_report([], decklist=None, card_index=CARD_INDEX)

    assert report.win_rate == 0.0


def test_deck_report_average_first_attack_turn_blends_wins_and_losses() -> None:
    episodes = [
        _episode(episode_id=1, result="win", first_attack_turn=2),
        _episode(episode_id=2, result="loss", first_attack_turn=8),
        _episode(episode_id=3, result="loss", first_attack_turn=None),  # excluded
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    assert report.average_first_attack_turn == 5.0


def test_deck_report_average_first_attack_turn_none_when_never_attacked() -> None:
    report = build_report(
        [_episode(episode_id=1, result="loss")], decklist=None, card_index=CARD_INDEX
    )

    assert report.average_first_attack_turn is None


def test_build_report_card_stats_track_stuck_and_win_rate() -> None:
    episodes = [
        _episode(episode_id=1, result="win", cards_seen={100}, cards_played={100}),
        _episode(episode_id=2, result="loss", cards_seen={100}, cards_played=set()),
        _episode(episode_id=3, result="loss", cards_seen={100}, cards_played={100}),
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    (stat,) = [s for s in report.card_stats if s.card_id == 100]
    assert stat == CardStat(
        card_id=100,
        name="Basic Mon",
        is_evolution=False,
        is_pokemon=True,
        games_seen=3,
        games_played=2,
        games_stuck=1,
        games_attacked_with=0,
        wins_when_played=1,
        losses_when_played=1,
        games_prized=0,
        times_knocked_out=0,
    )
    assert stat.play_rate == 2 / 3
    assert stat.win_rate_when_played == 0.5


def test_build_report_card_stats_track_times_knocked_out() -> None:
    episodes = [
        _episode(
            episode_id=1,
            result="loss",
            cards_seen={100},
            cards_played={100},
            our_pokemon_lost={100},
        ),
        _episode(
            episode_id=2,
            result="loss",
            cards_seen={100},
            cards_played={100},
            our_pokemon_lost={100},
        ),
        _episode(
            episode_id=3,
            result="win",
            cards_seen={100},
            cards_played={100},
        ),
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    (stat,) = [s for s in report.card_stats if s.card_id == 100]
    assert stat.times_knocked_out == 2
    assert stat.games_played == 3
    assert stat.ko_rate_when_played == 2 / 3


def test_build_report_card_stats_track_prize_rate() -> None:
    episodes = [
        _episode(
            episode_id=1,
            result="loss",
            cards_seen={100},
            cards_played={100},
            prized_cards={100},
        ),
        _episode(episode_id=2, result="win", cards_seen={100}, cards_played={100}),
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    (stat,) = [s for s in report.card_stats if s.card_id == 100]
    assert stat.games_prized == 1
    assert stat.prize_rate == 0.5


def test_build_report_evolution_stats_conversion_rate() -> None:
    episodes = [
        _episode(episode_id=1, result="win", cards_played={100}, evolutions_made={200}),
        _episode(episode_id=2, result="loss", cards_played={100}),  # played but never evolved
        _episode(episode_id=3, result="win", cards_played=set()),  # pre-evo never played
    ]

    report = build_report(episodes, decklist=[100, 200], card_index=CARD_INDEX)

    (stat,) = report.evolution_stats
    assert stat.evo_card_id == 200
    assert stat.pre_evo_card_id == 100
    assert stat.games_pre_evo_played == 2
    assert stat.games_evolved == 1
    assert stat.conversion_rate == 0.5


def test_build_report_evolution_stats_deduplicates_multi_copy_decklist_entries() -> None:
    """A decklist lists an id once per physical copy (e.g. 4 lines for a
    4-of) - a multi-copy evolution card must still produce one stat, not
    one per copy."""
    episodes = [
        _episode(episode_id=1, result="win", cards_played={100}, evolutions_made={200}),
    ]

    report = build_report(episodes, decklist=[100, 200, 200, 200, 200], card_index=CARD_INDEX)

    assert len(report.evolution_stats) == 1


def test_build_report_evolution_stats_empty_without_decklist() -> None:
    episodes = [_episode(episode_id=1, result="win", cards_played={100}, evolutions_made={200})]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    assert report.evolution_stats == []


def test_evolution_stat_conversion_rate_clips_at_one() -> None:
    """A stage-skip effect (e.g. Rare Candy) can evolve without the
    pre-evolution ever being separately "played" first, which would
    otherwise show as an impossible >100% conversion rate."""
    stat = EvolutionStat(
        evo_card_id=200,
        evo_name="Evolved Mon",
        pre_evo_card_id=100,
        pre_evo_name="Basic Mon",
        games_pre_evo_played=1,
        games_evolved=3,
    )

    assert stat.conversion_rate == 1.0


def test_build_report_decklist_surfaces_never_seen_cards() -> None:
    episodes = [_episode(episode_id=1, result="win", cards_seen={100}, cards_played={100})]

    report = build_report(episodes, decklist=[100, 200], card_index=CARD_INDEX)

    never_seen = next(stat for stat in report.card_stats if stat.card_id == 200)
    assert never_seen.games_seen == 0
    assert never_seen.games_played == 0
    assert never_seen.is_evolution is True  # 200 is a stage1 card


def test_build_report_attack_stats_aggregate_across_episodes() -> None:
    episodes = [
        _episode(
            episode_id=1,
            result="win",
            attacks=[AttackUsage(attack_id=999, card_id=100, damage=50, knocked_out=False)],
        ),
        _episode(
            episode_id=2,
            result="win",
            attacks=[AttackUsage(attack_id=999, card_id=100, damage=70, knocked_out=True)],
        ),
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    (stat,) = report.attack_stats
    assert stat.attack_id == 999
    assert stat.name == "Quick Attack"
    assert stat.uses == 2
    assert stat.total_damage == 120
    assert stat.average_damage == 60
    assert stat.kos == 1


def test_build_report_opponent_attack_stats_track_threats_to_us() -> None:
    episodes = [
        _episode(
            episode_id=1,
            result="loss",
            opponent_attacks=[
                AttackUsage(attack_id=700, card_id=401, damage=300, knocked_out=True)
            ],
        ),
        _episode(
            episode_id=2,
            result="loss",
            opponent_attacks=[
                AttackUsage(attack_id=700, card_id=401, damage=200, knocked_out=False)
            ],
        ),
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    (stat,) = report.opponent_attack_stats
    assert stat.attack_id == 700
    assert stat.name == "Enemy Strike"
    assert stat.uses == 2
    assert stat.total_damage == 500
    assert stat.kos == 1
    # Our own attack_stats must stay untouched by the opponent's.
    assert report.attack_stats == []


def test_build_report_game_lengths_reflect_final_turn_per_episode() -> None:
    episodes = [
        _episode(episode_id=1, result="win", final_turn=6),
        _episode(episode_id=2, result="loss", final_turn=20),
        _episode(episode_id=3, result="draw", final_turn=None),
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    assert report.game_lengths == [("win", 6), ("loss", 20)]


def test_build_report_card_stats_track_games_attacked_with() -> None:
    episodes = [
        _episode(
            episode_id=1,
            result="win",
            cards_seen={100},
            cards_played={100},
            attacks=[AttackUsage(attack_id=999, card_id=100)],
        ),
        _episode(
            episode_id=2,
            result="loss",
            cards_seen={100},
            cards_played={100},
        ),  # played, but knocked out before ever attacking
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    (stat,) = [s for s in report.card_stats if s.card_id == 100]
    assert stat.games_attacked_with == 1
    assert stat.games_played == 2
    assert stat.attack_utilization == 0.5


def test_card_stat_is_pokemon_distinguishes_trainer_cards() -> None:
    episodes = [
        _episode(episode_id=1, result="win", cards_seen={100, 300}, cards_played={100, 300})
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    pokemon = next(s for s in report.card_stats if s.card_id == 100)
    trainer = next(s for s in report.card_stats if s.card_id == 300)
    assert pokemon.is_pokemon is True
    assert trainer.is_pokemon is False


def test_build_report_loss_postmortems_only_cover_losses() -> None:
    episodes = [
        _episode(episode_id=1, result="win", had_basic_pokemon=False),
        _episode(
            episode_id=2,
            result="loss",
            had_basic_pokemon=False,
            our_kos=3,
            opponent_kos=1,
        ),
        _episode(
            episode_id=3,
            result="loss",
            cards_seen={200},
            cards_played=set(),
        ),
    ]

    report = build_report(episodes, decklist=[100, 200], card_index=CARD_INDEX)

    assert len(report.loss_postmortems) == 2
    first = next(p for p in report.loss_postmortems if p.episode_id == 2)
    assert first.no_basic_pokemon is True
    assert first.never_attacked is True
    assert first.ko_deficit == 2  # our_kos(3) - opponent_kos(1)

    second = next(p for p in report.loss_postmortems if p.episode_id == 3)
    assert second.evolution_stalled is True  # saw the evolution card, never played it


def test_loss_postmortem_unexplained_when_no_named_cause_applies() -> None:
    episodes = [
        _episode(
            episode_id=1,
            result="loss",
            had_basic_pokemon=True,
            attacks=[AttackUsage(attack_id=999, card_id=100)],
            our_kos=1,
            opponent_kos=2,  # ko_deficit = -1: we won the trade and still lost
        )
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    (postmortem,) = report.loss_postmortems
    assert postmortem.unexplained is True


def test_loss_postmortem_lost_prize_trade_despite_winning_the_ko_count() -> None:
    """Losing a single ex Pokemon (2 prizes) outweighs KO-ing two Basics (2
    prizes)... this one tips it further: we KO'd only one Basic but lost our
    ex, so we won the KO count but lost on prize value."""
    episodes = [
        _episode(
            episode_id=1,
            result="loss",
            had_basic_pokemon=True,
            attacks=[AttackUsage(attack_id=999, card_id=100)],
            our_kos=1,
            opponent_kos=1,  # ko_deficit = 0: doesn't explain the loss alone
            our_kos_cards=[400],  # lost an ex: gave up 2 prizes
            opponent_kos_cards=[100],  # KO'd a Basic: took 1 prize
        )
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    (postmortem,) = report.loss_postmortems
    assert postmortem.ko_deficit == 0
    assert postmortem.prize_deficit == 1
    assert postmortem.lost_prize_trade is True
    assert postmortem.unexplained is False


def test_loss_postmortem_decked_out() -> None:
    episodes = [
        _episode(episode_id=1, result="loss", game_end_reason=GameEndReason.DECKED_OUT)
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    (postmortem,) = report.loss_postmortems
    assert postmortem.decked_out is True
    assert postmortem.wiped_out is False
    assert postmortem.unexplained is False


def test_loss_postmortem_wiped_out() -> None:
    episodes = [
        _episode(episode_id=1, result="loss", game_end_reason=GameEndReason.NO_POKEMON_LEFT)
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    (postmortem,) = report.loss_postmortems
    assert postmortem.wiped_out is True
    assert postmortem.decked_out is False
    assert postmortem.unexplained is False


def test_loss_postmortem_ended_by_card_effect() -> None:
    episodes = [
        _episode(episode_id=1, result="loss", game_end_reason=GameEndReason.CARD_EFFECT)
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    (postmortem,) = report.loss_postmortems
    assert postmortem.ended_by_card_effect is True
    assert postmortem.unexplained is False


def test_loss_postmortem_prizes_taken_reason_is_not_itself_a_cause() -> None:
    """The normal win/loss condition (reason=1) doesn't explain anything on
    its own - it's the "expected" way a game ends, not a distinguishing tag."""
    episodes = [
        _episode(episode_id=1, result="loss", game_end_reason=GameEndReason.PRIZES_TAKEN)
    ]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    (postmortem,) = report.loss_postmortems
    assert postmortem.decked_out is False
    assert postmortem.wiped_out is False
    assert postmortem.ended_by_card_effect is False


def test_loss_postmortem_not_unexplained_when_a_named_cause_applies() -> None:
    episodes = [_episode(episode_id=1, result="loss", had_basic_pokemon=False)]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)

    (postmortem,) = report.loss_postmortems
    assert postmortem.unexplained is False


def test_build_report_handles_no_episodes() -> None:
    report = build_report([], decklist=None, card_index=CARD_INDEX)

    assert report.episodes_analyzed == 0
    assert report.no_basic_pokemon_rate == 0.0
    assert report.average_ko_margin == 0.0
    assert report.card_stats == []
    assert report.attack_stats == []
    assert report.loss_postmortems == []


def test_generate_recommendations_flags_dominant_loss_cause() -> None:
    episodes = [
        _episode(
            episode_id=i,
            result="loss",
            our_kos=5,
            opponent_kos=1,
            attacks=[AttackUsage(attack_id=999, card_id=100)],
        )
        for i in range(1, 5)
    ] + [_episode(episode_id=5, result="win")]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)
    recommendations = generate_recommendations(report)

    assert any(
        r.category == "loss cause" and "lost the KO trade" in r.text for r in recommendations
    )


def test_generate_recommendations_min_games_filters_noise() -> None:
    """A card played (and lost with) only once must not be flagged - one
    data point is noise, not a pattern."""
    episodes = [_episode(episode_id=1, result="loss", cards_seen={100}, cards_played={100})]

    report = build_report(episodes, decklist=None, card_index=CARD_INDEX)
    recommendations = generate_recommendations(report, min_games=3)

    assert not any(r.category == "card" for r in recommendations)


def test_generate_recommendations_handles_no_episodes() -> None:
    report = build_report([], decklist=None, card_index=CARD_INDEX)

    assert generate_recommendations(report) == []
