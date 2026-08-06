from __future__ import annotations

from pathlib import Path

from rich.box import ROUNDED
from rich.table import Table

from submission_analysis.console import console
from submission_analysis.deck_report import DeckReport, generate_recommendations


def report_overview(report: DeckReport) -> None:
    """
    Print the top-line win/loss/draw and consistency summary.

    :param report: Aggregated deck report to summarize.
    :return: None.
    """
    table = Table(
        box=ROUNDED,
        title="Deck refinement report",
        title_style="bold cyan",
        title_justify="left",
        show_header=False,
    )
    table.add_column("metric", style="dim")
    table.add_column("value", justify="right", style="bold")
    table.add_row("episodes analyzed", str(report.episodes_analyzed))
    table.add_row(
        "record", f"[green]{report.wins}W[/] [red]{report.losses}L[/] {report.draws}D"
    )
    table.add_row("win rate (decided games)", f"{report.win_rate:.1%}")
    table.add_row("no Basic Pokemon rate", f"{report.no_basic_pokemon_rate:.1%}")
    table.add_row(
        "average KO margin",
        f"[green]+{report.average_ko_margin:.2f}[/]"
        if report.average_ko_margin >= 0
        else f"[red]{report.average_ko_margin:.2f}[/]",
    )
    table.add_row(
        "average prize margin",
        f"[green]+{report.average_prize_margin:.2f}[/]"
        if report.average_prize_margin >= 0
        else f"[red]{report.average_prize_margin:.2f}[/]",
    )
    wins_turn = report.average_first_attack_turn_wins
    losses_turn = report.average_first_attack_turn_losses
    if wins_turn is not None or losses_turn is not None:
        wins_label = f"{wins_turn:.1f}" if wins_turn is not None else "-"
        losses_label = f"{losses_turn:.1f}" if losses_turn is not None else "-"
        table.add_row(
            "avg turn of first attack (wins / losses)",
            f"[green]{wins_label}[/] / [red]{losses_label}[/]",
        )
    console.print(table)


def report_recommendations(report: DeckReport) -> None:
    """
    Print a short, prioritized list of what's worth investigating.

    :param report: Aggregated deck report to synthesize recommendations from.
    :return: None.
    """
    recommendations = generate_recommendations(report)
    if not recommendations:
        console.print(
            "[dim]Nothing crossed the flagged thresholds - no specific recommendations.[/]"
        )
        return
    table = Table(
        box=ROUNDED,
        title="Worth looking into",
        title_style="bold yellow",
        title_justify="left",
    )
    table.add_column("category", style="dim")
    table.add_column("note")
    for recommendation in recommendations:
        table.add_row(recommendation.category, recommendation.text)
    console.print(table)


def report_card_stats(report: DeckReport, *, top: int = 25) -> None:
    """
    Print the cards most worth reconsidering: least played and most stuck.

    :param report: Aggregated deck report to draw card stats from.
    :param top: Maximum rows per table.
    :return: None.
    """
    seen = [stat for stat in report.card_stats if stat.games_seen > 0]

    stuck = sorted(seen, key=lambda stat: stat.games_stuck, reverse=True)[:top]
    stuck_table = Table(
        box=ROUNDED,
        title="Most often stuck in hand (seen but not played)",
        title_style="bold",
        title_justify="left",
    )
    stuck_table.add_column("card")
    stuck_table.add_column("seen", justify="right")
    stuck_table.add_column("played", justify="right")
    stuck_table.add_column("stuck", justify="right", style="yellow")
    stuck_table.add_column("play rate", justify="right")
    for stat in stuck:
        label = f"{stat.name} [dim](evo)[/]" if stat.is_evolution else stat.name
        stuck_table.add_row(
            label,
            str(stat.games_seen),
            str(stat.games_played),
            str(stat.games_stuck),
            f"{stat.play_rate:.0%}",
        )
    console.print(stuck_table)

    played = [stat for stat in seen if stat.games_played > 0]
    by_impact = sorted(played, key=lambda stat: stat.win_rate_when_played)
    impact_table = Table(
        box=ROUNDED,
        title="Win rate when played (lowest first)",
        title_style="bold",
        title_justify="left",
    )
    impact_table.add_column("card")
    impact_table.add_column("played", justify="right")
    impact_table.add_column("wins", justify="right")
    impact_table.add_column("losses", justify="right")
    impact_table.add_column("win rate", justify="right")
    impact_table.add_column("prized", justify="right")
    for stat in by_impact[:top]:
        # Relative to the submission's own win rate.
        color = "red" if stat.win_rate_when_played < report.win_rate else "green"
        impact_table.add_row(
            stat.name,
            str(stat.games_played),
            str(stat.wins_when_played),
            str(stat.losses_when_played),
            f"[{color}]{stat.win_rate_when_played:.0%}[/]",
            str(stat.games_prized) if stat.games_prized else "-",
        )
    console.print(impact_table)

    never_seen = sorted(
        (stat for stat in report.card_stats if stat.games_seen == 0), key=lambda stat: stat.name
    )
    if never_seen:
        console.print(
            f"[dim]Never drawn in any analyzed game: "
            f"{', '.join(stat.name for stat in never_seen)}[/]"
        )

    fragile = sorted(
        (stat for stat in played if stat.times_knocked_out > 0),
        key=lambda stat: stat.ko_rate_when_played,
        reverse=True,
    )
    if fragile:
        fragile_table = Table(
            box=ROUNDED,
            title="Most often knocked out (played but didn't survive)",
            title_style="bold",
            title_justify="left",
        )
        fragile_table.add_column("card")
        fragile_table.add_column("played", justify="right")
        fragile_table.add_column("knocked out", justify="right", style="red")
        fragile_table.add_column("KO rate", justify="right")
        for stat in fragile[:top]:
            fragile_table.add_row(
                stat.name,
                str(stat.games_played),
                str(stat.times_knocked_out),
                f"{stat.ko_rate_when_played:.0%}",
            )
        console.print(fragile_table)

    prized = sorted(
        (stat for stat in seen if stat.games_prized > 0),
        key=lambda stat: stat.prize_rate,
        reverse=True,
    )
    if prized:
        prized_table = Table(
            box=ROUNDED,
            title="Most often stuck in prizes",
            title_style="bold",
            title_justify="left",
        )
        prized_table.add_column("card")
        prized_table.add_column("seen", justify="right")
        prized_table.add_column("prized", justify="right", style="blue")
        prized_table.add_column("prize rate", justify="right")
        for stat in prized[:top]:
            prized_table.add_row(
                stat.name,
                str(stat.games_seen),
                str(stat.games_prized),
                f"{stat.prize_rate:.0%}",
            )
        console.print(prized_table)

    bench_warmers = sorted(
        (stat for stat in played if stat.is_pokemon),
        key=lambda stat: stat.attack_utilization,
    )
    if bench_warmers:
        utilization_table = Table(
            box=ROUNDED,
            title="Played but rarely attacks with (lowest utilization first)",
            title_style="bold",
            title_justify="left",
        )
        utilization_table.add_column("card")
        utilization_table.add_column("played", justify="right")
        utilization_table.add_column("attacked with", justify="right")
        utilization_table.add_column("utilization", justify="right")
        for stat in bench_warmers[:top]:
            color = "red" if stat.attack_utilization < 0.4 else "green"
            utilization_table.add_row(
                stat.name,
                str(stat.games_played),
                str(stat.games_attacked_with),
                f"[{color}]{stat.attack_utilization:.0%}[/]",
            )
        console.print(utilization_table)


def report_evolution_stats(report: DeckReport, *, top: int = 15) -> None:
    """
    Print how often a deployed pre-evolution actually gets evolved.

    :param report: Aggregated deck report to draw evolution stats from.
    :param top: Maximum rows in the table.
    :return: None.
    """
    deployed = sorted(
        (stat for stat in report.evolution_stats if stat.games_pre_evo_played > 0),
        key=lambda stat: stat.conversion_rate,
    )
    if not deployed:
        return
    table = Table(
        box=ROUNDED,
        title="Evolution conversion rate",
        title_style="bold",
        title_justify="left",
    )
    table.add_column("line")
    table.add_column("pre-evo played", justify="right")
    table.add_column("evolved", justify="right")
    table.add_column("conversion rate", justify="right")
    for stat in deployed[:top]:
        color = "red" if stat.conversion_rate < 0.6 else "green"
        table.add_row(
            f"{stat.pre_evo_name} -> {stat.evo_name}",
            str(stat.games_pre_evo_played),
            str(stat.games_evolved),
            f"[{color}]{stat.conversion_rate:.0%}[/]",
        )
    console.print(table)


def report_attack_stats(report: DeckReport, *, top: int = 15) -> None:
    """
    Print our most-used attacks, ranked by use, with damage/KO impact.

    :param report: Aggregated deck report to draw attack stats from.
    :param top: Maximum rows in the table.
    :return: None.
    """
    table = Table(
        box=ROUNDED, title="Attack usage", title_style="bold", title_justify="left"
    )
    table.add_column("attack")
    table.add_column("card")
    table.add_column("uses", justify="right")
    table.add_column("avg damage", justify="right")
    table.add_column("KOs", justify="right")
    for stat in report.attack_stats[:top]:
        table.add_row(
            stat.name,
            stat.card_name,
            str(stat.uses),
            f"{stat.average_damage:.0f}",
            str(stat.kos),
        )
    console.print(table)


def report_opponent_attack_stats(report: DeckReport, *, top: int = 15) -> None:
    """
    Print the opponent attacks/cards most responsible for knocking out ours.

    :param report: Aggregated deck report to draw opponent attack stats from.
    :param top: Maximum rows in the table.
    :return: None.
    """
    stats = sorted(report.opponent_attack_stats, key=lambda stat: stat.kos, reverse=True)
    if not stats:
        return
    table = Table(
        box=ROUNDED,
        title="Threats: opponent attacks against us",
        title_style="bold red",
        title_justify="left",
    )
    table.add_column("attack")
    table.add_column("card")
    table.add_column("uses", justify="right")
    table.add_column("avg damage", justify="right")
    table.add_column("our KOs caused", justify="right", style="red")
    for stat in stats[:top]:
        table.add_row(
            stat.name,
            stat.card_name,
            str(stat.uses),
            f"{stat.average_damage:.0f}",
            str(stat.kos),
        )
    console.print(table)


def report_loss_postmortems(report: DeckReport) -> None:
    """
    Print a breakdown of *why* the analyzed losses happened.

    :param report: Aggregated deck report to draw loss postmortems from.
    :return: None.
    """
    losses = report.loss_postmortems
    if not losses:
        console.print("[dim]No losses in the analyzed episodes.[/]")
        return

    total = len(losses)
    no_basic_pokemon_count = sum(1 for loss in losses if loss.no_basic_pokemon)
    never_attacked = sum(1 for loss in losses if loss.never_attacked)
    evolution_stalled = sum(1 for loss in losses if loss.evolution_stalled)
    lost_ko_trade = sum(1 for loss in losses if loss.ko_deficit > 0)
    lost_prize_trade = sum(1 for loss in losses if loss.lost_prize_trade)
    decked_out = sum(1 for loss in losses if loss.decked_out)
    wiped_out = sum(1 for loss in losses if loss.wiped_out)
    ended_by_card_effect = sum(1 for loss in losses if loss.ended_by_card_effect)
    unexplained = sum(1 for loss in losses if loss.unexplained)

    table = Table(
        box=ROUNDED, title="Why we lost", title_style="bold red", title_justify="left"
    )
    table.add_column("cause")
    table.add_column("losses", justify="right")
    table.add_column("share of losses", justify="right")
    for label, count in (
        ("no Basic Pokemon in opener", no_basic_pokemon_count),
        ("never landed an attack", never_attacked),
        ("evolution stalled", evolution_stalled),
        ("lost the KO trade", lost_ko_trade),
        ("lost the prize trade (despite the KO count)", lost_prize_trade),
        ("decked out", decked_out),
        ("wiped out (no Pokemon left)", wiped_out),
        ("ended by a card effect", ended_by_card_effect),
        ("none of the above", unexplained),
    ):
        table.add_row(label, str(count), f"{count / total:.0%}")
    console.print(table)


def report_all(report: DeckReport) -> None:
    """
    Print the full deck-refinement report in one pass.

    :param report: Aggregated deck report to print.
    :return: None.
    """
    report_overview(report)
    report_recommendations(report)
    report_card_stats(report)
    report_attack_stats(report)
    report_opponent_attack_stats(report)
    report_evolution_stats(report)
    report_loss_postmortems(report)


def save_report(path: Path) -> None:
    """
    Write everything printed so far to ``path`` as plain text.

    :param path: Output path.
    :return: None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(console.export_text(), encoding="utf-8")
    print(f"✓ Report written to {path}")
