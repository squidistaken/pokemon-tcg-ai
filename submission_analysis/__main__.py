from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from datetime import datetime

from dotenv import load_dotenv

from submission_analysis import episodes, reporting, scout, submissions
from submission_analysis.deck_report import SubmissionSummary, build_report
from submission_analysis.loading import (
    RatingHistoryRow,
    discover_submission_dirs,
    load_card_index,
    load_deck,
    load_parsed_episodes,
    load_rating_history,
)
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

OUTPUT_DIR = "outputs/submission_analysis"
DECK_PATH = "decks/example.csv"

_SLUG_MAX_LENGTH = 40


def _label_slug(label: str) -> str:
    """
    A short, filesystem-safe slug for a submission's Kaggle message.

    :param label: Submission label/message.
    :return: A lowercase, hyphenated, length-capped slug.
    """
    slug = re.sub(r"[’'\"]", "", label.strip().lower())
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug[:_SLUG_MAX_LENGTH].strip("-")


def _labels_by_ref(history: list[RatingHistoryRow]) -> dict[int, str]:
    """
    Most recent Kaggle message seen for each ref, from the local status history.

    :param history: Rating-history rows, as returned by
        :func:`~submission_analysis.loading.load_rating_history`.
    :return: Kaggle ref -> most recently seen label.
    """
    return {row.kaggle_ref: row.label for row in history}


def _submission_folder_name(ref: str, labels_by_ref: dict[int, str]) -> str:
    """
    The ``deck-report`` output folder name for one submission ref.

    :param ref: Submission ref (a replay directory's name).
    :param labels_by_ref: Kaggle ref -> label, from :func:`_labels_by_ref`.
    :return: The folder name to use for this submission's report.
    """
    label = labels_by_ref.get(int(ref)) if ref.isdigit() else None
    if not label:
        return ref
    slug = _label_slug(label)
    return f"{ref}-{slug}" if slug else ref


def _deck_report_parser() -> argparse.ArgumentParser:
    """
    :return: The argument parser for the ``deck-report`` subcommand.
    """
    parser = argparse.ArgumentParser(
        prog="python -m submission_analysis deck-report",
        description=(
            "Build a deck-refinement report (card/attack/loss-cause stats "
            "plus plots) from replays downloaded by "
            "'episodes --download-replays'."
        ),
    )
    parser.add_argument(
        "--submission-ref",
        type=int,
        action="append",
        metavar="REF",
        help="Only analyze this submission's replays. Repeatable. Default: all downloaded.",
    )
    parser.add_argument(
        "--deck",
        help=f"Deck CSV to cross-reference against (default: {DECK_PATH}).",
    )
    return parser


def _run_deck_report(argv: Sequence[str] | None) -> int:
    """
    Run the ``deck-report`` subcommand.

    :param argv: Argument vector for the ``deck-report`` subcommand.
    :return: Process exit code.
    """
    args = _deck_report_parser().parse_args(argv)
    root = submissions.repo_root()
    env_path = root / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)

    replays_dir = submissions.resolve_local_path(episodes.REPLAYS_DIR, root)
    submission_dirs = discover_submission_dirs(replays_dir)
    if args.submission_ref:
        wanted = {str(ref) for ref in args.submission_ref}
        submission_dirs = [d for d in submission_dirs if d.name in wanted]
    if not submission_dirs:
        print(f"No downloaded replays found under {replays_dir}.")
        print("Run 'python -m submission_analysis episodes --download-replays' first.")
        return 0

    deck_path = submissions.resolve_local_path(args.deck or DECK_PATH, root)
    decklist = None
    if deck_path.is_file():
        decklist = load_deck(deck_path)
    else:
        print(f"warning: deck CSV not found at {deck_path}; skipping never-drawn-card coverage.")

    card_index = load_card_index()
    rating_history_path = submissions.resolve_local_path(submissions.RATING_HISTORY_FILE, root)
    rating_history = load_rating_history(rating_history_path)
    labels_by_ref = _labels_by_ref(rating_history)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005

    summaries: list[SubmissionSummary] = []
    for submission_dir in submission_dirs:
        parsed_episodes = load_parsed_episodes([submission_dir])
        if not parsed_episodes:
            print(f"No replays could be parsed for submission {submission_dir.name}; skipping.")
            continue

        folder_name = _submission_folder_name(submission_dir.name, labels_by_ref)
        report = build_report(parsed_episodes, decklist=decklist, card_index=card_index)
        summaries.append(
            SubmissionSummary(
                label=folder_name,
                win_rate=report.win_rate,
                average_ko_margin=report.average_ko_margin,
                average_first_attack_turn=report.average_first_attack_turn,
            )
        )
        reporting.console.rule(f"[bold cyan]Submission {folder_name}[/]")
        reporting.report_all(report)

        output_dir = submissions.resolve_local_path(f"{OUTPUT_DIR}/{folder_name}", root)
        output_dir.mkdir(parents=True, exist_ok=True)
        reporting.console.print()
        reporting.console.rule("[bold]Plots[/]")
        plot_card_play_rate(report.card_stats, output_dir / "card_play_rate.png")
        plot_win_rate_when_played(
            report.card_stats,
            output_dir / "win_rate_when_played.png",
            win_rate_baseline=report.win_rate,
        )
        plot_ko_rate(report.card_stats, output_dir / "ko_rate.png")
        plot_play_rate_vs_win_rate(
            report.card_stats,
            output_dir / "play_rate_vs_win_rate.png",
            win_rate_baseline=report.win_rate,
        )
        plot_ko_rate_vs_win_rate(
            report.card_stats,
            output_dir / "ko_rate_vs_win_rate.png",
            win_rate_baseline=report.win_rate,
        )
        plot_attack_utilization(report.card_stats, output_dir / "attack_utilization.png")
        plot_attack_usage(report.attack_stats, output_dir / "attack_usage.png")
        plot_damage_by_card(report.attack_stats, output_dir / "damage_by_card.png")
        plot_opponent_attack_usage(
            report.opponent_attack_stats, output_dir / "opponent_attack_usage.png"
        )
        plot_evolution_conversion_rate(
            report.evolution_stats, output_dir / "evolution_conversion_rate.png"
        )
        plot_loss_causes(report, output_dir / "loss_causes.png")
        plot_first_attack_turn_distribution(
            report.first_attack_turns, output_dir / "first_attack_turn_distribution.png"
        )
        plot_ko_margin_distribution(report.ko_margins, output_dir / "ko_margin_distribution.png")
        plot_game_length_distribution(
            report.game_lengths, output_dir / "game_length_distribution.png"
        )
        reporting.save_report(output_dir / f"report_{stamp}.txt")

    root_output_dir = submissions.resolve_local_path(OUTPUT_DIR, root)
    root_output_dir.mkdir(parents=True, exist_ok=True)
    plot_rating_history(rating_history, root_output_dir / "rating_history.png")
    plot_leaderboard_rank_over_time(
        rating_history, root_output_dir / "leaderboard_rank_over_time.png"
    )
    plot_win_rate_by_submission(summaries, root_output_dir / "win_rate_by_submission.png")
    plot_average_ko_margin_by_submission(
        summaries, root_output_dir / "average_ko_margin_by_submission.png"
    )
    plot_average_first_attack_turn_by_submission(
        summaries, root_output_dir / "average_first_attack_turn_by_submission.png"
    )

    return 0


_SUBCOMMANDS = ("status", "episodes", "deck-report", "scout")


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI entry point: dispatch to the requested subcommand.

    :param argv: Argument vector; defaults to ``sys.argv`` when None.
    :return: Process exit code.
    """
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] not in _SUBCOMMANDS:
        print(f"usage: python -m submission_analysis {{{','.join(_SUBCOMMANDS)}}} ...")
        return 2
    command, rest = argv[0], argv[1:]
    if command == "status":
        return submissions.main(rest)
    if command == "episodes":
        return episodes.main(rest)
    if command == "scout":
        return scout.main(rest)
    return _run_deck_report(rest)


if __name__ == "__main__":
    raise SystemExit(main())
