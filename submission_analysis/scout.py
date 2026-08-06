"""Scout top-leaderboard teams' decks via their public replays."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from rich.box import ROUNDED
from rich.table import Table

from submission_analysis.console import console
from submission_analysis.loading import CardIndex, card_name, load_card_index, load_deck
from submission_analysis.parser import parse_replay
from submission_analysis.plots import _BLUE, _ORANGE, plt
from submission_analysis.reporting import save_report
from submission_analysis.submissions import (
    COMPETITION,
    FetchError,
    repo_root,
    resolve_local_path,
)

SCOUTED_REPLAYS_DIR = "logs/scouted_replays"
OUTPUT_DIR = "outputs/submission_analysis/scout"
DEFAULT_TOP_N = 8
DEFAULT_EPISODES_PER_TEAM = 3
DECK_PATH = "decks/example.csv"


@dataclass(frozen=True)
class ScoutedTeam:
    """One leaderboard team's sampled deck, reconstructed from their public replays."""

    team_id: int
    team_name: str
    score: float
    submission_id: int
    episodes_sampled: int
    cards: frozenset[int]
    """Union of cards_played across every sampled episode - different games
    can reveal different slices of the same 60-card decklist, so more
    samples fill in more of it (never a guaranteed-complete reconstruction)."""


def fetch_leaderboard(competition: str, *, top_n: int) -> list[Any]:
    """
    Fetch the top ``top_n`` rows of ``competition``'s public leaderboard.

    :param competition: Kaggle competition slug.
    :param top_n: Number of leaderboard rows to fetch.
    :return: Leaderboard rows, best score first.
    :raises FetchError: If Kaggle API authentication fails.
    """
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as error:
        raise FetchError(
            f"Could not authenticate with the Kaggle API: {error}\n"
            "Put KAGGLE_API_TOKEN in the untracked .env, run "
            "'uv run kaggle auth login', or configure another credential "
            "method supported by the CLI."
        ) from error
    rows = api.competition_leaderboard_view(competition, page_size=top_n) or []
    return rows[:top_n]


def scout_team(
    row: Any, *, episodes_per_team: int, replay_dir: Path
) -> ScoutedTeam | None:
    """
    Download a sample of one leaderboard team's replays and reconstruct their deck.

    :param row: One leaderboard row from :func:`fetch_leaderboard`.
    :param episodes_per_team: Replays to sample for this team.
    :param replay_dir: Root directory to download this team's replays into
        (one subdirectory per team ID).
    :return: The reconstructed team, or None if it has no public submissions
        or episodes to sample.
    """
    api = KaggleApi()
    api.authenticate()

    subs = [sub for sub in (api.competition_team_submissions(row.team_id) or []) if sub]
    if not subs:
        return None
    submission = max(
        subs, key=lambda sub: float(sub.public_score) if sub.public_score else 0.0
    )

    episodes = [ep for ep in (api.competition_list_episodes(submission.id) or []) if ep]
    if not episodes:
        return None
    sample = episodes[:episodes_per_team]

    team_dir = replay_dir / str(row.team_id)
    team_dir.mkdir(parents=True, exist_ok=True)

    cards: set[int] = set()
    for episode in sample:
        replay_path = team_dir / f"episode-{episode.id}-replay.json"
        if not replay_path.is_file():
            api.competition_episode_replay(episode.id, path=str(team_dir), quiet=True)
        if not replay_path.is_file():
            continue
        raw = json.loads(replay_path.read_text(encoding="utf-8"))
        for side in (0, 1):
            parsed = parse_replay(
                raw, episode_id=episode.id, our_index=side, result="", opponent_team=""
            )
            cards |= parsed.cards_played

    return ScoutedTeam(
        team_id=row.team_id,
        team_name=row.team_name,
        score=float(row.score) if row.score else 0.0,  # SDK returns this as a string
        submission_id=submission.id,
        episodes_sampled=len(sample),
        cards=frozenset(cards),
    )


def card_frequency(teams: Sequence[ScoutedTeam]) -> Counter[int]:
    """
    How many scouted teams' decks include each card.

    :param teams: Scouted teams to count across.
    :return: Card ID -> number of teams whose sampled deck includes it.
    """
    frequency: Counter[int] = Counter()
    for team in teams:
        for card_id in team.cards:
            frequency[card_id] += 1
    return frequency


def gap_analysis(
    teams: Sequence[ScoutedTeam], our_deck: Sequence[int], *, min_teams: int
) -> list[tuple[int, int]]:
    """
    Cards run by at least ``min_teams`` scouted teams but missing from ``our_deck``.

    Sorted by frequency descending - the strongest, most-consensus gaps first.

    :param teams: Scouted teams to check against.
    :param our_deck: Our own deck's card IDs.
    :param min_teams: Minimum scouted-team count for a card to count as a gap.
    :return: ``(card_id, team_count)`` pairs, most consensus first.
    """
    frequency = card_frequency(teams)
    our_cards = set(our_deck)
    gaps = [
        (card_id, count)
        for card_id, count in frequency.items()
        if count >= min_teams and card_id not in our_cards
    ]
    return sorted(gaps, key=lambda pair: pair[1], reverse=True)


def render_report(
    teams: Sequence[ScoutedTeam],
    card_index: CardIndex,
    *,
    our_deck: Sequence[int] | None,
    min_teams: int,
    top: int = 25,
) -> None:
    """
    Print the scouted teams and a cross-team card-frequency/gap report.

    :param teams: Scouted teams to report on.
    :param card_index: Card/attack lookup tables for name display.
    :param our_deck: Our own deck's card IDs, or None to skip the gap table.
    :param min_teams: Minimum scouted-team count for a card to count as a gap.
    :param top: Maximum rows in the card-frequency/gap tables.
    :return: None.
    """
    teams_table = Table(box=ROUNDED, title="Scouted teams", title_justify="left")
    teams_table.add_column("team")
    teams_table.add_column("score", justify="right")
    teams_table.add_column("submission", justify="right")
    teams_table.add_column("episodes sampled", justify="right")
    teams_table.add_column("cards seen", justify="right")
    for team in teams:
        teams_table.add_row(
            team.team_name,
            f"{team.score:.1f}",
            str(team.submission_id),
            str(team.episodes_sampled),
            str(len(team.cards)),
        )
    console.print(teams_table)

    if not teams:
        console.print(
            "[dim]No teams could be scouted (no public submissions/episodes).[/]"
        )
        return

    frequency = card_frequency(teams)
    freq_table = Table(
        box=ROUNDED,
        title=f"Card frequency across {len(teams)} scouted teams",
        title_justify="left",
    )
    freq_table.add_column("card")
    freq_table.add_column("teams", justify="right")
    for card_id, count in frequency.most_common(top):
        freq_table.add_row(card_name(card_index, card_id), f"{count}/{len(teams)}")
    console.print(freq_table)

    if our_deck is None:
        return

    gaps = gap_analysis(teams, our_deck, min_teams=min_teams)
    if not gaps:
        console.print(
            f"[green]No gaps: our deck already includes every card run by "
            f"{min_teams}+ of the scouted teams.[/]"
        )
        return
    gap_table = Table(
        box=ROUNDED,
        title=f"Worth considering: cards {min_teams}+ scouted teams run that we don't",
        title_style="bold yellow",
        title_justify="left",
    )
    gap_table.add_column("card")
    gap_table.add_column("teams", justify="right")
    for card_id, count in gaps[:top]:
        gap_table.add_row(card_name(card_index, card_id), f"{count}/{len(teams)}")
    console.print(gap_table)


def plot_card_frequency(
    teams: Sequence[ScoutedTeam],
    card_index: CardIndex,
    out_path: Path,
    *,
    top: int = 25,
) -> None:
    """
    Save a horizontal bar chart of card frequency across scouted teams.

    :param teams: Scouted teams to chart.
    :param card_index: Card/attack lookup tables for name display.
    :param out_path: PNG path to write.
    :param top: Maximum cards shown.
    :return: None.
    """
    if not teams:
        return
    frequency = card_frequency(teams).most_common(top)
    if not frequency:
        return
    labels = [card_name(card_index, card_id) for card_id, _ in frequency]
    values = [count / len(teams) for _, count in frequency]
    plt.figure(figsize=(9, max(4.0, len(labels) * 0.28)))
    plt.barh(range(len(labels)), values, color=_BLUE)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.gca().invert_yaxis()
    plt.xlabel("scouted teams running this card / scouted teams")
    plt.xlim(0, 1)
    plt.title(f"Card frequency across {len(teams)} scouted teams")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def plot_gap_analysis(
    teams: Sequence[ScoutedTeam],
    our_deck: Sequence[int],
    card_index: CardIndex,
    out_path: Path,
    *,
    min_teams: int,
    top: int = 25,
) -> None:
    """
    Save a horizontal bar chart of gap cards, most-consensus first.

    :param teams: Scouted teams to check against.
    :param our_deck: Our own deck's card IDs.
    :param card_index: Card/attack lookup tables for name display.
    :param out_path: PNG path to write.
    :param min_teams: Minimum scouted-team count for a card to count as a gap.
    :param top: Maximum cards shown.
    :return: None.
    """
    gaps = gap_analysis(teams, our_deck, min_teams=min_teams)[:top]
    if not gaps:
        return
    labels = [card_name(card_index, card_id) for card_id, _ in gaps]
    values = [count / len(teams) for _, count in gaps]
    plt.figure(figsize=(9, max(4.0, len(labels) * 0.28)))
    plt.barh(range(len(labels)), values, color=_ORANGE)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.gca().invert_yaxis()
    plt.xlabel("scouted teams running this card / scouted teams")
    plt.xlim(0, 1)
    plt.title(f"Cards {min_teams}+ scouted teams run that we don't")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓[/] {out_path}")


def _parser() -> argparse.ArgumentParser:
    """
    :return: The argument parser for the ``scout`` subcommand.
    """
    parser = argparse.ArgumentParser(
        prog="python -m submission_analysis scout",
        description=(
            "Scout top-leaderboard teams' decks via their public replays, and "
            "compare against our own deck."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=(
            f"How many leaderboard teams to scout (default: {DEFAULT_TOP_N}, this "
            "competition's round-2 cutoff)."
        ),
    )
    parser.add_argument(
        "--episodes-per-team",
        type=int,
        default=DEFAULT_EPISODES_PER_TEAM,
        help=f"Replays to sample per scouted team (default: {DEFAULT_EPISODES_PER_TEAM}).",
    )
    parser.add_argument(
        "--deck",
        help=(
            f"Our own deck CSV to diff against (default: {DECK_PATH}). Pass an "
            "empty string to skip the gap analysis."
        ),
    )
    parser.add_argument(
        "--min-teams",
        type=int,
        help=(
            "Only flag a gap card if at least this many scouted teams run it "
            "(default: a majority of --top-n)."
        ),
    )
    parser.add_argument(
        "--competition", help=f"Kaggle competition (default: {COMPETITION})."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI entry point.

    :param argv: Argument vector; defaults to ``sys.argv`` when None.
    :return: Process exit code.
    """
    args = _parser().parse_args(argv)
    root = repo_root()
    env_path = root / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)

    competition = args.competition or os.environ.get("KAGGLE_COMPETITION", COMPETITION)
    min_teams = args.min_teams or (args.top_n // 2 + 1)

    try:
        rows = fetch_leaderboard(competition, top_n=args.top_n)
    except FetchError as error:
        print(f"error: {error}")
        return 2
    if not rows:
        print("No leaderboard data returned.")
        return 0

    replays_dir = resolve_local_path(SCOUTED_REPLAYS_DIR, root)
    teams = []
    for row in rows:
        team = scout_team(
            row, episodes_per_team=args.episodes_per_team, replay_dir=replays_dir
        )
        if team is None:
            print(
                f"warning: could not scout {row.team_name!r} (team_id={row.team_id}); skipping."
            )
            continue
        teams.append(team)

    card_index = load_card_index()
    our_deck = None
    deck_slug = "no-deck"
    if args.deck != "":
        deck_path = resolve_local_path(args.deck or DECK_PATH, root)
        deck_slug = deck_path.stem
        if deck_path.is_file():
            our_deck = load_deck(deck_path)
        else:
            print(f"warning: deck CSV not found at {deck_path}; skipping gap analysis.")

    render_report(teams, card_index, our_deck=our_deck, min_teams=min_teams)

    output_dir = resolve_local_path(OUTPUT_DIR, root) / deck_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print()
    console.rule("[bold]Plots[/]")
    plot_card_frequency(teams, card_index, output_dir / "card_frequency.png")
    if our_deck is not None:
        plot_gap_analysis(
            teams,
            our_deck,
            card_index,
            output_dir / "gap_analysis.png",
            min_teams=min_teams,
        )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005
    save_report(output_dir / f"report_{stamp}.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
