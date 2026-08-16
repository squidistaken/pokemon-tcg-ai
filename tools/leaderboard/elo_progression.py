"""
Fetch the live Kaggle ratings of our submitted agents and plot the Elo progression.

Run it again at any time: the figure is drawn from a fresh API call, so it always
shows the ratings as they stand at that moment.

    python tools/leaderboard/elo_progression.py
    python tools/leaderboard/elo_progression.py --output docs/elo_progression.png
"""

import argparse
import json
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from elo_figure import EloProgressionFigure
from elo_history import EloHistory
from kaggle_client import KaggleCompetitionClient

DEFAULT_COMPETITION = "pokemon-tcg-ai-battle"
DEFAULT_OUTPUT = Path("outputs/leaderboard/elo_progression.png")


def parse_arguments() -> argparse.Namespace:
    """
    Read the command line.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--competition", default=DEFAULT_COMPETITION, help="Kaggle competition slug"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="image to write; the suffix picks the format (.png, .pdf, .svg)",
    )
    parser.add_argument(
        "--dpi", type=int, default=200, help="resolution for bitmap formats"
    )
    parser.add_argument("--style", default="whitegrid", help="seaborn axes style")
    parser.add_argument(
        "--context",
        default="talk",
        choices=("paper", "notebook", "talk", "poster"),
        help="seaborn context; it scales every font and line",
    )
    parser.add_argument(
        "--figsize",
        default="14x8",
        help="figure size in inches, as WIDTHxHEIGHT",
    )
    parser.add_argument(
        "--dump-json",
        type=Path,
        default=None,
        help="also write the raw payload as JSON",
    )
    parser.add_argument(
        "--team-name",
        default=None,
        help="team name to match when the account lookup fails",
    )
    parser.add_argument(
        "--no-leaderboard",
        action="store_true",
        help="skip the leaderboard download, so the figure has no field reference lines",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="open the figure when it is written",
    )
    return parser.parse_args()


def main() -> None:
    """
    Fetch, plot and write the figure.
    """
    arguments = parse_arguments()
    client = KaggleCompetitionClient(arguments.competition)

    print(f"fetching submissions for {arguments.competition}")
    submissions = client.fetch_submissions()
    leaderboard = []
    if not arguments.no_leaderboard:
        print("fetching the public leaderboard")
        leaderboard = client.fetch_leaderboard()

    history = EloHistory(
        competition=arguments.competition,
        submissions=submissions,
        leaderboard=leaderboard,
        username=client.username(),
        team_name=arguments.team_name,
    )
    payload = history.to_payload()

    width, _, height = arguments.figsize.lower().partition("x")
    written = EloProgressionFigure(
        payload,
        style=arguments.style,
        context=arguments.context,
        figsize=(float(width), float(height)),
    ).render(arguments.output, dpi=arguments.dpi)
    if arguments.dump_json is not None:
        arguments.dump_json.parent.mkdir(parents=True, exist_ok=True)
        arguments.dump_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"payload  -> {arguments.dump_json}")

    best = (
        max(payload["points"], key=lambda point: point["score"])
        if payload["points"]
        else None
    )
    print(
        f"{len(payload['points'])} scored submissions, {len(payload['failures'])} without a score"
    )
    if best is not None:
        print(f"best {best['score']:.1f} ({best['label']})")
    print(f"figure   -> {written}")
    if arguments.open:
        webbrowser.open(written.resolve().as_uri())


if __name__ == "__main__":
    main()
