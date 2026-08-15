"""
Download top-leaderboard teams' replays as a behaviour-cloning corpus.

``submission_analysis/scout.py`` already pulls these replays and keeps only the
decklists. This keeps the whole file and records, per episode, which seat the
scouted team played and how that episode ended, which is everything the
cloning step needs on top of the raw JSON.

Network-bound, so it runs alongside GPU work.
"""
import argparse
import json
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

OUTPUT_DIR = Path("logs/expert_replays")


def harvest(competition: str, top_n: int, per_team: int, pause: float) -> None:
    """
    Pull replays for the top teams and write a manifest per team.

    :param competition: Kaggle competition slug.
    :param top_n: Leaderboard teams to harvest, best first.
    :param per_team: Episodes to pull per team.
    :param pause: Seconds between replay downloads, to stay under rate limits.
    """
    api = KaggleApi()
    api.authenticate()
    # competition_leaderboard_view returns one page of 20; asking for 40 teams
    # silently gave 20 before this. Page until the requested depth is reached.
    leaderboard: list = []
    page = 1
    while len(leaderboard) < top_n:
        try:
            chunk = api.competition_leaderboard_view(competition, page=page)
        except TypeError:
            chunk = api.competition_leaderboard_view(competition)
        if not chunk:
            break
        leaderboard.extend(chunk)
        if len(chunk) < 20:
            break
        page += 1
    leaderboard = leaderboard[:top_n]
    print(f"harvesting {len(leaderboard)} teams, up to {per_team} episodes each\n")

    grand_total = 0
    for rank, row in enumerate(leaderboard, start=1):
        label = f"#{rank} {getattr(row, 'team_name', row.team_id)}"
        try:
            subs = [s for s in (api.competition_team_submissions(row.team_id) or []) if s]
        except Exception as error:
            print(f"{label}: submissions unavailable ({type(error).__name__})")
            continue
        if not subs:
            print(f"{label}: no public submissions")
            continue
        # Every submission a team made has its own episodes. Taking only the
        # best one left roughly 60% of their games undownloaded.
        episodes = []
        submission_ids = set()
        for submission in subs:
            try:
                found = [e for e in (api.competition_list_episodes(submission.id) or []) if e]
            except Exception:
                continue
            submission_ids.add(submission.id)
            episodes.extend(found)

        team_dir = OUTPUT_DIR / str(row.team_id)
        team_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = team_dir / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        )

        pulled = 0
        for episode in episodes[:per_team]:
            theirs = next(
                (
                    agent
                    for agent in (episode.agents or [])
                    if agent.submission_id in submission_ids
                ),
                None,
            )
            if theirs is None:
                continue
            replay_path = team_dir / f"episode-{episode.id}-replay.json"
            if not replay_path.is_file():
                try:
                    api.competition_episode_replay(
                        episode.id, path=str(team_dir), quiet=True
                    )
                except Exception:
                    continue
                time.sleep(pause)
            if not replay_path.is_file():
                continue
            manifest[str(episode.id)] = {
                "expert_index": theirs.index,
                "expert_reward": getattr(theirs, "reward", None),
                "team_id": row.team_id,
                "team_name": getattr(row, "team_name", None),
                "rank": rank,
                "public_score": getattr(row, "score", None),
            }
            pulled += 1
        manifest_path.write_text(json.dumps(manifest, indent=1))
        grand_total += pulled
        print(f"{label}: {pulled} new, {len(manifest)} total in {team_dir}")

    print(f"\nnew replays this run: {grand_total}")
    print(f"corpus now: {len(list(OUTPUT_DIR.glob('*/episode-*.json')))} replays")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--per-team", type=int, default=40)
    parser.add_argument("--pause", type=float, default=0.4)
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    harvest(args.competition, args.top_n, args.per_team, args.pause)


if __name__ == "__main__":
    main()
