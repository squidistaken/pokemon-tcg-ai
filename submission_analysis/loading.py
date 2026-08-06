"""Load card data, a decklist, and downloaded replays for analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import NamedTuple

from cg.api import Attack, CardData, all_attack, all_card_data
from submission_analysis.parser import ParsedEpisode, parse_replay

REPLAY_MANIFEST_FILENAME = "manifest.json"


class RatingHistoryRow(NamedTuple):
    """One row of ``logs/kaggle_rating_history.csv``, typed for plotting."""

    fetched_at_utc: str
    kaggle_ref: int
    label: str
    status: str
    public_score: float | None
    leaderboard_rank: int | None = None


class CardIndex(NamedTuple):
    """Card/attack lookup tables, built once and passed around by reference."""

    cards: dict[int, CardData]
    attacks: dict[int, Attack]


def load_card_index() -> CardIndex:
    """
    Load every card and attack the engine knows about.

    :return: Lookup tables keyed by card ID and attack ID.
    """
    cards = {card.cardId: card for card in all_card_data()}
    attacks = {attack.attackId: attack for attack in all_attack()}
    return CardIndex(cards=cards, attacks=attacks)


def card_name(index: CardIndex, card_id: int) -> str:
    """
    Look up a card's name, falling back to its ID if unknown.

    :param index: Card/attack lookup tables from :func:`load_card_index`.
    :param card_id: Card ID to look up.
    :return: The card's name, or ``"card <id>"`` if not found.
    """
    card = index.cards.get(card_id)
    return card.name if card is not None else f"card {card_id}"


def attack_name(index: CardIndex, attack_id: int) -> str:
    """
    Look up an attack's name, falling back to its ID if unknown.

    :param index: Card/attack lookup tables from :func:`load_card_index`.
    :param attack_id: Attack ID to look up.
    :return: The attack's name, or ``"attack <id>"`` if not found.
    """
    attack = index.attacks.get(attack_id)
    return attack.name if attack is not None else f"attack {attack_id}"


def load_deck(path: Path) -> list[int]:
    """
    Read a deck CSV: one integer card ID per non-empty line.

    :param path: Deck CSV path.
    :return: Card IDs in file order, one entry per physical copy.
    """
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [int(line) for line in lines if line]


def discover_submission_dirs(replays_dir: Path) -> list[Path]:
    """
    Find every submission-ref subdirectory with a manifest under replays_dir.

    :param replays_dir: Root directory ``episodes --download-replays`` wrote
        into (one subdirectory per submission ref).
    :return: Matching subdirectories, sorted by name.
    """
    if not replays_dir.is_dir():
        return []
    return sorted(
        child
        for child in replays_dir.iterdir()
        if child.is_dir() and (child / REPLAY_MANIFEST_FILENAME).is_file()
    )


def load_manifest(submission_dir: Path) -> dict[int, dict]:
    """
    Read one submission's manifest.json into episode_id -> outcome info.

    :param submission_dir: One submission ref's replay directory.
    :return: Episode ID -> outcome dict (as written by
        ``episodes.download_replays``), or empty if there's no manifest yet.
    """
    manifest_path = submission_dir / REPLAY_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return {}
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {int(episode_id): info for episode_id, info in raw.items()}


def load_parsed_episodes(submission_dirs: list[Path]) -> list[ParsedEpisode]:
    """
    Parse every replay with a manifest entry across the given submission dirs.

    Replay files with no matching manifest row (an interrupted download, or a
    replay dropped in by hand) are skipped rather than guessed at.

    :param submission_dirs: Replay directories to parse, as returned by
        :func:`discover_submission_dirs`.
    :return: One :class:`~submission_analysis.parser.ParsedEpisode` per
        manifest-backed replay, across all given directories.
    """
    episodes: list[ParsedEpisode] = []
    for submission_dir in submission_dirs:
        manifest = load_manifest(submission_dir)
        for episode_id, info in manifest.items():
            replay_path = submission_dir / f"episode-{episode_id}-replay.json"
            if not replay_path.is_file():
                continue
            raw = json.loads(replay_path.read_text(encoding="utf-8"))
            episodes.append(
                parse_replay(
                    raw,
                    episode_id=episode_id,
                    our_index=info["our_index"],
                    result=info["result"],
                    opponent_team=info["opponent_team"],
                )
            )
    return episodes


def load_rating_history(path: Path) -> list[RatingHistoryRow]:
    """
    Read the local rating history.

    :param path: Rating-history CSV path.
    :return: One row per recorded snapshot, or empty if the file doesn't
        exist yet.
    """
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8", newline="") as history_file:
        for row in csv.DictReader(history_file):
            score = row.get("public_score")
            rank = row.get("leaderboard_rank")
            rows.append(
                RatingHistoryRow(
                    fetched_at_utc=row["fetched_at_utc"],
                    kaggle_ref=int(row["kaggle_ref"]),
                    label=row["label"],
                    status=row["status"],
                    public_score=float(score) if score else None,
                    leaderboard_rank=int(rank) if rank else None,
                )
            )
    return rows
