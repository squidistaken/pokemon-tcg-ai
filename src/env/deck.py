import glob
from collections.abc import Iterable
from pathlib import Path

DECK_SIZE = 60


def load_deck(path: str) -> list[int]:
    """
    Load a 60-card deck from a CSV file with one card ID per line.

    :param path: Path to the deck CSV file.
    :return: List of 60 card IDs.
    :raises ValueError: If the file does not resolve to exactly 60 card IDs.
    """
    with open(path, "r") as file:
        lines = [line for line in file.read().split("\n") if line.strip()]

    if lines and not lines[0].strip().lstrip("-").isdigit():
        lines = lines[1:]  # account for headers, if ever

    if len(lines) != DECK_SIZE:
        raise ValueError(
            f"Deck must contain exactly {DECK_SIZE} cards, found {len(lines)}"
        )

    return [int(line) for line in lines]


def resolve_deck_paths(spec: str | Iterable[str]) -> list[str]:
    """
    Expand a deck-pool spec into a sorted list of deck CSV paths.

    A spec entry may be a directory, a glob pattern, or a single CSV path. A
    string spec is treated as one entry; an iterable is expanded entry by entry.

    Directories are searched recursively, since the corpus is organized into
    per-strategy subfolders. When a directory contains subfolders, loose CSVs
    sitting directly in it are ignored. Pointing at a single archetype folder
    still picks up its decks directly.

    :param spec: Directory, glob, single path, or an iterable of those.
    :return: Sorted, de-duplicated list of deck CSV paths.
    :raises ValueError: If the spec matches no CSV files.
    """
    entries = [spec] if isinstance(spec, str) else list(spec)
    paths: list[str] = []
    for entry in entries:
        candidate = Path(entry)
        if candidate.is_dir():
            paths.extend(_dir_deck_paths(candidate))
        else:
            matched = glob.glob(entry)
            paths.extend(matched if matched else [entry])
    unique = sorted({p for p in paths if p.endswith(".csv")})
    if not unique:
        raise ValueError(
            f"deck pool spec {spec!r} matched no CSV files; the scraped corpus is a "
            f"pulled artifact -- run ./scripts/fetch_decks.sh to install it."
        )
    return unique


def _dir_deck_paths(directory: Path) -> list[str]:
    """
    Collect deck CSV paths under ``directory``, preferring strategy subfolders.

    :param directory: Directory to search recursively for ``*.csv``.
    :return: CSVs nested in subfolders if any exist, else the directory's own.
    """
    csvs = sorted(directory.rglob("*.csv"))
    nested = [p for p in csvs if p.parent != directory]
    return [str(p) for p in (nested or csvs)]


def load_decks(paths: Iterable[str], *, skip_invalid: bool = True) -> list[list[int]]:
    """
    Load every deck CSV in ``paths`` into a list of card-ID lists.

    :param paths: Deck CSV paths (e.g. from :func:`resolve_deck_paths`).
    :param skip_invalid: Skip files that do not resolve to a valid 60-card deck
        instead of raising; matches the corpus-analysis loader's behaviour.
    :return: One card-ID list per successfully loaded deck.
    :raises ValueError: If ``skip_invalid`` is False and any deck is invalid, or
        if no valid deck could be loaded.
    """
    decks: list[list[int]] = []
    for path in paths:
        try:
            decks.append(load_deck(path))
        except (ValueError, OSError):
            if not skip_invalid:
                raise
    if not decks:
        raise ValueError("no valid decks could be loaded from the given paths")
    return decks
