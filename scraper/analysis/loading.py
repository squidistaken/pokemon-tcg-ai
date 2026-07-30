"""Corpus, manifest, and engine loaders for the deck analysis."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from .. import manifest as manifest_mod
from ..manifest import Manifest
from .console import console

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env.deck import load_deck

if TYPE_CHECKING:
    from scraper.card_index import CardIndex
    from src.env.card_database import CardDatabase


def load_all_decks(deck_dir: Path) -> tuple[list[str], list[list[int]]]:
    """
    Load every deck CSV in ``deck_dir`` exactly once.

    :param deck_dir: Directory containing the scraped deck CSV files.
    :return: (names, decks) where names[i] is the file stem and decks[i] is the
             list of 60 card IDs for that deck.
    """
    names: list[str] = []
    decks: list[list[int]] = []
    skipped = 0
    for path in ordered_deck_paths(deck_dir):
        try:
            decks.append(load_deck(str(path)))
            names.append(path.stem)
        except ValueError as exc:  # not a valid 60-card deck
            print(f"  skipping {path.name}: {exc}", file=sys.stderr)
            skipped += 1
    if skipped:
        print(f"  ({skipped} file(s) skipped)", file=sys.stderr)
    return names, decks


def ordered_deck_paths(deck_dir: Path) -> list[Path]:
    """
    Deck CSV paths in a stable, deterministic order.

    :param deck_dir: Corpus directory (searched recursively).
    :return: Nested (archetype-subfolder) deck CSV paths, sorted; falls back to
             loose files if the corpus is flat.
    """
    all_csvs = sorted(deck_dir.rglob("*.csv"))
    nested = [p for p in all_csvs if p.parent != deck_dir]
    return nested or all_csvs


def load_manifest(deck_dir: Path) -> Manifest:
    """
    Load the corpus manifest, keyed by deck stem.

    :param deck_dir: Directory that may contain ``manifest.json``.
    :return: The :class:`~scraper.manifest.Manifest`; empty if there is no manifest
        file. A v1 (pre-observations) file is upgraded in memory on read.
    :raises ManifestError: If a manifest exists but cannot be parsed.
    """
    return manifest_mod.load(deck_dir)


def load_archetypes(deck_dir: Path, names: list[str]) -> list[str] | None:
    """
    Look up each deck's archetype label from the corpus manifest, if present.

    :param deck_dir: Directory that may contain ``manifest.json``.
    :param names: Deck file stems, in matrix order.
    :return: Archetype label per deck, or None if no manifest is available.
    """
    manifest = load_manifest(deck_dir)
    if not manifest.decks:
        return None
    return [
        entry.archetype if (entry := manifest.decks.get(name)) else "Unknown"
        for name in names
    ]


def count_unique_decks(decks: list[list[int]]) -> int:
    """
    Count decks with a distinct card multiset.

    Uses the same notion of identity as the corpus's dedup key
    (:func:`scraper.writer.deck_hash`): order-agnostic but copy-count sensitive. A
    ``frozenset`` would be copy-count *blind*, so two decks differing only in how
    many copies they run — genuinely different decks, and separate manifest
    entries — would be reported as one.

    :param decks: List of decks, each a list of card IDs.
    :return: Number of distinct card multisets in the corpus.
    """
    return len({tuple(sorted(deck)) for deck in decks})


def load_card_database() -> CardDatabase | None:
    """
    Load the engine card database, or return None if it is unavailable.

    :return: A loaded ``CardDatabase``, or None if construction failed.
    """
    console.print("[dim]Loading CardDatabase ...[/]")
    try:
        from src.env.card_database import CardDatabase

        return CardDatabase()
    except Exception as exc:  # noqa: BLE001 - engine may be unbuilt
        console.print(f"[yellow](CardDatabase unavailable: {exc})[/]")
        return None


def load_card_index() -> CardIndex | None:
    """
    Load the scraper's card index (card ID -> set code / name), or None.

    :return: A ``scraper.card_index.CardIndex``, or None if it could not load.
    """
    try:
        from scraper.card_index import CardIndex

        return CardIndex()
    except Exception as exc:  # noqa: BLE001 - card CSV may be absent
        console.print(f"[yellow](CardIndex unavailable: {exc})[/]")
        return None
