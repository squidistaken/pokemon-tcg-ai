from __future__ import annotations

import hashlib
import json
import os
import re

from .models import ResolvedDeck

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DECKS_DIR = os.path.join(_ROOT, "decks")
MANIFEST_NAME = "manifest.json"


def slugify(name: str) -> str:
    """Turn an archetype name into a safe file slug.

    :param name: Human-readable archetype name.
    :return: A lowercase, hyphenated slug (``"deck"`` if it would be empty).
    """
    s = name.strip().lower()
    s = re.sub(r"[’'\"]", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "deck"


def deck_hash(ids: list[int]) -> str:
    """Canonical hash of a deck's card multiset (order-independent).

    :param ids: The deck's Card IDs (one entry per copy).
    :return: A 12-char hex digest identifying the card multiset (dedup key).
    """
    key = ",".join(str(i) for i in sorted(ids))
    return hashlib.sha1(key.encode()).hexdigest()[:12]


class DeckWriter:
    """Writes decks and tracks a manifest, deduping by card multiset."""

    def __init__(self, decks_dir: str = DEFAULT_DECKS_DIR):
        """
        Open the deck directory and load any existing manifest for dedup.

        :param decks_dir: Corpus root the writer creates decks under (defaults
            to ``decks/``); its ``manifest.json`` is loaded if present.
        """
        self.decks_dir = decks_dir
        self.manifest_path = os.path.join(decks_dir, MANIFEST_NAME)
        os.makedirs(decks_dir, exist_ok=True)
        self.manifest: dict[str, dict] = self._load_manifest()
        self._seen_hashes = {
            entry.get("id_hash")
            for entry in self.manifest.values()
            if entry.get("id_hash")
        }

    def _load_manifest(self) -> dict[str, dict]:
        """
        :return: The existing manifest, or an empty dict if it is missing or
            unreadable.
        """
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _unique_slug(self, slug: str) -> str:
        """
        Disambiguate a slug against existing decks by appending ``-2``, ``-3``…

        :param slug: Desired base slug (the archetype slug).
        :return: A slug not already used by a manifest entry or on-disk deck.
        """
        candidate = slug
        n = 2
        existing = self._existing_slugs()
        while candidate in existing:
            candidate = f"{slug}-{n}"
            n += 1
        return candidate

    def _existing_slugs(self) -> set[str]:
        """
        :return: Every deck slug already in use, from both the manifest keys and
            the ``*.csv`` files on disk (searched recursively).
        """
        slugs = set(self.manifest.keys())
        if os.path.isdir(self.decks_dir):
            for _root, _dirs, files in os.walk(self.decks_dir):
                slugs |= {f[:-4] for f in files if f.endswith(".csv")}
        return slugs

    def is_duplicate(self, ids: list[int]) -> bool:
        """
        :param ids: A resolved deck's Card IDs.
        :return: True if a deck with the same card multiset was already written.
        """
        return deck_hash(ids) in self._seen_hashes

    def write(self, resolved: ResolvedDeck, *, date: str | None = None) -> str | None:
        """
        Write a deck CSV into its archetype folder and record its manifest entry.

        :param resolved: The resolved deck to write.
        :param date: Scrape date stored in the manifest entry.
        :return: The deck's slug, or None if it was skipped as a duplicate.
        """
        ids = resolved.ids
        h = deck_hash(ids)
        if h in self._seen_hashes:
            return None

        archetype_slug = slugify(resolved.raw.archetype)
        slug = self._unique_slug(archetype_slug)
        rel_file = os.path.join(archetype_slug, f"{slug}.csv")
        path = os.path.join(self.decks_dir, rel_file)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(str(i) for i in ids) + "\n")

        self.manifest[slug] = {
            "file": rel_file,
            "source": resolved.raw.source,
            "archetype": resolved.raw.archetype,
            "event": resolved.raw.event,
            "placing": resolved.raw.placing,
            "url": resolved.raw.url,
            "format": resolved.raw.fmt,
            "record": resolved.raw.record,
            "date": date,
            "id_hash": h,
        }
        self._seen_hashes.add(h)
        self._save_manifest()
        return slug

    def _save_manifest(self) -> None:
        """
        Write the manifest to disk (pretty-printed, key-sorted, UTF-8).
        """
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
            f.write("\n")
