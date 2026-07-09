"""Write resolved decks to ``decks/`` and maintain the metadata manifest.

The deck CSV itself stays a bare 60-line list of Card IDs so that
``main.read_deck_csv`` (and the engine) can load it unchanged. All metadata
(source, URL, archetype, dedup hash) lives in ``decks/manifest.json``.
"""

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
    """Turn an archetype name into a safe file slug."""
    s = name.strip().lower()
    s = re.sub(r"[’'\"]", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "deck"


def deck_hash(ids: list[int]) -> str:
    """Canonical hash of a deck's card multiset (order-independent)."""
    key = ",".join(str(i) for i in sorted(ids))
    return hashlib.sha1(key.encode()).hexdigest()[:12]


class DeckWriter:
    """Writes decks and tracks a manifest, deduping by card multiset."""

    def __init__(self, decks_dir: str = DEFAULT_DECKS_DIR):
        self.decks_dir = decks_dir
        self.manifest_path = os.path.join(decks_dir, MANIFEST_NAME)
        os.makedirs(decks_dir, exist_ok=True)
        self.manifest: dict[str, dict] = self._load_manifest()
        self._seen_hashes = {
            entry.get("id_hash") for entry in self.manifest.values() if entry.get("id_hash")
        }

    def _load_manifest(self) -> dict[str, dict]:
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _unique_slug(self, slug: str) -> str:
        candidate = slug
        n = 2
        while f"{candidate}.csv" in self._existing_filenames():
            candidate = f"{slug}-{n}"
            n += 1
        return candidate

    def _existing_filenames(self) -> set[str]:
        names = {entry["file"] for entry in self.manifest.values() if "file" in entry}
        if os.path.isdir(self.decks_dir):
            names |= {f for f in os.listdir(self.decks_dir) if f.endswith(".csv")}
        return names

    def is_duplicate(self, ids: list[int]) -> bool:
        return deck_hash(ids) in self._seen_hashes

    def write(self, resolved: ResolvedDeck, *, date: str | None = None) -> str | None:
        """Write a deck CSV + manifest entry. Returns the slug, or ``None`` if
        it was skipped as a duplicate."""
        ids = resolved.ids
        h = deck_hash(ids)
        if h in self._seen_hashes:
            return None

        slug = self._unique_slug(slugify(resolved.raw.archetype))
        filename = f"{slug}.csv"
        path = os.path.join(self.decks_dir, filename)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(str(i) for i in ids) + "\n")

        self.manifest[slug] = {
            "file": filename,
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
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
            f.write("\n")
