from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass

from . import manifest as manifest_mod
from .manifest import DeckEntry, Manifest, Observation
from .models import ResolvedDeck

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DECKS_DIR = os.path.join(_ROOT, "decks")
MANIFEST_NAME = manifest_mod.MANIFEST_NAME


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
    """Canonical hash of a deck's card multiset.

    The corpus's identity contract: this hash is **card-order agnostic** but
    **card-copy-count sensitive**. Sorting collapses every permutation of the same
    list onto one key, while keeping one entry per copy, so ``[1, 2, 2]`` and
    ``[2, 1, 2]`` hash alike but ``[1, 2, 2]`` and ``[1, 1, 2]`` do not. The comma
    separator is load-bearing: without it ``[11, 2]`` and ``[1, 12]`` would both
    encode as ``"112"``.

    :param ids: The deck's Card IDs (one entry per copy).
    :return: A 12-char hex digest identifying the card multiset (the dedup key).
    :raises ValueError: If ``ids`` is empty — an empty deck is not a deck, and
        hashing it would mint a stable-looking key for "nothing".
    :raises TypeError: If any ID is not an integer. IDs must not arrive as strings:
        they would sort lexicographically (``"10" < "9"``) and hash differently
        from the same deck's integer IDs.
    """
    if not ids:
        raise ValueError("cannot hash an empty deck")
    for i in ids:
        if isinstance(i, bool) or not isinstance(i, int):
            raise TypeError(f"deck IDs must be ints, found {type(i).__name__}: {i!r}")
    key = ",".join(str(i) for i in sorted(ids))
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def read_deck_ids(path: str) -> list[int]:
    """Read a deck CSV back into its Card IDs, in file order.

    :param path: Path to a deck CSV (one Card ID per line, no header).
    :return: The Card IDs, so a written deck can be re-hashed and checked against
        the manifest.
    :raises ValueError: If a line is not an integer.
    """
    with open(path, encoding="utf-8") as f:
        return [int(line) for line in f.read().split("\n") if line.strip()]


@dataclass
class WriteResult:
    """What :meth:`DeckWriter.write` did with a deck.

    The three cases are distinct and the caller needs to tell them apart: a new
    deck file, a new occurrence of a deck already on file, or an occurrence already
    recorded (a re-scrape, which must change nothing).
    """

    slug: str
    new_deck: bool  # a deck CSV was created
    new_observation: bool  # observation_count went up

    @property
    def reobserved(self) -> bool:
        """
        :return: True if this deck already existed and this occurrence of it was
            already on record — the idempotent re-scrape case.
        """
        return not self.new_deck and not self.new_observation


class DeckWriter:
    """Writes decks and maintains the manifest, merging repeat occurrences.

    Duplicate decks are *not* discarded: a deck whose card multiset is already in
    the corpus has its new occurrence appended to the existing entry's
    observations, which is what makes ``observation_count`` a popularity signal.
    """

    def __init__(self, decks_dir: str = DEFAULT_DECKS_DIR):
        """
        Open the deck directory and load any existing manifest.

        :param decks_dir: Corpus root the writer creates decks under (defaults to
            ``decks/``); its ``manifest.json`` is loaded if present.
        :raises ManifestError: If an existing manifest cannot be read. The writer
            refuses to run rather than overwrite accumulated provenance.
        """
        self.decks_dir = decks_dir
        self.manifest_path = manifest_mod.manifest_path(decks_dir)
        os.makedirs(decks_dir, exist_ok=True)
        self.manifest: Manifest = manifest_mod.load(decks_dir)
        self._slugs_by_hash = self.manifest.slugs_by_hash()

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
        slugs = set(self.manifest.decks)
        if os.path.isdir(self.decks_dir):
            for _root, _dirs, files in os.walk(self.decks_dir):
                slugs |= {f[:-4] for f in files if f.endswith(".csv")}
        return slugs

    def is_duplicate(self, ids: list[int]) -> bool:
        """
        :param ids: A resolved deck's Card IDs.
        :return: True if a deck with the same card multiset is already in the
            corpus. Note this no longer means "will be dropped" — a duplicate deck
            still contributes its occurrence to the existing entry.
        """
        return self.find_slug(ids) is not None

    def find_slug(self, ids: list[int]) -> str | None:
        """Find the corpus deck holding this exact card multiset.

        ``id_hash`` is truncated to 48 bits, so a hash match is strong evidence but
        not proof: two different decks sharing one is astronomically unlikely, and
        merging their provenance would be silent corruption. So a hash match is
        confirmed against the deck CSV on disk before it counts as the same deck.

        :param ids: A resolved deck's Card IDs.
        :return: The matching slug, or None if the corpus doesn't have this deck
            (including the case where a hash matches but the cards differ).
        """
        wanted = sorted(ids)
        for slug in self._slugs_by_hash.get(deck_hash(ids), []):
            entry = self.manifest.decks.get(slug)
            if entry is None:
                continue
            try:
                existing = read_deck_ids(os.path.join(self.decks_dir, entry.file))
            except (OSError, ValueError):
                # The CSV is gone or unreadable; trust the manifest's hash rather
                # than writing a second copy of a deck we believe we already have.
                return slug
            if sorted(existing) == wanted:
                return slug
        return None

    @staticmethod
    def observation_for(
        resolved: ResolvedDeck, *, date: str | None = None
    ) -> Observation:
        """Build the occurrence record for a scraped deck.

        :param resolved: The resolved deck, carrying its source deck's provenance.
        :param date: The date this scrape ran (not the event's date).
        :return: The :class:`~scraper.manifest.Observation` for this occurrence.
        """
        raw = resolved.source_deck
        return Observation(
            source=raw.source,
            archetype=raw.archetype,
            fmt=raw.fmt,
            event=raw.event,
            event_date=raw.event_date,
            scraped_date=date,
            record=raw.record,
            placing=raw.placing,
            url=raw.url,
            external_ids=dict(raw.external_ids),
            substitutions=[swap.to_json() for swap in resolved.swaps],
        )

    def classify(
        self, resolved: ResolvedDeck, *, date: str | None = None
    ) -> WriteResult:
        """Report what :meth:`write` would do, without touching the disk.

        :param resolved: The resolved deck to test.
        :param date: The date this scrape ran.
        :return: The :class:`WriteResult` a write would produce; its ``slug`` is
            the prospective slug when the deck is new.
        """
        slug = self.find_slug(resolved.ids)
        if slug is None:
            return WriteResult(
                slug=self._unique_slug(slugify(resolved.source_deck.archetype)),
                new_deck=True,
                new_observation=True,
            )
        obs = self.observation_for(resolved, date=date)
        entry = self.manifest.decks[slug]
        is_new = all(existing.key() != obs.key() for existing in entry.observations)
        return WriteResult(slug=slug, new_deck=False, new_observation=is_new)

    def write(
        self,
        resolved: ResolvedDeck,
        *,
        date: str | None = None,
        warnings: list[str] | None = None,
    ) -> WriteResult:
        """
        Record a deck: write its CSV if new, and add this occurrence either way.

        A deck whose card multiset is already in the corpus is not written again —
        its occurrence is appended to the existing entry instead, bumping that
        entry's ``observation_count``. An occurrence already on record (the same
        source IDs, i.e. a re-scrape) changes nothing at all.

        :param resolved: The resolved deck to record.
        :param date: The date this scrape ran, stored as the observation's
            ``scraped_date``. The event's own date travels on the deck.
        :param warnings: Non-fatal validator warnings (e.g. impossible evolutions)
            to record alongside the deck, if any.
        :return: A :class:`WriteResult` saying whether the deck and/or the
            occurrence were new.
        """
        ids = resolved.ids
        h = deck_hash(ids)
        obs = self.observation_for(resolved, date=date)

        slug = self.find_slug(ids)
        if slug is not None:
            entry = self.manifest.decks[slug]
            existing = next(
                (item for item in entry.observations if item.key() == obs.key()),
                None,
            )
            provenance_added = False
            if (
                existing is not None
                and not existing.substitutions
                and obs.substitutions
            ):
                existing.substitutions = list(obs.substitutions)
                provenance_added = True
            added = entry.add_observation(obs)
            new_warnings = [w for w in warnings or [] if w not in entry.warnings]
            entry.warnings.extend(new_warnings)
            if added or new_warnings or provenance_added:
                self._save()
            return WriteResult(slug=slug, new_deck=False, new_observation=added)

        archetype_slug = slugify(resolved.source_deck.archetype)
        slug = self._unique_slug(archetype_slug)
        rel_file = os.path.join(archetype_slug, f"{slug}.csv")
        path = os.path.join(self.decks_dir, rel_file)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(str(i) for i in ids) + "\n")

        entry = DeckEntry(
            file=rel_file,
            id_hash=h,
            archetype=resolved.source_deck.archetype,
            fmt=resolved.source_deck.fmt,
            warnings=list(warnings or []),
        )
        entry.add_observation(obs)
        self.manifest.decks[slug] = entry
        self._slugs_by_hash.setdefault(h, []).append(slug)
        self._save()
        return WriteResult(slug=slug, new_deck=True, new_observation=True)

    def _save(self) -> None:
        """
        Persist the manifest atomically (see :func:`scraper.manifest.save`).
        """
        manifest_mod.save(self.manifest, self.decks_dir)

    def ensure_manifest(self) -> None:
        """Create an empty manifest when a completed run wrote no valid decks."""
        if not os.path.exists(self.manifest_path):
            self._save()
