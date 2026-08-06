"""The corpus manifest: provenance schema, occurrence merging, and durable I/O.

The manifest is the corpus's provenance record, stored as ``decks/manifest.json``
beside the deck CSVs. Every deck file gets one entry keyed by its slug (the CSV's
stem), and every *occurrence* of that deck — each time a player brought the same
60-card multiset to an event — is retained as an :class:`Observation`.

That last part is the point of schema v2. Under v1 a deck whose card multiset had
already been written was silently discarded, so a one-off brew and a list fourteen
players independently piloted across five tournaments were indistinguishable in the
corpus. ``observation_count`` is that popularity signal, and each observation keeps
its own event, dates, record, placing, URL, and source-native IDs.

Schema v3 adds structured card substitutions to each observation. Keeping them on
the occurrence rather than the deduplicated deck preserves which original cards
produced the same final competition-legal 60-card list.

Re-scrapes must not inflate the count, so every occurrence carries a stable
identity (:meth:`Observation.key`) built from the source's own IDs. Re-running the
same scrape re-observes occurrences already on file and changes nothing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 3
#: How ``id_hash`` is computed. See :func:`scraper.writer.deck_hash`; a change to
#: the algorithm must bump this string so stale hashes can't be mistaken for live ones.
HASH_ALGO = "sha1-sorted-card-ids-12"
MANIFEST_NAME = "manifest.json"


class ManifestError(RuntimeError):
    """Raised when an existing manifest cannot be read.

    Deliberately fatal: the manifest is accumulated state (observation counts,
    dedup hashes) that a fresh run cannot reconstruct from the current scrape, so
    silently starting from an empty manifest would destroy the corpus's history and
    rewrite every deck as a fresh duplicate.
    """


@dataclass
class Observation:
    """One occurrence of a deck — a single player bringing it to a single event.

    A deck with three observations is the same 60 cards seen three times; the
    per-occurrence metadata (who, where, how it placed) lives here rather than on
    the deck, because it genuinely differs between occurrences.
    """

    source: str
    archetype: str | None = None
    fmt: str | None = None
    event: str | None = None
    event_date: str | None = None  # when the event happened (YYYY-MM-DD)
    scraped_date: str | None = None  # when *we* fetched it; never an identity field
    record: str | None = None  # "W-L-T", if the source reports one
    placing: int | None = None
    url: str | None = None
    # Source-native identifiers that pin this occurrence down across re-scrapes,
    # e.g. {"tournament_id": "...", "placing": "3"} for limitless.
    external_ids: dict[str, str] = field(default_factory=dict)
    # Structured source-printing -> competition-card decisions for this occurrence.
    substitutions: list[dict[str, Any]] = field(default_factory=list)
    # Set when near-duplicate pruning folded this occurrence into another deck's
    # entry: the slug it was originally observed under. The cards it was observed
    # with differed slightly from its host entry's, so the provenance says so
    # rather than pretending it was an exact occurrence of the surviving list.
    merged_from: str | None = None

    def key(self) -> str:
        """Stable identity of this occurrence, so re-scrapes are idempotent.

        Prefers the source's own IDs; falls back to the descriptive fields when a
        source has none. ``scraped_date`` is deliberately excluded — including it
        would make the same occurrence look new on every calendar day.

        :return: An opaque string equal for two observations of the same occurrence.
        """
        if self.external_ids:
            ids = ";".join(
                f"{k}={self.external_ids[k]}" for k in sorted(self.external_ids)
            )
            return f"{self.source}|{ids}"
        parts = (self.url, self.event, self.placing, self.record, self.event_date)
        return f"{self.source}|" + "|".join("" if p is None else str(p) for p in parts)

    @property
    def seen_date(self) -> str | None:
        """
        :return: The event's date if known, else the date we scraped it.
        """
        return self.event_date or self.scraped_date

    def to_json(self) -> dict[str, Any]:
        """
        :return: The JSON object stored in the manifest (``format`` is spelled out
            in the file even though the field is ``fmt`` in code, matching v1).
        """
        return {
            "source": self.source,
            "archetype": self.archetype,
            "format": self.fmt,
            "event": self.event,
            "event_date": self.event_date,
            "scraped_date": self.scraped_date,
            "record": self.record,
            "placing": self.placing,
            "url": self.url,
            "external_ids": dict(self.external_ids),
            "substitutions": [dict(item) for item in self.substitutions],
            "merged_from": self.merged_from,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Observation:
        """
        :param data: One observation object from a manifest file.
        :return: The parsed :class:`Observation`.
        """
        return cls(
            source=data.get("source") or "",
            archetype=data.get("archetype"),
            fmt=data.get("format"),
            event=data.get("event"),
            event_date=data.get("event_date"),
            scraped_date=data.get("scraped_date"),
            record=data.get("record"),
            placing=data.get("placing"),
            url=data.get("url"),
            external_ids=dict(data.get("external_ids") or {}),
            substitutions=[
                dict(item)
                for item in data.get("substitutions") or []
                if isinstance(item, dict)
            ],
            merged_from=data.get("merged_from"),
        )


@dataclass
class DeckEntry:
    """A manifest entry: one deck file plus every occurrence of it we've seen."""

    file: str  # path relative to the corpus root
    id_hash: str  # canonical hash of the card multiset (the dedup key)
    archetype: str  # the label the deck file was created under (drives its slug)
    fmt: str | None = None
    warnings: list[str] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)

    @property
    def observation_count(self) -> int:
        """
        :return: How many distinct occurrences of this deck are on record. Always
            derived from ``observations`` so the stored count cannot drift.
        """
        return len(self.observations)

    @property
    def first_seen(self) -> str | None:
        """
        :return: Earliest date across observations, or None if none are dated.
        """
        dates = [o.seen_date for o in self.observations if o.seen_date]
        return min(dates) if dates else None

    @property
    def last_seen(self) -> str | None:
        """
        :return: Latest date across observations, or None if none are dated.
        """
        dates = [o.seen_date for o in self.observations if o.seen_date]
        return max(dates) if dates else None

    def add_observation(self, obs: Observation) -> bool:
        """Record an occurrence, unless it is one we already have.

        :param obs: The occurrence to record.
        :return: True if it was appended (a genuinely new occurrence), False if an
            observation with the same :meth:`Observation.key` was already present.
        """
        if any(existing.key() == obs.key() for existing in self.observations):
            return False
        self.observations.append(obs)
        return True

    def to_json(self) -> dict[str, Any]:
        """
        :return: The JSON object stored under this deck's slug.
        """
        return {
            "file": self.file,
            "id_hash": self.id_hash,
            "archetype": self.archetype,
            "format": self.fmt,
            "observation_count": self.observation_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "warnings": list(self.warnings),
            "observations": [o.to_json() for o in self.observations],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> DeckEntry:
        """
        :param data: One deck object from a v2 manifest file.
        :return: The parsed :class:`DeckEntry`.
        """
        return cls(
            file=data.get("file") or "",
            id_hash=data.get("id_hash") or "",
            archetype=data.get("archetype") or "",
            fmt=data.get("format"),
            warnings=list(data.get("warnings") or []),
            observations=[
                Observation.from_json(o) for o in data.get("observations") or []
            ],
        )

    @classmethod
    def from_v1_json(cls, data: dict[str, Any]) -> DeckEntry:
        """Upgrade a flat v1 entry, whose single occurrence becomes observation one.

        v1 stored only one occurrence per deck and used ``date`` for the scrape
        date (the event date was never recorded), so the upgraded observation has
        ``scraped_date`` set and ``event_date`` left None.

        :param data: One deck object from a v1 manifest file.
        :return: The equivalent current-schema :class:`DeckEntry`.
        """
        entry = cls(
            file=data.get("file") or "",
            id_hash=data.get("id_hash") or "",
            archetype=data.get("archetype") or "",
            fmt=data.get("format"),
            warnings=list(data.get("warnings") or []),
        )
        entry.observations.append(
            Observation(
                source=data.get("source") or "",
                archetype=data.get("archetype"),
                fmt=data.get("format"),
                event=data.get("event"),
                scraped_date=data.get("date"),
                record=data.get("record"),
                placing=data.get("placing"),
                url=data.get("url"),
            )
        )
        return entry


@dataclass
class Manifest:
    """The whole manifest: every deck entry, plus the schema header."""

    decks: dict[str, DeckEntry] = field(default_factory=dict)
    #: Schema version the file was read at; saves always write SCHEMA_VERSION.
    read_version: int = SCHEMA_VERSION

    def slugs_by_hash(self) -> dict[str, list[str]]:
        """
        :return: Card-multiset hash -> deck slugs, for finding the deck an incoming
            occurrence belongs to. The value is a list, not a single slug, because
            ``id_hash`` is truncated: a genuine collision must be able to hold two
            distinct decks under one hash rather than silently merging them. Slugs
            are sorted, so the mapping is deterministic.
        """
        index: dict[str, list[str]] = {}
        for slug in sorted(self.decks):
            h = self.decks[slug].id_hash
            if h:
                index.setdefault(h, []).append(slug)
        return index

    def observations(self) -> list[tuple[str, Observation]]:
        """
        :return: Every ``(slug, observation)`` pair in the corpus, so consumers can
            aggregate over occurrences rather than over deck files.
        """
        return [(slug, o) for slug, e in self.decks.items() for o in e.observations]

    def to_json(self) -> dict[str, Any]:
        """
        :return: The full manifest document, including the schema header.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "hash_algo": HASH_ALGO,
            "decks": {
                slug: entry.to_json() for slug, entry in sorted(self.decks.items())
            },
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Manifest:
        """Parse either schema version.

        A v1 file has no ``schema_version`` and is a bare slug -> entry mapping; it
        is upgraded in memory and rewritten as v3 on the next save. Schema v2 is
        read directly with empty per-observation substitutions.

        :param data: The parsed manifest document.
        :return: The :class:`Manifest`.
        :raises ManifestError: If the document isn't an object, or is a newer schema
            version than this code understands (writing it would drop fields).
        """
        if not isinstance(data, dict):
            raise ManifestError(
                f"manifest must be a JSON object, found {type(data).__name__}"
            )
        version = data.get("schema_version")
        if version is None:
            return cls(
                decks={
                    slug: DeckEntry.from_v1_json(entry)
                    for slug, entry in data.items()
                    if isinstance(entry, dict)
                },
                read_version=1,
            )
        if not isinstance(version, int) or version > SCHEMA_VERSION:
            raise ManifestError(
                f"manifest schema_version {version!r} is newer than this scraper "
                f"understands (v{SCHEMA_VERSION}); upgrade the scraper rather than "
                "letting it rewrite the file and drop fields"
            )
        raw_decks = data.get("decks")
        if not isinstance(raw_decks, dict):
            raise ManifestError("versioned manifest is missing its 'decks' object")
        return cls(
            decks={
                slug: DeckEntry.from_json(entry)
                for slug, entry in raw_decks.items()
                if isinstance(entry, dict)
            },
            read_version=version,
        )


def manifest_path(decks_dir: str | os.PathLike[str]) -> str:
    """
    :param decks_dir: Corpus root.
    :return: Path to that corpus's manifest file.
    """
    return os.path.join(os.fspath(decks_dir), MANIFEST_NAME)


def load(decks_dir: str | os.PathLike[str]) -> Manifest:
    """Load a corpus's manifest, or an empty one if it has none yet.

    :param decks_dir: Corpus root.
    :return: The :class:`Manifest`.
    :raises ManifestError: If the file exists but cannot be read or parsed. This is
        fatal on purpose — see :class:`ManifestError`. Every deck CSV re-hashes to
        its ``id_hash``, so a damaged manifest can be rebuilt from the corpus;
        overwriting it with a fresh one cannot be undone.
    """
    path = manifest_path(decks_dir)
    if not os.path.exists(path):
        return Manifest()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise ManifestError(
            f"cannot read manifest {path}: {exc}. It holds the corpus's accumulated "
            "provenance, so the scraper will not overwrite it. Move it aside "
            "(and rebuild from the deck CSVs) to continue."
        ) from exc
    return Manifest.from_json(data)


def save(manifest: Manifest, decks_dir: str | os.PathLike[str]) -> None:
    """Write a manifest to disk atomically.

    Serialises to a temporary file in the same directory, fsyncs it, then renames
    it over the target, so an interrupted save leaves the previous manifest intact
    rather than a truncated one.

    :param manifest: The manifest to persist.
    :param decks_dir: Corpus root.
    """
    path = manifest_path(decks_dir)
    tmp = f"{path}.tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest.to_json(), f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
