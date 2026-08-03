"""Deterministic inventory of source card printings seen in scraped decks."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .card_index import CardIndex
from .card_swapper import SourceProfile
from .models import RawCard, RawDeck

INVENTORY_SCHEMA_VERSION = 1
DEFAULT_EXAMPLE_LIMIT = 3


class InventoryCheckpointMismatch(ValueError):
    """Raised when a checkpoint belongs to a different inventory context."""


@dataclass(frozen=True)
class CanonicalPrinting:
    """Source-independent identity supplied by a discovery canonicalizer."""

    name: str
    set_code: str | None = None
    number: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a canonical printing must have a name")

    def to_json(self) -> dict[str, str | None]:
        """Return the stable JSON representation of this identity."""
        return {
            "name": self.name,
            "set_code": self.set_code,
            "number": self.number,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> CanonicalPrinting:
        """Load a canonical printing from an inventory record."""
        return cls(
            name=_required_string(value, "name"),
            set_code=_optional_string(value.get("set_code")),
            number=_optional_string(value.get("number")),
        )


class Canonicalizer(Protocol):
    """Convert a source spelling into a stable cross-source printing identity."""

    def __call__(self, source: str, card: RawCard) -> CanonicalPrinting: ...


class InventoryProfileLoader(Protocol):
    """Load metadata for a canonicalized source printing without mutating it."""

    def __call__(self, card: RawCard) -> SourceProfile | None: ...


@dataclass(frozen=True)
class SourceSpelling:
    """One exact way a source represented a canonical printing."""

    source: str
    name: str
    set_code: str | None
    number: str | None
    category: str | None

    def to_json(self) -> dict[str, str | None]:
        """Return this spelling in stable field order."""
        return {
            "source": self.source,
            "name": self.name,
            "set_code": self.set_code,
            "number": self.number,
            "category": self.category,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> SourceSpelling:
        """Load a source spelling from JSON."""
        return cls(
            source=_required_string(value, "source"),
            name=_required_string(value, "name"),
            set_code=_optional_string(value.get("set_code")),
            number=_optional_string(value.get("number")),
            category=_optional_string(value.get("category")),
        )


@dataclass
class SourceCounts:
    """Frequency totals for a printing within one scraper source."""

    decks: int = 0
    lines: int = 0
    copies: int = 0

    def to_json(self) -> dict[str, int]:
        """Return source counts in stable field order."""
        return {"decks": self.decks, "lines": self.lines, "copies": self.copies}

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> SourceCounts:
        """Load and validate non-negative source counts."""
        result = cls(
            decks=_required_non_negative_int(value, "decks"),
            lines=_required_non_negative_int(value, "lines"),
            copies=_required_non_negative_int(value, "copies"),
        )
        return result

    def merge(self, other: SourceCounts) -> None:
        """Add another independent batch of observations."""
        self.decks += other.decks
        self.lines += other.lines
        self.copies += other.copies


@dataclass(frozen=True)
class DeckExample:
    """A stable, compact example of a deck containing a printing."""

    source: str
    archetype: str
    url: str | None = None
    event: str | None = None
    event_date: str | None = None
    external_ids: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_deck(cls, deck: RawDeck) -> DeckExample:
        """Build an example while sorting source-native identifiers."""
        return cls(
            source=deck.source,
            archetype=deck.archetype,
            url=deck.url,
            event=deck.event,
            event_date=deck.event_date,
            external_ids=tuple(sorted(deck.external_ids.items())),
        )

    def to_json(self) -> dict[str, object]:
        """Return this example in stable field order."""
        return {
            "source": self.source,
            "archetype": self.archetype,
            "url": self.url,
            "event": self.event,
            "event_date": self.event_date,
            "external_ids": dict(self.external_ids),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> DeckExample:
        """Load an example from JSON."""
        raw_ids = value.get("external_ids", {})
        if not isinstance(raw_ids, dict) or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in raw_ids.items()
        ):
            raise ValueError("external_ids must be an object containing strings")
        return cls(
            source=_required_string(value, "source"),
            archetype=_required_string(value, "archetype"),
            url=_optional_string(value.get("url")),
            event=_optional_string(value.get("event")),
            event_date=_optional_string(value.get("event_date")),
            external_ids=tuple(sorted(raw_ids.items())),
        )


@dataclass(frozen=True)
class OrdinaryMatch:
    """The ordinary competition-index result for a canonical printing."""

    method: str
    card_id: int | None
    matched_name: str | None
    score: float | None
    candidate_ids: tuple[int, ...]

    def to_json(self) -> dict[str, object]:
        """Return the auditable match result."""
        return {
            "method": self.method,
            "card_id": self.card_id,
            "matched_name": self.matched_name,
            "score": self.score,
            "candidate_ids": list(self.candidate_ids),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> OrdinaryMatch:
        """Load a match result from JSON."""
        candidate_ids = value.get("candidate_ids", [])
        if not isinstance(candidate_ids, list) or not all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in candidate_ids
        ):
            raise ValueError("candidate_ids must be an array of integers")
        return cls(
            method=_required_string(value, "method"),
            card_id=_optional_int(value.get("card_id"), "card_id"),
            matched_name=_optional_string(value.get("matched_name")),
            score=_optional_float(value.get("score"), "score"),
            candidate_ids=tuple(candidate_ids),
        )


@dataclass
class SeenCardRecord:
    """All observations and metadata accumulated for one canonical printing."""

    identity: CanonicalPrinting
    match: OrdinaryMatch
    spellings: set[SourceSpelling] = field(default_factory=set)
    counts_by_source: dict[str, SourceCounts] = field(default_factory=dict)
    examples: set[DeckExample] = field(default_factory=set)
    profile: SourceProfile | None = None
    metadata_error: str | None = None

    def merge(self, other: SeenCardRecord, *, example_limit: int) -> None:
        """Merge an independent record for the same canonical printing."""
        if self.identity != other.identity:
            raise ValueError("cannot merge records for different printings")
        if self.match != other.match:
            raise ValueError(
                f"ordinary match changed for canonical printing {self.identity!r}"
            )
        self.spellings.update(other.spellings)
        for source, other_counts in other.counts_by_source.items():
            self.counts_by_source.setdefault(source, SourceCounts()).merge(other_counts)
        self.examples.update(other.examples)
        self.examples = set(
            sorted(self.examples, key=_example_sort_key)[:example_limit]
        )
        self._merge_metadata(other)

    def _merge_metadata(self, other: SeenCardRecord) -> None:
        if self.profile is not None and other.profile is not None:
            if _profile_to_json(self.profile) != _profile_to_json(other.profile):
                raise ValueError(
                    f"metadata changed for canonical printing {self.identity!r}"
                )
        elif other.profile is not None:
            self.profile = other.profile
        if self.profile is not None:
            self.metadata_error = None
        elif other.metadata_error is not None:
            errors = {
                error for error in (self.metadata_error, other.metadata_error) if error
            }
            self.metadata_error = min(errors)

    def to_json(self) -> dict[str, object]:
        """Return one schema-versioned inventory line."""
        return {
            "schema_version": INVENTORY_SCHEMA_VERSION,
            "record_type": "card",
            "identity": self.identity.to_json(),
            "spellings": [
                item.to_json()
                for item in sorted(self.spellings, key=_spelling_sort_key)
            ],
            "counts_by_source": {
                source: self.counts_by_source[source].to_json()
                for source in sorted(self.counts_by_source)
            },
            "match": self.match.to_json(),
            "examples": [
                item.to_json() for item in sorted(self.examples, key=_example_sort_key)
            ],
            "profile": _profile_to_json(self.profile) if self.profile else None,
            "metadata_error": self.metadata_error,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> SeenCardRecord:
        """Load and validate one inventory line."""
        if value.get("schema_version") != INVENTORY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported inventory schema version {value.get('schema_version')!r}"
            )
        identity = _required_mapping(value, "identity")
        match = _required_mapping(value, "match")
        raw_spellings = _required_list(value, "spellings")
        raw_counts = _required_mapping(value, "counts_by_source")
        raw_examples = _required_list(value, "examples")
        profile_value = value.get("profile")
        if profile_value is not None and not isinstance(profile_value, dict):
            raise ValueError("profile must be an object or null")
        error = _optional_string(value.get("metadata_error"))
        return cls(
            identity=CanonicalPrinting.from_json(identity),
            match=OrdinaryMatch.from_json(match),
            spellings={
                SourceSpelling.from_json(_as_mapping(item, "spelling"))
                for item in raw_spellings
            },
            counts_by_source={
                source: SourceCounts.from_json(_as_mapping(counts, "source counts"))
                for source, counts in raw_counts.items()
                if isinstance(source, str)
            },
            examples={
                DeckExample.from_json(_as_mapping(item, "example"))
                for item in raw_examples
            },
            profile=_profile_from_json(profile_value) if profile_value else None,
            metadata_error=error,
        )


class SeenCardInventory:
    """Incrementally build and checkpoint a deterministic printing inventory."""

    def __init__(
        self,
        index: CardIndex,
        canonicalizer: Canonicalizer,
        profile_loader: InventoryProfileLoader | None = None,
        *,
        example_limit: int = DEFAULT_EXAMPLE_LIMIT,
        checkpoint_label: str | None = None,
    ):
        if example_limit < 0:
            raise ValueError("example_limit must not be negative")
        self.index = index
        self.canonicalizer = canonicalizer
        self.profile_loader = profile_loader
        self.example_limit = example_limit
        self.checkpoint_label = checkpoint_label
        self._records: dict[CanonicalPrinting, SeenCardRecord] = {}
        self._observation_keys: set[str] = set()

    @property
    def records(self) -> tuple[SeenCardRecord, ...]:
        """Return records sorted by their canonical printing identity."""
        return tuple(
            self._records[key] for key in sorted(self._records, key=_identity_sort_key)
        )

    @property
    def observation_keys(self) -> frozenset[str]:
        """Return deck-occurrence identities already included in the totals."""
        return frozenset(self._observation_keys)

    def observe(self, deck: RawDeck) -> bool:
        """Aggregate a deck once, returning false when it was already seen."""
        observation_key = raw_deck_observation_key(deck)
        if observation_key in self._observation_keys:
            return False
        canonicalized = [
            (card, self.canonicalizer(deck.source, card)) for card in deck.cards
        ]
        identities_in_deck: set[CanonicalPrinting] = set()
        example = DeckExample.from_deck(deck)
        for card, identity in canonicalized:
            record = self._records.get(identity)
            if record is None:
                record = self._new_record(identity, card)
                self._records[identity] = record
            record.spellings.add(
                SourceSpelling(
                    source=deck.source,
                    name=card.name,
                    set_code=card.set_code,
                    number=card.number,
                    category=card.category,
                )
            )
            counts = record.counts_by_source.setdefault(deck.source, SourceCounts())
            counts.lines += 1
            counts.copies += max(0, card.count)
            identities_in_deck.add(identity)
        for identity in identities_in_deck:
            record = self._records[identity]
            record.counts_by_source[deck.source].decks += 1
            record.examples.add(example)
            record.examples = set(
                sorted(record.examples, key=_example_sort_key)[: self.example_limit]
            )
        self._observation_keys.add(observation_key)
        return True

    def extend(self, decks: Iterable[RawDeck]) -> None:
        """Aggregate an iterable of source decks."""
        for deck in decks:
            self.observe(deck)

    def merge_file(self, path: Path | str) -> None:
        """Load a previous checkpoint and merge its independent totals."""
        checkpoint = _read_checkpoint(path)
        if not checkpoint.has_header:
            raise ValueError(
                "inventory checkpoint has no deck-observation header; "
                "it cannot be resumed safely"
            )
        if checkpoint.checkpoint_label != self.checkpoint_label:
            raise InventoryCheckpointMismatch(
                "inventory checkpoint belongs to a different mapping/discovery context"
            )
        overlap = self._observation_keys & checkpoint.observation_keys
        if overlap:
            if checkpoint.observation_keys <= self._observation_keys:
                return
            raise ValueError(
                "cannot merge checkpoints with partially overlapping deck observations"
            )
        for record in checkpoint.records:
            current = self._records.get(record.identity)
            if current is None:
                record.examples = set(
                    sorted(record.examples, key=_example_sort_key)[: self.example_limit]
                )
                self._records[record.identity] = record
            else:
                current.merge(record, example_limit=self.example_limit)
        self._observation_keys.update(checkpoint.observation_keys)

    def write(self, path: Path | str) -> None:
        """Atomically write the complete inventory as deterministic gzip JSONL."""
        write_inventory(
            path,
            self.records,
            observation_keys=self._observation_keys,
            checkpoint_label=self.checkpoint_label,
        )

    def refresh_metadata_errors(self) -> int:
        """Retry profile-less records, returning how many were repaired."""
        if self.profile_loader is None:
            return 0
        repaired = 0
        for record in self.records:
            if record.profile is not None:
                continue
            card = RawCard(
                count=1,
                name=record.identity.name,
                set_code=record.identity.set_code,
                number=record.identity.number,
            )
            try:
                profile = self.profile_loader(card)
                if profile is None:
                    record.metadata_error = "profile unavailable"
                    continue
                record.profile = profile
                record.metadata_error = None
                repaired += 1
            except Exception as exc:  # noqa: BLE001 - retained for the next retry
                record.metadata_error = f"{type(exc).__name__}: {exc}"
        return repaired

    def _new_record(
        self, identity: CanonicalPrinting, observed_card: RawCard
    ) -> SeenCardRecord:
        result = self.index.match(identity.name, identity.set_code, identity.number)
        match = OrdinaryMatch(
            method=result.method,
            card_id=result.card_id,
            matched_name=result.matched_name,
            score=result.score,
            candidate_ids=tuple(sorted(result.candidate_ids)),
        )
        profile: SourceProfile | None = None
        metadata_error: str | None = None
        if self.profile_loader is not None:
            canonical_card = RawCard(
                count=observed_card.count,
                name=identity.name,
                set_code=identity.set_code,
                number=identity.number,
                category=observed_card.category,
            )
            try:
                profile = self.profile_loader(canonical_card)
                if profile is None:
                    metadata_error = "profile unavailable"
            except Exception as exc:  # noqa: BLE001 - inventory records source failures
                metadata_error = f"{type(exc).__name__}: {exc}"
        return SeenCardRecord(
            identity=identity,
            match=match,
            profile=profile,
            metadata_error=metadata_error,
        )


@dataclass(frozen=True)
class _InventoryCheckpoint:
    records: tuple[SeenCardRecord, ...]
    observation_keys: frozenset[str]
    has_header: bool
    checkpoint_label: str | None


def read_inventory(path: Path | str) -> tuple[SeenCardRecord, ...]:
    """Read card records from a plain or gzip JSONL inventory checkpoint."""
    return _read_checkpoint(path).records


def resolve_inventory_path(path: Path | str) -> Path:
    """Resolve a logical inventory path, rejecting divergent plain/gzip copies."""
    requested = Path(path)
    variants = _inventory_path_variants(requested)
    if variants is None:
        if requested.is_file():
            return requested
        raise FileNotFoundError(requested)

    plain, compressed = variants
    available = [candidate for candidate in (plain, compressed) if candidate.is_file()]
    if not available:
        raise FileNotFoundError(requested)
    if len(available) == 2 and inventory_content_sha256(
        plain
    ) != inventory_content_sha256(compressed):
        raise ValueError(
            f"plain and gzip inventory copies differ: {plain} and {compressed}"
        )
    if requested in available:
        return requested
    return compressed if compressed in available else plain


def inventory_content_sha256(path: Path | str) -> str:
    """Hash the decompressed inventory content without loading it into memory."""
    digest = hashlib.sha256()
    with _open_inventory_binary(Path(path)) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_checkpoint(path: Path | str) -> _InventoryCheckpoint:
    """Read card records and the resumability header from one checkpoint."""
    records: dict[CanonicalPrinting, SeenCardRecord] = {}
    observation_keys: frozenset[str] = frozenset()
    found_header = False
    checkpoint_label: str | None = None
    with _open_inventory_text(Path(path)) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                data = _as_mapping(value, "record")
                record_type = data.get("record_type", "card")
                if record_type == "header":
                    if found_header or line_number != 1:
                        raise ValueError("inventory header must be the first record")
                    observation_keys = _header_observation_keys(data)
                    checkpoint_label = _optional_string(data.get("checkpoint_label"))
                    found_header = True
                    continue
                if record_type != "card":
                    raise ValueError(f"unknown inventory record type {record_type!r}")
                record = SeenCardRecord.from_json(data)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid inventory line {line_number}: {exc}"
                ) from exc
            if record.identity in records:
                raise ValueError(
                    f"duplicate canonical printing on inventory line {line_number}"
                )
            records[record.identity] = record
    ordered = tuple(records[key] for key in sorted(records, key=_identity_sort_key))
    return _InventoryCheckpoint(
        ordered, observation_keys, found_header, checkpoint_label
    )


def _inventory_path_variants(path: Path) -> tuple[Path, Path] | None:
    name = path.name
    if name.endswith(".jsonl.gz"):
        return path.with_name(name.removesuffix(".gz")), path
    if name.endswith(".jsonl"):
        return path, path.with_name(f"{name}.gz")
    if path.suffix:
        return None
    return path.with_name(f"{name}.jsonl"), path.with_name(f"{name}.jsonl.gz")


def _open_inventory_binary(path: Path):
    with path.open("rb") as handle:
        is_gzip = handle.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rb") if is_gzip else path.open("rb")


def _open_inventory_text(path: Path):
    with path.open("rb") as handle:
        is_gzip = handle.read(2) == b"\x1f\x8b"
    if is_gzip:
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def write_inventory(
    path: Path | str,
    records: Iterable[SeenCardRecord],
    *,
    observation_keys: Iterable[str] = (),
    checkpoint_label: str | None = None,
) -> None:
    """Atomically write sorted records with a reproducible gzip header."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered: dict[CanonicalPrinting, SeenCardRecord] = {}
    for record in records:
        if record.identity in ordered:
            raise ValueError(f"duplicate canonical printing {record.identity!r}")
        ordered[record.identity] = record
    checked_observation_keys: set[str] = set()
    for key in observation_keys:
        if not isinstance(key, str) or not key:
            raise ValueError("observation keys must be non-empty strings")
        checked_observation_keys.add(key)
    sorted_observation_keys = sorted(checked_observation_keys)

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as raw:
            temporary = Path(raw.name)
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                header = json.dumps(
                    {
                        "schema_version": INVENTORY_SCHEMA_VERSION,
                        "record_type": "header",
                        "observation_keys": sorted_observation_keys,
                        "checkpoint_label": checkpoint_label,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                zipped.write(header.encode("utf-8"))
                zipped.write(b"\n")
                for identity in sorted(ordered, key=_identity_sort_key):
                    payload = json.dumps(
                        ordered[identity].to_json(),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    zipped.write(payload.encode("utf-8"))
                    zipped.write(b"\n")
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def raw_deck_observation_key(deck: RawDeck) -> str:
    """Return the same stable occurrence identity used by corpus manifests."""
    if deck.external_ids:
        identifiers = ";".join(
            f"{key}={deck.external_ids[key]}" for key in sorted(deck.external_ids)
        )
        return f"{deck.source}|{identifiers}"
    parts = (deck.url, deck.event, deck.placing, deck.record, deck.event_date)
    return f"{deck.source}|" + "|".join(
        "" if part is None else str(part) for part in parts
    )


def _header_observation_keys(value: Mapping[str, object]) -> frozenset[str]:
    if value.get("schema_version") != INVENTORY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported inventory schema version {value.get('schema_version')!r}"
        )
    raw_keys = _required_list(value, "observation_keys")
    keys: list[str] = []
    for key in raw_keys:
        if not isinstance(key, str) or not key:
            raise ValueError("observation_keys must contain non-empty strings")
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError("observation_keys must not contain duplicates")
    return frozenset(keys)


def _profile_to_json(profile: SourceProfile) -> dict[str, object]:
    return {
        "name": profile.name,
        "stage": profile.stage,
        "rule": profile.rule,
        "previous_stage": profile.previous_stage,
        "hp": profile.hp,
        "energy_type": profile.energy_type,
        "weakness": profile.weakness,
        "resistance": profile.resistance,
        "retreat": profile.retreat,
        "moves": [list(move) for move in profile.moves],
    }


def _identity_sort_key(identity: CanonicalPrinting) -> tuple[str, str, str]:
    return (identity.name, identity.set_code or "", identity.number or "")


def _spelling_sort_key(
    spelling: SourceSpelling,
) -> tuple[str, str, str, str, str]:
    return (
        spelling.source,
        spelling.name,
        spelling.set_code or "",
        spelling.number or "",
        spelling.category or "",
    )


def _example_sort_key(
    example: DeckExample,
) -> tuple[str, str, str, str, str, tuple[tuple[str, str], ...]]:
    return (
        example.source,
        example.archetype,
        example.url or "",
        example.event or "",
        example.event_date or "",
        example.external_ids,
    )


def _profile_from_json(value: Mapping[str, object]) -> SourceProfile:
    raw_moves = _required_list(value, "moves")
    moves: list[tuple[str, str, str, str]] = []
    for raw_move in raw_moves:
        if (
            not isinstance(raw_move, list)
            or len(raw_move) != 4
            or not all(isinstance(part, str) for part in raw_move)
        ):
            raise ValueError("each profile move must contain four strings")
        moves.append(tuple(raw_move))
    return SourceProfile(
        name=_required_string(value, "name"),
        stage=_required_string(value, "stage"),
        rule=_required_string(value, "rule"),
        previous_stage=_optional_string(value.get("previous_stage")),
        hp=_optional_int(value.get("hp"), "hp"),
        energy_type=_optional_string(value.get("energy_type")),
        weakness=_optional_string(value.get("weakness")),
        resistance=_optional_string(value.get("resistance")),
        retreat=_optional_int(value.get("retreat"), "retreat"),
        moves=tuple(moves),
    )


def _required_mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    return _as_mapping(value.get(key), key)


def _as_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object")
    return value


def _required_list(value: Mapping[str, object], key: str) -> list[object]:
    result = value.get(key)
    if not isinstance(result, list):
        raise TypeError(f"{key} must be an array")
    return result


def _required_string(value: Mapping[str, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be a non-empty string")
    return result


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("optional string field has a non-string value")
    return value


def _optional_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer or null")
    return value


def _optional_float(value: object, label: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise TypeError(f"{label} must be a number or null")
    return float(value)


def _required_non_negative_int(value: Mapping[str, object], key: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool) or result < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return result
