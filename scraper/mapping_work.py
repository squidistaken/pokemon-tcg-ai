"""Prepare and validate local-only agent work for explicit card mappings.

This module deliberately contains no model client or agent orchestration.  It turns
an immutable discovery inventory into deterministic files that built-in Codex
agents can inspect, and validates their positive proposals and independent reviews.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .card_index import CardIndex, normalize_name, normalize_number
from .card_swapper import LimitlessProfileLoader, SourceProfile
from .inventory import (
    SeenCardRecord,
    inventory_content_sha256,
    read_inventory,
    resolve_inventory_path,
)
from .models import RawCard

WORK_SCHEMA_VERSION = 1
RULE_SCHEMA_VERSION = 2
DEFAULT_INVENTORY = Path("outputs/card_discovery/seen_cards")
DEFAULT_WORK_ROOT = Path("outputs/card_mapping_work")
DEFAULT_SHARD_COUNT = 24


class MappingWorkError(ValueError):
    """Raised when mapping work is malformed, stale, or unsafe."""


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _write_json(path: Path, value: object) -> None:
    _atomic_write(path, _json_bytes(value))


def _write_jsonl(path: Path, values: Iterable[object]) -> None:
    content = b"".join(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
        for value in values
    )
    _atomic_write(path, content)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MappingWorkError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MappingWorkError(f"{path} must contain a JSON object")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise MappingWorkError(
            f"{location} contains unknown field(s): {', '.join(unknown)}"
        )


def _profile_json(profile: SourceProfile) -> dict[str, object]:
    return {
        "name": profile.name,
        "stage": _canonical_stage(profile.stage),
        "raw_stage": profile.stage,
        "rule": profile.rule,
        "previous_stage": profile.previous_stage,
        "hp": profile.hp,
        "energy_type": profile.energy_type,
        "weakness": profile.weakness,
        "resistance": profile.resistance,
        "retreat": profile.retreat,
        "moves": [list(move) for move in profile.moves],
    }


def _canonical_stage(stage: str | None) -> str | None:
    """Treat source ``Tool`` and competition ``Pokémon Tool`` identically."""
    normalized = normalize_name(stage or "")
    if normalized in {"tool", "pokemon tool"}:
        return "Pokémon Tool"
    return stage.strip() if stage and stage.strip() else None


def _frequency(record: SeenCardRecord) -> int:
    return sum(counts.copies for counts in record.counts_by_source.values())


def _identity_key(record: SeenCardRecord) -> str:
    identity = record.identity
    return "\x1f".join(
        (
            normalize_name(identity.name),
            (identity.set_code or "").strip().casefold(),
            normalize_number(identity.number) or "",
        )
    )


def _context(
    inventory_path: Path,
    csv_path: Path,
    *,
    shard_count: int,
    hydrate_missing: bool,
) -> tuple[str, str, str]:
    inventory_digest = inventory_content_sha256(inventory_path)
    csv_digest = _sha256(csv_path)
    digest = hashlib.sha256()
    digest.update(b"inventory\0")
    digest.update(inventory_digest.encode("ascii"))
    digest.update(b"\0competition-csv\0")
    digest.update(csv_digest.encode("ascii"))
    digest.update(b"\0shards\0")
    digest.update(str(shard_count).encode("ascii"))
    digest.update(b"\0hydrate-missing\0")
    digest.update(b"1" if hydrate_missing else b"0")
    return inventory_digest, csv_digest, digest.hexdigest()


def _load_supplements(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MappingWorkError(
                    f"invalid supplemental profile line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict) or not isinstance(
                value.get("record_key"), str
            ):
                raise MappingWorkError(
                    f"invalid supplemental profile line {line_number}"
                )
            result[value["record_key"]] = value
    return result


def _hydrate_missing_profiles(
    records: Sequence[SeenCardRecord], run_dir: Path
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    supplement_path = run_dir / "supplemental_profiles.jsonl"
    supplements = _load_supplements(supplement_path)
    loader = LimitlessProfileLoader(cache_dir=run_dir / "cache" / "limitless")
    errors: list[dict[str, object]] = []
    candidates = [
        record
        for record in records
        if record.profile is None
        and record.identity.set_code is not None
        and record.identity.number is not None
    ]
    for record in sorted(candidates, key=_identity_key):
        record_key = _identity_key(record)
        if record_key in supplements:
            continue
        identity = record.identity
        try:
            profile = loader(
                RawCard(
                    count=1,
                    name=identity.name,
                    set_code=identity.set_code,
                    number=identity.number,
                )
            )
            if profile is None:
                raise MappingWorkError("profile unavailable")
            if normalize_name(profile.name) != normalize_name(identity.name):
                raise MappingWorkError(
                    f"profile name {profile.name!r} does not match "
                    f"source identity {identity.name!r}"
                )
        except Exception as exc:  # noqa: BLE001 - fail closed and preserve audit trail
            errors.append(
                {
                    "schema_version": WORK_SCHEMA_VERSION,
                    "record_key": record_key,
                    "identity": identity.to_json(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        supplements[record_key] = {
            "schema_version": WORK_SCHEMA_VERSION,
            "record_key": record_key,
            "identity": identity.to_json(),
            "profile": _profile_json(profile),
        }
        _write_jsonl(
            supplement_path,
            (supplements[key] for key in sorted(supplements)),
        )
    return supplements, errors


def _catalog_rows(index: CardIndex) -> Iterable[dict[str, object]]:
    yield {
        "schema_version": WORK_SCHEMA_VERSION,
        "record_type": "header",
        "card_count": len(index.by_id),
    }
    for card_id in sorted(index.by_id):
        info = index.by_id[card_id]
        profile = index.profiles[card_id]
        yield {
            "schema_version": WORK_SCHEMA_VERSION,
            "record_type": "card",
            "card_id": card_id,
            "name": info.name,
            "set_code": info.set_code,
            "number": info.number,
            "stage": _canonical_stage(info.stage),
            "raw_stage": info.stage,
            "rule": profile.rule,
            "previous_stage": profile.previous_stage,
            "is_ace_spec": info.is_ace_spec,
            "is_basic_energy": info.is_basic_energy,
            "hp": profile.hp,
            "energy_type": profile.energy_type,
            "weakness": profile.weakness,
            "resistance": profile.resistance,
            "retreat": profile.retreat,
            "moves": [list(move) for move in profile.moves],
        }


class _Names:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


def _work_units(
    records: Sequence[SeenCardRecord],
    supplements: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    names = {normalize_name(record.identity.name) for record in records}
    connected = _Names(names)
    for record in records:
        supplement = supplements.get(_identity_key(record))
        supplemental_profile = supplement.get("profile") if supplement else None
        previous_stage = (
            record.profile.previous_stage
            if record.profile is not None
            else supplemental_profile.get("previous_stage")
            if isinstance(supplemental_profile, dict)
            else None
        )
        previous = normalize_name(str(previous_stage or ""))
        current = normalize_name(record.identity.name)
        if previous in names:
            connected.union(current, previous)

    grouped: dict[str, list[SeenCardRecord]] = defaultdict(list)
    for record in records:
        grouped[connected.find(normalize_name(record.identity.name))].append(record)

    units: list[dict[str, object]] = []
    for family_name, family_records in grouped.items():
        ordered = sorted(family_records, key=_identity_key)
        frequency = sum(_frequency(record) for record in ordered)
        serialized: list[dict[str, object]] = []
        for record in ordered:
            record_key = _identity_key(record)
            supplement = supplements.get(record_key)
            profile = (
                _profile_json(record.profile)
                if record.profile is not None
                else supplement.get("profile")
                if supplement is not None
                else None
            )
            serialized.append(
                {
                    "record_key": record_key,
                    "identity": record.identity.to_json(),
                    "frequency": _frequency(record),
                    "can_emit_exact_rule": bool(
                        record.identity.set_code and record.identity.number
                    ),
                    "counts_by_source": {
                        source: record.counts_by_source[source].to_json()
                        for source in sorted(record.counts_by_source)
                    },
                    "spellings": [
                        spelling.to_json()
                        for spelling in sorted(
                            record.spellings,
                            key=lambda item: (
                                item.source,
                                item.name,
                                item.set_code or "",
                                item.number or "",
                            ),
                        )
                    ],
                    "examples": [
                        example.to_json()
                        for example in sorted(
                            record.examples,
                            key=lambda item: (
                                item.source,
                                item.archetype,
                                item.url or "",
                            ),
                        )
                    ],
                    "profile": profile,
                    "profile_source": (
                        "inventory"
                        if record.profile is not None
                        else "supplemental"
                        if supplement is not None
                        else "missing"
                    ),
                    "metadata_error": record.metadata_error,
                }
            )
        units.append(
            {
                "unit_id": hashlib.sha256(
                    "\n".join(item["record_key"] for item in serialized).encode()
                ).hexdigest()[:16],
                "family_name": family_name,
                "frequency": frequency,
                "records": serialized,
            }
        )
    return sorted(units, key=lambda unit: (-int(unit["frequency"]), unit["unit_id"]))


def _shard_units(
    units: Sequence[dict[str, object]], shard_count: int
) -> list[list[dict[str, object]]]:
    if shard_count <= 0:
        raise MappingWorkError("--shards must be positive")
    shards: list[list[dict[str, object]]] = [
        [] for _ in range(min(shard_count, max(1, len(units))))
    ]
    weights = [0] * len(shards)
    for unit in units:
        destination = min(
            range(len(shards)), key=lambda offset: (weights[offset], offset)
        )
        shards[destination].append(unit)
        weights[destination] += int(unit["frequency"])
    return shards


def _instructions() -> str:
    return """# Local mapping-agent file contract

No agent may make network requests. Inspect one shard and `competition_catalog.jsonl`.
Emit only positive exact-printing rules; omit every source with no defensible mapping.
Never emit a name fallback. A source without both set and number cannot emit a rule.

`proposals/shard-NNN.json` uses schema 1, kind `mapping_proposals`, and contains
`run_id`, `shard_id`, `shard_sha256`, `proposer`, and `rules`. Each rule contains
`rule_id`, exact `source_name`/`source_set`/`source_number`, optional source gameplay
fields (`source_rule`, `source_stage`, `source_previous_stage`, `family_id`,
`allow_cross_subtype`), and ordered `targets`. Each target contains competition
`card_id`, `expected_name`, `mapping_confidence` (integer 1..5), and `rationale`.

Confidence: 5 gameplay-equivalent; 4 same mechanics/numbers with one narrow edge
case; 3 same core mechanic with a modest numeric/secondary-effect difference; 2
same role with a material effectiveness/timing difference; 1 same deck slot despite
a major card difference. Reject rather than score when the primary role differs,
evidence is incomplete, deck-specific conditions are needed, or the match is a guess.

`reviews/shard-NNN.json` uses kind `mapping_reviews`, repeats the run/shard hashes,
names a reviewer distinct from the proposer, records `proposal_sha256`, and has one
decision per proposed rule. A decision is `{rule_id,status,targets}`. Status is
`approved` or `rejected`; approved targets use the same target shape and are scored
independently. Rejected targets are an empty list.

Receipts are `receipts/shard-NNN.proposer.json` and
`receipts/shard-NNN.reviewer.json`. Each contains schema/kind/run/shard hashes,
`role` (`proposer` or `reviewer`), `agent`, `input_sha256`, `records_considered`,
`positive_rules`, and `output_sha256`; reviewer `positive_rules` is its approved
decision count. Receipts contain aggregate counts only.
"""


def prepare(args: argparse.Namespace) -> int:
    inventory_path = resolve_inventory_path(args.inventory)
    csv_path = Path(args.csv).resolve()
    if args.shards <= 0:
        raise MappingWorkError("--shards must be positive")
    before, csv_digest, run_id = _context(
        inventory_path,
        csv_path,
        shard_count=args.shards,
        hydrate_missing=args.hydrate_missing,
    )
    records = read_inventory(inventory_path)
    problematic = tuple(record for record in records if record.match.card_id is None)
    run_dir = Path(args.work_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("shards", "proposals", "reviews", "receipts"):
        (run_dir / child).mkdir(exist_ok=True)

    existing_run_path = run_dir / "run.json"
    has_agent_work = any(
        path.is_file()
        for directory in ("proposals", "reviews", "receipts")
        for path in (run_dir / directory).glob("*.json")
    )
    if has_agent_work:
        if not existing_run_path.exists():
            raise MappingWorkError("agent outputs exist without run.json")
        existing_run = _run_data(run_dir)
        _verify_prepared_inputs(run_dir, existing_run)
        print(run_dir)
        print("agent work already exists; retained the locked prepared inputs")
        return 0

    supplements = _load_supplements(run_dir / "supplemental_profiles.jsonl")
    errors: list[dict[str, object]] = []
    if args.hydrate_missing:
        supplements, errors = _hydrate_missing_profiles(problematic, run_dir)
        _write_jsonl(run_dir / "fetch_errors.jsonl", errors)

    index = CardIndex(str(csv_path))
    catalog_path = run_dir / "competition_catalog.jsonl"
    catalog_content = b"".join(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
        for value in _catalog_rows(index)
    )
    units = _work_units(problematic, supplements)
    shards = _shard_units(units, args.shards)
    shard_manifest: list[dict[str, object]] = []
    shard_contents: list[tuple[Path, bytes]] = []
    for offset, shard_units in enumerate(shards, 1):
        shard_id = f"shard-{offset:03d}"
        record_count = sum(len(unit["records"]) for unit in shard_units)
        value = {
            "schema_version": WORK_SCHEMA_VERSION,
            "kind": "mapping_shard",
            "run_id": run_id,
            "shard_id": shard_id,
            "catalog": "../competition_catalog.jsonl",
            "record_count": record_count,
            "frequency": sum(int(unit["frequency"]) for unit in shard_units),
            "units": shard_units,
        }
        shard_path = run_dir / "shards" / f"{shard_id}.json"
        content = _json_bytes(value)
        shard_contents.append((shard_path, content))
        shard_manifest.append(
            {
                "shard_id": shard_id,
                "sha256": hashlib.sha256(content).hexdigest(),
                "record_count": record_count,
                "unit_count": len(shard_units),
                "frequency": value["frequency"],
            }
        )

    _atomic_write(catalog_path, catalog_content)
    for shard_path, content in shard_contents:
        _atomic_write(shard_path, content)

    after = inventory_content_sha256(inventory_path)
    if after != before:
        raise MappingWorkError("inventory changed while preparing mapping work")
    run = {
        "schema_version": WORK_SCHEMA_VERSION,
        "run_id": run_id,
        "inventory_path": str(inventory_path.resolve()),
        "inventory_sha256": before,
        "competition_csv_path": str(csv_path),
        "competition_csv_sha256": csv_digest,
        "catalog_sha256": _sha256(catalog_path),
        "inventory_records": len(records),
        "problematic_records": len(problematic),
        "identity_less_records": sum(
            not (record.identity.set_code and record.identity.number)
            for record in problematic
        ),
        "missing_profiles": sum(record.profile is None for record in problematic),
        "supplemental_profiles": len(supplements),
        "fetch_errors": len(errors),
        "shard_count": args.shards,
        "hydrate_missing": args.hydrate_missing,
        "supplemental_profiles_sha256": (
            _sha256(run_dir / "supplemental_profiles.jsonl")
            if (run_dir / "supplemental_profiles.jsonl").exists()
            else None
        ),
        "shards": shard_manifest,
    }
    _write_json(run_dir / "run.json", run)
    _atomic_write(run_dir / "AGENT_INSTRUCTIONS.md", _instructions().encode())
    print(run_dir)
    print(
        f"prepared {len(problematic)} problematic records in {len(shards)} shards; "
        f"{len(supplements)} supplemental profiles, {len(errors)} fetch errors"
    )
    return 0


def _require_string(value: Mapping[str, Any], key: str, location: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise MappingWorkError(f"{location}.{key} must be a non-empty string")
    return item.strip()


def _require_list(value: Mapping[str, Any], key: str, location: str) -> list[Any]:
    item = value.get(key)
    if not isinstance(item, list):
        raise MappingWorkError(f"{location}.{key} must be a list")
    return item


def _optional_string(value: Mapping[str, Any], key: str, location: str) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, str) or not item.strip():
        raise MappingWorkError(f"{location}.{key} must be a non-empty string or null")
    return item.strip()


def _run_data(run_dir: Path) -> dict[str, Any]:
    run = _read_json(run_dir / "run.json")
    if run.get("schema_version") != WORK_SCHEMA_VERSION:
        raise MappingWorkError("unsupported mapping-work schema")
    inventory = Path(_require_string(run, "inventory_path", "run"))
    csv_path = Path(_require_string(run, "competition_csv_path", "run"))
    if inventory_content_sha256(inventory) != run.get("inventory_sha256"):
        raise MappingWorkError("inventory SHA-256 no longer matches run.json")
    if _sha256(csv_path) != run.get("competition_csv_sha256"):
        raise MappingWorkError("competition CSV SHA-256 no longer matches run.json")
    return run


def _verify_prepared_inputs(run_dir: Path, run: Mapping[str, Any]) -> None:
    catalog = run_dir / "competition_catalog.jsonl"
    if not catalog.is_file() or _sha256(catalog) != run.get("catalog_sha256"):
        raise MappingWorkError("competition catalog SHA-256 no longer matches run.json")
    for raw in _require_list(run, "shards", "run"):
        if not isinstance(raw, dict):
            raise MappingWorkError("run.shards entries must be objects")
        shard_id = _require_string(raw, "shard_id", "run.shards")
        expected = _require_string(raw, "sha256", "run.shards")
        shard = run_dir / "shards" / f"{shard_id}.json"
        if not shard.is_file() or _sha256(shard) != expected:
            raise MappingWorkError(f"{shard} SHA-256 no longer matches run.json")
    supplement = run_dir / "supplemental_profiles.jsonl"
    actual_supplement = _sha256(supplement) if supplement.is_file() else None
    if actual_supplement != run.get("supplemental_profiles_sha256"):
        raise MappingWorkError(
            "supplemental profile SHA-256 no longer matches run.json"
        )


def _shard_records(
    shard: Mapping[str, Any],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for unit in _require_list(shard, "units", "shard"):
        if not isinstance(unit, dict):
            raise MappingWorkError("shard unit must be an object")
        for record in _require_list(unit, "records", "shard unit"):
            if not isinstance(record, dict) or not isinstance(
                record.get("identity"), dict
            ):
                raise MappingWorkError("shard record must contain an identity")
            identity = record["identity"]
            name = _require_string(identity, "name", "identity")
            set_code = identity.get("set_code")
            number = identity.get("number")
            key = (
                normalize_name(name),
                str(set_code).strip().casefold() if set_code is not None else "",
                normalize_number(str(number)) if number is not None else "",
            )
            records[key] = record
    return records


def _validate_target(raw: Any, index: CardIndex, location: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise MappingWorkError(f"{location} must be an object")
    _reject_unknown(
        raw,
        {"card_id", "expected_name", "mapping_confidence", "rationale"},
        location,
    )
    card_id = raw.get("card_id")
    confidence = raw.get("mapping_confidence")
    if isinstance(card_id, bool) or not isinstance(card_id, int) or card_id <= 0:
        raise MappingWorkError(f"{location}.card_id must be a positive integer")
    if card_id not in index.by_id:
        raise MappingWorkError(f"{location} targets illegal Card ID {card_id}")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, int)
        or not 1 <= confidence <= 5
    ):
        raise MappingWorkError(
            f"{location}.mapping_confidence must be an integer from 1 to 5"
        )
    expected = _require_string(raw, "expected_name", location)
    actual = index.by_id[card_id].name
    if normalize_name(expected) != normalize_name(actual):
        raise MappingWorkError(
            f"{location} expects {expected!r}, but Card ID {card_id} is {actual!r}"
        )
    return {
        "card_id": card_id,
        "expected_name": actual,
        "mapping_confidence": confidence,
        "rationale": _require_string(raw, "rationale", location),
    }


def _validate_rule(
    raw: Any,
    records: Mapping[tuple[str, str, str], dict[str, Any]],
    index: CardIndex,
    location: str,
) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise MappingWorkError(f"{location} must be an object")
    _reject_unknown(
        raw,
        {
            "rule_id",
            "source_name",
            "source_set",
            "source_number",
            "source_rule",
            "source_stage",
            "source_previous_stage",
            "targets",
            "family_id",
            "allow_cross_subtype",
        },
        location,
    )
    source_name = _require_string(raw, "source_name", location)
    source_set = _require_string(raw, "source_set", location)
    source_number = _require_string(raw, "source_number", location)
    key = (
        normalize_name(source_name),
        source_set.casefold(),
        normalize_number(source_number) or "",
    )
    source = records.get(key)
    if source is None:
        raise MappingWorkError(
            f"{location} source is not an exact printing in its shard"
        )
    targets = [
        _validate_target(target, index, f"{location}.targets[{offset}]")
        for offset, target in enumerate(_require_list(raw, "targets", location))
    ]
    if not targets:
        raise MappingWorkError(f"{location}.targets must contain positive mappings")
    if len({target["card_id"] for target in targets}) != len(targets):
        raise MappingWorkError(f"{location}.targets repeats a Card ID")
    source_profile = source.get("profile")
    source_rule = _optional_string(raw, "source_rule", location)
    if source_rule is None and isinstance(source_profile, dict):
        source_rule = source_profile.get("rule")
    source_is_ace = normalize_name(str(source_rule or "")) == "ace spec"
    source_stage = _optional_string(raw, "source_stage", location)
    if source_stage is None and isinstance(source_profile, dict):
        source_stage = source_profile.get("stage")
    if source_stage is None:
        raise MappingWorkError(f"{location} has no source stage evidence")
    canonical_source_stage = _canonical_stage(str(source_stage))
    allow_cross = raw.get("allow_cross_subtype", False)
    if not isinstance(allow_cross, bool):
        raise MappingWorkError(f"{location}.allow_cross_subtype must be boolean")
    family_id = _optional_string(raw, "family_id", location)
    source_previous_stage = _optional_string(raw, "source_previous_stage", location)
    for target in targets:
        info = index.by_id[int(target["card_id"])]
        if info.is_ace_spec and not source_is_ace:
            raise MappingWorkError(f"{location} maps an ordinary card to ACE SPEC")
        target_stage = _canonical_stage(info.stage)
        if canonical_source_stage != target_stage and not (
            allow_cross
            and normalize_name(str(canonical_source_stage)).endswith("energy")
            and normalize_name(str(target_stage)).endswith("energy")
        ):
            raise MappingWorkError(
                f"{location} crosses subtype {canonical_source_stage!r} -> "
                f"{target_stage!r}"
            )
        source_is_pokemon = normalize_name(str(canonical_source_stage)).endswith(
            "pokemon"
        )
        if (
            source_is_pokemon
            and normalize_name(info.name) != normalize_name(source_name)
            and family_id is None
        ):
            raise MappingWorkError(
                f"{location} cross-species Pokémon target requires family_id"
            )
        if (
            source_is_pokemon
            and normalize_name(info.name) != normalize_name(source_name)
            and normalize_name(str(canonical_source_stage)) != "basic pokemon"
            and source_previous_stage is None
        ):
            raise MappingWorkError(
                f"{location} evolved cross-species target requires "
                "source_previous_stage"
            )
    normalized: dict[str, object] = {
        "rule_id": _require_string(raw, "rule_id", location),
        "source_name": source_name,
        "source_set": source_set,
        "source_number": normalize_number(source_number) or source_number,
        "source_rule": str(source_rule).strip() if source_rule else None,
        "source_stage": canonical_source_stage,
        "source_previous_stage": source_previous_stage,
        "targets": targets,
        "family_id": family_id,
        "allow_cross_subtype": allow_cross,
    }
    return normalized


def _validate_receipt(
    path: Path,
    *,
    run_id: str,
    shard_id: str,
    shard_sha: str,
    role: str,
    output_path: Path,
    record_count: int,
    positive_rules: int,
    agent: str,
    input_sha: str,
) -> None:
    receipt = _read_json(path)
    expected = {
        "schema_version": WORK_SCHEMA_VERSION,
        "kind": "completion_receipt",
        "run_id": run_id,
        "shard_id": shard_id,
        "shard_sha256": shard_sha,
        "role": role,
        "agent": agent,
        "input_sha256": input_sha,
        "records_considered": record_count,
        "positive_rules": positive_rules,
        "output_sha256": _sha256(output_path),
    }
    _reject_unknown(receipt, set(expected), str(path))
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise MappingWorkError(
                f"{path}.{key} is {receipt.get(key)!r}; expected {value!r}"
            )


def _validate_shard_outputs(
    run_dir: Path,
    run: Mapping[str, Any],
    shard_meta: Mapping[str, Any],
    index: CardIndex,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    run_id = _require_string(run, "run_id", "run")
    shard_id = _require_string(shard_meta, "shard_id", "run.shards")
    shard_sha = _require_string(shard_meta, "sha256", "run.shards")
    shard_path = run_dir / "shards" / f"{shard_id}.json"
    if _sha256(shard_path) != shard_sha:
        raise MappingWorkError(f"{shard_path} SHA-256 does not match run.json")
    shard = _read_json(shard_path)
    records = _shard_records(shard)
    proposal_path = run_dir / "proposals" / f"{shard_id}.json"
    review_path = run_dir / "reviews" / f"{shard_id}.json"
    if not proposal_path.exists():
        if review_path.exists():
            raise MappingWorkError(f"{review_path} exists without a proposal")
        return None, None
    proposal = _read_json(proposal_path)
    _reject_unknown(
        proposal,
        {
            "schema_version",
            "kind",
            "run_id",
            "shard_id",
            "shard_sha256",
            "proposer",
            "rules",
        },
        str(proposal_path),
    )
    for key, expected in {
        "schema_version": WORK_SCHEMA_VERSION,
        "kind": "mapping_proposals",
        "run_id": run_id,
        "shard_id": shard_id,
        "shard_sha256": shard_sha,
    }.items():
        if proposal.get(key) != expected:
            raise MappingWorkError(f"{proposal_path}.{key} must be {expected!r}")
    proposer = _require_string(proposal, "proposer", str(proposal_path))
    rules = [
        _validate_rule(raw, records, index, f"{proposal_path}:rules[{offset}]")
        for offset, raw in enumerate(
            _require_list(proposal, "rules", str(proposal_path))
        )
    ]
    if len({rule["rule_id"] for rule in rules}) != len(rules):
        raise MappingWorkError(f"{proposal_path} repeats a rule_id")
    proposal["validated_rules"] = rules
    proposal_sha = _sha256(proposal_path)
    _validate_receipt(
        run_dir / "receipts" / f"{shard_id}.proposer.json",
        run_id=run_id,
        shard_id=shard_id,
        shard_sha=shard_sha,
        role="proposer",
        output_path=proposal_path,
        record_count=int(shard_meta["record_count"]),
        positive_rules=len(rules),
        agent=proposer,
        input_sha=shard_sha,
    )
    if not review_path.exists():
        return proposal, None
    review = _read_json(review_path)
    _reject_unknown(
        review,
        {
            "schema_version",
            "kind",
            "run_id",
            "shard_id",
            "shard_sha256",
            "proposal_sha256",
            "reviewer",
            "decisions",
        },
        str(review_path),
    )
    for key, expected in {
        "schema_version": WORK_SCHEMA_VERSION,
        "kind": "mapping_reviews",
        "run_id": run_id,
        "shard_id": shard_id,
        "shard_sha256": shard_sha,
        "proposal_sha256": proposal_sha,
    }.items():
        if review.get(key) != expected:
            raise MappingWorkError(f"{review_path}.{key} must be {expected!r}")
    reviewer = _require_string(review, "reviewer", str(review_path))
    if normalize_name(reviewer) == normalize_name(proposer):
        raise MappingWorkError(f"{review_path} reviewer must differ from proposer")
    decisions = _require_list(review, "decisions", str(review_path))
    proposal_ids = {str(rule["rule_id"]) for rule in rules}
    normalized_decisions: list[dict[str, object]] = []
    for offset, raw in enumerate(decisions):
        location = f"{review_path}:decisions[{offset}]"
        if not isinstance(raw, dict):
            raise MappingWorkError(f"{location} must be an object")
        _reject_unknown(raw, {"rule_id", "status", "targets"}, location)
        rule_id = _require_string(raw, "rule_id", location)
        status = _require_string(raw, "status", location)
        if status not in {"approved", "rejected"}:
            raise MappingWorkError(f"{location}.status must be approved or rejected")
        targets = _require_list(raw, "targets", location)
        if status == "rejected" and targets:
            raise MappingWorkError(f"{location} rejected decision must have no targets")
        normalized_targets = [
            _validate_target(target, index, f"{location}.targets[{target_offset}]")
            for target_offset, target in enumerate(targets)
        ]
        if status == "approved" and not normalized_targets:
            raise MappingWorkError(f"{location} approved decision needs targets")
        normalized_decisions.append(
            {"rule_id": rule_id, "status": status, "targets": normalized_targets}
        )
    decision_ids = [str(item["rule_id"]) for item in normalized_decisions]
    if set(decision_ids) != proposal_ids or len(decision_ids) != len(proposal_ids):
        raise MappingWorkError(
            f"{review_path} must decide every proposed rule exactly once"
        )
    review["validated_decisions"] = normalized_decisions
    _validate_receipt(
        run_dir / "receipts" / f"{shard_id}.reviewer.json",
        run_id=run_id,
        shard_id=shard_id,
        shard_sha=shard_sha,
        role="reviewer",
        output_path=review_path,
        record_count=int(shard_meta["record_count"]),
        positive_rules=sum(
            item["status"] == "approved" for item in normalized_decisions
        ),
        agent=reviewer,
        input_sha=proposal_sha,
    )
    return proposal, review


def _validated_outputs(
    run_dir: Path,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any] | None, dict[str, Any] | None]]]:
    run = _run_data(run_dir)
    _verify_prepared_inputs(run_dir, run)
    index = CardIndex(_require_string(run, "competition_csv_path", "run"))
    shards = _require_list(run, "shards", "run")
    outputs = []
    rule_ids: set[str] = set()
    source_keys: set[tuple[str, str, str]] = set()
    for shard_meta in shards:
        if not isinstance(shard_meta, dict):
            raise MappingWorkError("run.shards entries must be objects")
        output = _validate_shard_outputs(run_dir, run, shard_meta, index)
        proposal = output[0]
        if proposal is not None:
            for rule in proposal["validated_rules"]:
                rule_id = str(rule["rule_id"])
                source_key = (
                    normalize_name(str(rule["source_name"])),
                    str(rule["source_set"]).casefold(),
                    str(rule["source_number"]),
                )
                if rule_id in rule_ids:
                    raise MappingWorkError(
                        f"duplicate rule_id across shards: {rule_id}"
                    )
                if source_key in source_keys:
                    raise MappingWorkError(
                        f"multiple rules for exact source printing {source_key!r}"
                    )
                rule_ids.add(rule_id)
                source_keys.add(source_key)
        outputs.append(output)
    return run, outputs


def status(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    run, outputs = _validated_outputs(run_dir)
    proposed = sum(proposal is not None for proposal, _ in outputs)
    reviewed = sum(review is not None for _, review in outputs)
    print(f"run:                 {run['run_id']}")
    print(f"problematic records: {run['problematic_records']}")
    print(f"shards:              {len(outputs)}")
    print(f"proposed:            {proposed}")
    print(f"reviewed:            {reviewed}")
    print(f"remaining reviews:   {len(outputs) - reviewed}")
    return 0


def validate(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    _, outputs = _validated_outputs(run_dir)
    complete = sum(review is not None for _, review in outputs)
    print(f"valid: {complete}/{len(outputs)} shards independently reviewed")
    if args.require_complete and complete != len(outputs):
        raise MappingWorkError("mapping work is structurally valid but incomplete")
    return 0


def compile_work(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    run, outputs = _validated_outputs(run_dir)
    accepted: list[dict[str, object]] = []
    rejected_rules: list[dict[str, object]] = []
    failed_families: set[str] = set()
    rejected = 0
    disagreements = 0
    for proposal, review in outputs:
        if proposal is None or review is None:
            continue
        proposals = {str(rule["rule_id"]): rule for rule in proposal["validated_rules"]}
        for decision in review["validated_decisions"]:
            proposed = proposals[str(decision["rule_id"])]
            family_id = proposed.get("family_id")
            if decision["status"] == "rejected":
                rejected += 1
                rejected_rules.append(proposed)
                if isinstance(family_id, str):
                    failed_families.add(family_id)
                continue
            proposed_targets = proposed["targets"]
            reviewed_targets = decision["targets"]
            if [item["card_id"] for item in proposed_targets] != [
                item["card_id"] for item in reviewed_targets
            ]:
                disagreements += 1
                if isinstance(family_id, str):
                    failed_families.add(family_id)
                continue
            if any(
                abs(
                    int(proposed_target["mapping_confidence"])
                    - int(reviewed_target["mapping_confidence"])
                )
                > 1
                for proposed_target, reviewed_target in zip(
                    proposed_targets, reviewed_targets, strict=True
                )
            ):
                disagreements += 1
                if isinstance(family_id, str):
                    failed_families.add(family_id)
                continue
            targets = []
            for proposed_target, reviewed_target in zip(
                proposed_targets, reviewed_targets, strict=True
            ):
                targets.append(
                    {
                        "card_id": proposed_target["card_id"],
                        "expected_name": proposed_target["expected_name"],
                        "mapping_confidence": min(
                            int(proposed_target["mapping_confidence"]),
                            int(reviewed_target["mapping_confidence"]),
                        ),
                        "rationale": reviewed_target["rationale"],
                    }
                )
            accepted.append(
                {
                    **{
                        key: value
                        for key, value in proposed.items()
                        if key != "targets"
                    },
                    "targets": targets,
                }
            )
    accepted = [
        rule
        for rule in accepted
        if not (
            isinstance(rule.get("family_id"), str)
            and rule["family_id"] in failed_families
        )
    ]
    accepted.sort(key=lambda rule: str(rule["rule_id"]))
    rejected_rules = [
        rule
        for rule in rejected_rules
        if not (
            isinstance(rule.get("family_id"), str)
            and rule["family_id"] in failed_families
        )
    ]
    rejected_rules.sort(key=lambda rule: str(rule["rule_id"]))
    compiled = {"schema_version": RULE_SCHEMA_VERSION, "rules": accepted}
    rejected_compiled = {
        "schema_version": RULE_SCHEMA_VERSION,
        "rules": rejected_rules,
    }
    summary = {
        "schema_version": WORK_SCHEMA_VERSION,
        "run_id": run["run_id"],
        "reviewed_shards": sum(review is not None for _, review in outputs),
        "total_shards": len(outputs),
        "agreed_rules": len(accepted),
        "rejected_rules": len(rejected_rules),
        "reviewer_rejections": rejected,
        "disagreements": disagreements,
        "atomic_family_exclusions": len(failed_families),
    }
    _write_json(run_dir / "compiled_rules.json", compiled)
    _write_json(run_dir / "rejected_by_review_agent.json", rejected_compiled)
    _write_json(run_dir / "compile_summary.json", summary)
    print(
        f"compiled {len(accepted)} accepted rules; {rejected} agent-rejected, "
        f"{disagreements} disagreements"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scraper.mapping_work",
        description="Prepare and validate local-only explicit mapping-agent work.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    prepare_parser.add_argument("--csv", type=Path, default=Path(CardIndex().csv_path))
    prepare_parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    prepare_parser.add_argument("--shards", type=int, default=DEFAULT_SHARD_COUNT)
    prepare_parser.add_argument("--hydrate-missing", action="store_true")
    prepare_parser.set_defaults(handler=prepare)

    for name, handler in (("status", status), ("validate", validate)):
        command = commands.add_parser(name)
        command.add_argument("run_dir", type=Path)
        if name == "validate":
            command.add_argument("--require-complete", action="store_true")
        command.set_defaults(handler=handler)

    compile_parser = commands.add_parser("compile")
    compile_parser.add_argument("run_dir", type=Path)
    compile_parser.set_defaults(handler=compile_work)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (MappingWorkError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
