"""Versioned rules for deterministic card mapping."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .card_index import CardIndex, normalize_name, normalize_number
from .models import CardSwap, RawCard

SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
ACE_SPEC_RULE = "ace spec"
REJECTED_RULES_FILENAME = "rejected_by_review_agent.json"


class MappingRuleError(ValueError):
    """Raised when mapping-rule data is malformed, conflicting, or stale."""


class SetCanonicalizer(Protocol):
    """Normalize a source code or expansion name to one canonical set code."""

    def __call__(self, value: str | None) -> str | None: ...


@dataclass(frozen=True)
class MappingTarget:
    """One competition-card target in a rule's fallback order."""

    card_id: int
    expected_name: str
    mapping_confidence: int
    rationale: str


@dataclass(frozen=True)
class MappingRule:
    """An exact-printing mapping rule."""

    rule_id: str
    source_name: str
    source_set: str
    source_number: str
    source_rule: str | None
    source_stage: str | None
    source_previous_stage: str | None
    targets: tuple[MappingTarget, ...]
    family_id: str | None = None
    allow_cross_subtype: bool = False

    @property
    def normalized_name(self) -> str:
        """Return the canonical source name used for rule lookup."""
        return normalize_name(self.source_name)

    @property
    def source_key(self) -> tuple[str, str, str]:
        """Return the canonical source-printing key."""
        return (
            self.normalized_name,
            _normalize_set(self.source_set) or "",
            normalize_number(self.source_number) or "",
        )


class MappingRuleSet:
    """Validated mapping rules indexed by exact source printing."""

    def __init__(
        self,
        rules: tuple[MappingRule, ...],
        index: CardIndex,
        set_canonicalizer: SetCanonicalizer | None = None,
        *,
        minimum_mapping_confidence: int = 1,
    ):
        """Validate and index rules against the competition card pool."""
        self.index = index
        self.rules = rules
        self.set_canonicalizer = set_canonicalizer
        if not 1 <= minimum_mapping_confidence <= 5:
            raise MappingRuleError("minimum mapping confidence must be from 1 to 5")
        self.minimum_mapping_confidence = minimum_mapping_confidence
        self._exact: dict[tuple[str, str, str], MappingRule] = {}
        self._validate_and_index()

    @classmethod
    def load(
        cls,
        directory: str | Path,
        index: CardIndex,
        set_canonicalizer: SetCanonicalizer | None = None,
        *,
        use_rejected_mappings: bool = False,
        minimum_mapping_confidence: int = 1,
    ) -> MappingRuleSet:
        """Load every JSON fragment below ``directory`` in stable path order.

        A missing directory and a directory without JSON fragments are both valid
        empty rule sets.
        """
        root = Path(directory)
        if not root.exists():
            return cls(
                (),
                index,
                set_canonicalizer,
                minimum_mapping_confidence=minimum_mapping_confidence,
            )
        if not root.is_dir():
            raise MappingRuleError(f"mapping rule path is not a directory: {root}")

        rules: list[MappingRule] = []
        paths = sorted(
            root.rglob("*.json"), key=lambda path: path.relative_to(root).as_posix()
        )
        for path in paths:
            if path.name == REJECTED_RULES_FILENAME and not use_rejected_mappings:
                continue
            rules.extend(_load_fragment(path))
        return cls(
            tuple(rules),
            index,
            set_canonicalizer,
            minimum_mapping_confidence=minimum_mapping_confidence,
        )

    @property
    def fingerprint(self) -> str:
        """Return a stable digest used to identify compatible gap checkpoints."""
        payload = json.dumps(
            [asdict(rule) for rule in self.rules],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def rule_for(self, card: RawCard) -> MappingRule | None:
        """Return the exact-printing rule for ``card``."""
        name = normalize_name(card.name)
        source_set = self._canonical_set(card.set_code)
        source_number = normalize_number(card.number)
        if source_set is None or source_number is None:
            return None
        return self._exact.get((name, source_set, source_number))

    def _canonical_set(self, value: str | None) -> str | None:
        if self.set_canonicalizer is not None:
            value = self.set_canonicalizer(value) or value
        return _normalize_set(value)

    def candidates_for(self, card: RawCard) -> tuple[CardSwap, ...]:
        """Return deterministic, ordered candidates from the selected rule."""
        rule = self.rule_for(card)
        if rule is None:
            return ()
        return tuple(
            CardSwap(
                source_name=card.name,
                source_set=card.set_code,
                source_number=normalize_number(card.number),
                count=max(0, card.count),
                target_id=target.card_id,
                target_name=self.index.by_id[target.card_id].name,
                kind="mapping",
                confidence=None,
                rationale=target.rationale,
                mapping_confidence=target.mapping_confidence,
                rule_id=rule.rule_id,
                family_id=rule.family_id,
                source_stage=rule.source_stage,
                source_previous_stage=rule.source_previous_stage,
            )
            for target in rule.targets
            if target.mapping_confidence >= self.minimum_mapping_confidence
        )

    def _validate_and_index(self) -> None:
        rule_ids: set[str] = set()
        for rule in self.rules:
            if rule.rule_id in rule_ids:
                raise MappingRuleError(f"duplicate mapping rule_id {rule.rule_id!r}")
            rule_ids.add(rule.rule_id)

            if not rule.normalized_name:
                raise MappingRuleError(
                    f"rule {rule.rule_id!r} has an empty source_name"
                )
            if not rule.source_set.strip() or not rule.source_number.strip():
                raise MappingRuleError(
                    f"rule {rule.rule_id!r} must specify source_set and source_number"
                )
            if not rule.targets:
                raise MappingRuleError(f"rule {rule.rule_id!r} has no targets")
            if rule.source_stage is None:
                raise MappingRuleError(
                    f"rule {rule.rule_id!r} must declare source_stage"
                )
            if rule.allow_cross_subtype and _category(rule.source_stage) != "energy":
                raise MappingRuleError(
                    f"rule {rule.rule_id!r} may allow cross-subtype mapping only "
                    "for Energy cards"
                )

            seen_targets: set[int] = set()
            source_is_ace = normalize_name(rule.source_rule or "") == ACE_SPEC_RULE
            for target in rule.targets:
                if target.card_id in seen_targets:
                    raise MappingRuleError(
                        f"rule {rule.rule_id!r} repeats target Card ID {target.card_id}"
                    )
                seen_targets.add(target.card_id)
                info = self.index.by_id.get(target.card_id)
                if info is None:
                    raise MappingRuleError(
                        f"rule {rule.rule_id!r} targets unknown competition Card ID "
                        f"{target.card_id}"
                    )
                if normalize_name(info.name) != normalize_name(target.expected_name):
                    raise MappingRuleError(
                        f"rule {rule.rule_id!r} expects Card ID {target.card_id} to be "
                        f"{target.expected_name!r}, but the competition index names it "
                        f"{info.name!r}"
                    )
                if info.is_ace_spec and not source_is_ace:
                    raise MappingRuleError(
                        f"rule {rule.rule_id!r} maps an ordinary source to ACE SPEC "
                        f"Card ID {target.card_id}; source_rule must be 'ACE SPEC'"
                    )
                if rule.source_stage is not None:
                    source_category = _category(rule.source_stage)
                    target_category = _category(info.stage)
                    same_subtype = _canonical_stage(info.stage) == _canonical_stage(
                        rule.source_stage
                    )
                    if source_category != target_category or (
                        not same_subtype
                        and not (
                            source_category == "energy" and rule.allow_cross_subtype
                        )
                    ):
                        raise MappingRuleError(
                            f"rule {rule.rule_id!r} declares source_stage "
                            f"{rule.source_stage!r}, but target Card ID {target.card_id} "
                            f"has stage {info.stage!r}"
                        )
                    if (
                        source_category == "pokemon"
                        and normalize_name(info.name) != rule.normalized_name
                        and rule.family_id is None
                    ):
                        raise MappingRuleError(
                            f"cross-species Pokémon rule {rule.rule_id!r} requires "
                            "family_id"
                        )
                    if (
                        source_category == "pokemon"
                        and normalize_name(info.name) != rule.normalized_name
                        and normalize_name(rule.source_stage or "") != "basic pokemon"
                        and rule.source_previous_stage is None
                    ):
                        raise MappingRuleError(
                            f"evolved cross-species rule {rule.rule_id!r} requires "
                            "source_previous_stage"
                        )

            name = rule.normalized_name
            source_set = self._canonical_set(rule.source_set)
            source_number = normalize_number(rule.source_number)
            assert source_set is not None and source_number is not None
            key = (name, source_set, source_number)
            if key in self._exact:
                other = self._exact[key]
                raise MappingRuleError(
                    f"conflicting exact rules {other.rule_id!r} and "
                    f"{rule.rule_id!r} for {rule.source_name!r} "
                    f"{rule.source_set} {rule.source_number}"
                )
            self._exact[key] = rule


def _normalize_set(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().casefold()
    return normalized or None


def _load_fragment(path: Path) -> list[MappingRule]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MappingRuleError(f"cannot read mapping rules from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise MappingRuleError(f"mapping fragment {path} must be a JSON object")
    version = data.get("schema_version")
    if (
        not isinstance(version, bool)
        and version == LEGACY_SCHEMA_VERSION
        and data.get("rules") == []
    ):
        return []
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != SCHEMA_VERSION
    ):
        raise MappingRuleError(
            f"mapping fragment {path} has schema_version {version!r}; "
            f"expected {SCHEMA_VERSION} (only an empty schema v1 fragment is accepted)"
        )
    raw_rules = data.get("rules")
    if not isinstance(raw_rules, list):
        raise MappingRuleError(f"mapping fragment {path} must contain a rules list")
    return [_parse_rule(raw, path, offset) for offset, raw in enumerate(raw_rules)]


def _parse_rule(raw: Any, path: Path, offset: int) -> MappingRule:
    location = f"{path}:rules[{offset}]"
    if not isinstance(raw, dict):
        raise MappingRuleError(f"{location} must be an object")
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

    targets = _required_list(raw, "targets", location)
    return MappingRule(
        rule_id=_required_string(raw, "rule_id", location),
        source_name=_required_string(raw, "source_name", location),
        source_set=_required_string(raw, "source_set", location),
        source_number=_required_string(raw, "source_number", location),
        source_rule=_optional_string(raw, "source_rule", location),
        source_stage=_optional_string(raw, "source_stage", location),
        source_previous_stage=_optional_string(raw, "source_previous_stage", location),
        targets=tuple(
            _parse_target(target, f"{location}.targets[{target_offset}]")
            for target_offset, target in enumerate(targets)
        ),
        family_id=_optional_string(raw, "family_id", location),
        allow_cross_subtype=_optional_bool(
            raw, "allow_cross_subtype", location, default=False
        ),
    )


def _parse_target(raw: Any, location: str) -> MappingTarget:
    if not isinstance(raw, dict):
        raise MappingRuleError(f"{location} must be an object")
    _reject_unknown(
        raw,
        {"card_id", "expected_name", "mapping_confidence", "rationale"},
        location,
    )
    card_id = raw.get("card_id")
    if isinstance(card_id, bool) or not isinstance(card_id, int) or card_id <= 0:
        raise MappingRuleError(f"{location}.card_id must be a positive integer")
    mapping_confidence = raw.get("mapping_confidence")
    if (
        isinstance(mapping_confidence, bool)
        or not isinstance(mapping_confidence, int)
        or not 1 <= mapping_confidence <= 5
    ):
        raise MappingRuleError(
            f"{location}.mapping_confidence must be an integer from 1 to 5"
        )
    return MappingTarget(
        card_id=card_id,
        expected_name=_required_string(raw, "expected_name", location),
        mapping_confidence=mapping_confidence,
        rationale=_required_string(raw, "rationale", location),
    )


def _required_string(data: dict[str, Any], key: str, location: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MappingRuleError(f"{location}.{key} must be a non-empty string")
    return value.strip()


def _optional_string(data: dict[str, Any], key: str, location: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise MappingRuleError(f"{location}.{key} must be a non-empty string or null")
    return value.strip()


def _optional_bool(
    data: dict[str, Any], key: str, location: str, *, default: bool
) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise MappingRuleError(f"{location}.{key} must be a boolean")
    return value


def _category(stage: str | None) -> str | None:
    normalized = _canonical_stage(stage)
    if normalized.endswith(" pokemon"):
        return "pokemon"
    if normalized.endswith("energy"):
        return "energy"
    if normalized:
        return "trainer"
    return None


def _canonical_stage(stage: str | None) -> str:
    """Normalize equivalent source and competition subtype labels."""
    normalized = normalize_name(stage or "")
    if normalized in {"tool", "pokemon tool"}:
        return "pokemon tool"
    return normalized


def _required_list(data: dict[str, Any], key: str, location: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise MappingRuleError(f"{location}.{key} must be a list")
    return value


def _reject_unknown(data: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise MappingRuleError(
            f"{location} contains unknown field(s): {', '.join(unknown)}"
        )
