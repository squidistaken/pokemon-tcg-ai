"""Versioned, reviewed rules for deterministic card mapping."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from .card_index import CardIndex, normalize_name, normalize_number
from .models import CardSwap, RawCard

SCHEMA_VERSION = 1
APPROVED_REVIEW_STATUS = "approved"
APPROVED_REVIEWERS = {"stef", "stef timmermans", "teun"}
ACE_SPEC_RULE = "ace spec"
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


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


@dataclass(frozen=True)
class MappingReview:
    """Human review state attached to a mapping rule."""

    status: str
    reviewer: str | None = None
    reviewed_at: str | None = None


@dataclass(frozen=True)
class MappingRule:
    """An exact-printing rule or a name-level fallback rule."""

    rule_id: str
    source_name: str
    source_set: str | None
    source_number: str | None
    source_rule: str | None
    source_stage: str | None
    source_previous_stage: str | None
    targets: tuple[MappingTarget, ...]
    active: bool
    review: MappingReview
    rationale: str
    family_id: str | None = None
    allow_cross_subtype: bool = False

    @property
    def normalized_name(self) -> str:
        """Return the canonical source name used for rule lookup."""
        return normalize_name(self.source_name)

    @property
    def source_key(self) -> tuple[str, str | None, str | None]:
        """Return the canonical source-printing key."""
        return (
            self.normalized_name,
            _normalize_set(self.source_set),
            normalize_number(self.source_number),
        )

    @property
    def is_name_fallback(self) -> bool:
        """Return whether this rule applies to any printing with its name."""
        return self.source_set is None and self.source_number is None


class MappingRuleSet:
    """Validated mapping rules indexed by exact printing and source name."""

    def __init__(
        self,
        rules: tuple[MappingRule, ...],
        index: CardIndex,
        set_canonicalizer: SetCanonicalizer | None = None,
    ):
        """Validate and index rules against the competition card pool."""
        self.index = index
        self.rules = rules
        self.set_canonicalizer = set_canonicalizer
        self._exact: dict[tuple[str, str, str], MappingRule] = {}
        self._by_name: dict[str, MappingRule] = {}
        self._validate_and_index()

    @classmethod
    def load(
        cls,
        directory: str | Path,
        index: CardIndex,
        set_canonicalizer: SetCanonicalizer | None = None,
    ) -> MappingRuleSet:
        """Load every JSON fragment below ``directory`` in stable path order.

        A missing directory and a directory without JSON fragments are both valid
        empty rule sets.
        """
        root = Path(directory)
        if not root.exists():
            return cls((), index, set_canonicalizer)
        if not root.is_dir():
            raise MappingRuleError(f"mapping rule path is not a directory: {root}")

        rules: list[MappingRule] = []
        paths = sorted(
            root.rglob("*.json"), key=lambda path: path.relative_to(root).as_posix()
        )
        for path in paths:
            rules.extend(_load_fragment(path))
        return cls(tuple(rules), index, set_canonicalizer)

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
        """Return the most specific rule for ``card``, including inactive rules.

        An exact rule takes precedence over a name fallback. This also lets an
        inactive exact rule deliberately prevent a broad fallback from being used
        for a known-incompatible printing.
        """
        name = normalize_name(card.name)
        source_set = self._canonical_set(card.set_code)
        source_number = normalize_number(card.number)
        if source_set is not None and source_number is not None:
            exact = self._exact.get((name, source_set, source_number))
            if exact is not None:
                return exact
        return self._by_name.get(name)

    def _canonical_set(self, value: str | None) -> str | None:
        if self.set_canonicalizer is not None:
            value = self.set_canonicalizer(value) or value
        return _normalize_set(value)

    def candidates_for(self, card: RawCard) -> tuple[CardSwap, ...]:
        """Return deterministic, ordered candidates from the selected active rule."""
        rule = self.rule_for(card)
        if rule is None or not rule.active:
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
                confidence=1.0,
                rationale=rule.rationale,
                rule_id=rule.rule_id,
                family_id=rule.family_id,
                source_stage=rule.source_stage,
                source_previous_stage=rule.source_previous_stage,
            )
            for target in rule.targets
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
            if (rule.source_set is None) != (rule.source_number is None):
                raise MappingRuleError(
                    f"rule {rule.rule_id!r} must specify both source_set and "
                    "source_number, or neither for a name fallback"
                )
            if not rule.targets:
                raise MappingRuleError(f"rule {rule.rule_id!r} has no targets")
            if not rule.rationale.strip():
                raise MappingRuleError(f"rule {rule.rule_id!r} has no rationale")
            if rule.active and (
                normalize_name(rule.review.status) != APPROVED_REVIEW_STATUS
                or normalize_name(rule.review.reviewer or "") not in APPROVED_REVIEWERS
                or not rule.review.reviewed_at
            ):
                raise MappingRuleError(
                    f"active rule {rule.rule_id!r} requires an approved review "
                    "with a dated Stef or Teun review"
                )
            if rule.active:
                reviewed_at = rule.review.reviewed_at
                try:
                    if reviewed_at is None or not _ISO_DATE_RE.fullmatch(reviewed_at):
                        raise ValueError
                    date.fromisoformat(reviewed_at)
                except ValueError as exc:
                    raise MappingRuleError(
                        f"active rule {rule.rule_id!r} review date must be a valid "
                        "YYYY-MM-DD date"
                    ) from exc
            if rule.active and rule.source_stage is None:
                raise MappingRuleError(
                    f"active rule {rule.rule_id!r} must declare source_stage"
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
                    same_subtype = normalize_name(info.stage) == normalize_name(
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
            if rule.is_name_fallback:
                if name in self._by_name:
                    other = self._by_name[name]
                    raise MappingRuleError(
                        f"conflicting name fallback rules {other.rule_id!r} and "
                        f"{rule.rule_id!r} for {rule.source_name!r}"
                    )
                self._by_name[name] = rule
            else:
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
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != SCHEMA_VERSION
    ):
        raise MappingRuleError(
            f"mapping fragment {path} has schema_version {version!r}; "
            f"expected {SCHEMA_VERSION}"
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
            "active",
            "review",
            "rationale",
            "family_id",
            "allow_cross_subtype",
        },
        location,
    )

    targets = _required_list(raw, "targets", location)
    review = _required_dict(raw, "review", location)
    _reject_unknown(review, {"status", "reviewer", "reviewed_at"}, f"{location}.review")
    return MappingRule(
        rule_id=_required_string(raw, "rule_id", location),
        source_name=_required_string(raw, "source_name", location),
        source_set=_optional_string(raw, "source_set", location),
        source_number=_optional_string(raw, "source_number", location),
        source_rule=_optional_string(raw, "source_rule", location),
        source_stage=_optional_string(raw, "source_stage", location),
        source_previous_stage=_optional_string(raw, "source_previous_stage", location),
        targets=tuple(
            _parse_target(target, f"{location}.targets[{target_offset}]")
            for target_offset, target in enumerate(targets)
        ),
        active=_required_bool(raw, "active", location),
        review=MappingReview(
            status=_required_string(review, "status", f"{location}.review"),
            reviewer=_optional_string(review, "reviewer", f"{location}.review"),
            reviewed_at=_optional_string(review, "reviewed_at", f"{location}.review"),
        ),
        rationale=_required_string(raw, "rationale", location),
        family_id=_optional_string(raw, "family_id", location),
        allow_cross_subtype=_optional_bool(
            raw, "allow_cross_subtype", location, default=False
        ),
    )


def _parse_target(raw: Any, location: str) -> MappingTarget:
    if not isinstance(raw, dict):
        raise MappingRuleError(f"{location} must be an object")
    _reject_unknown(raw, {"card_id", "expected_name"}, location)
    card_id = raw.get("card_id")
    if isinstance(card_id, bool) or not isinstance(card_id, int) or card_id <= 0:
        raise MappingRuleError(f"{location}.card_id must be a positive integer")
    return MappingTarget(
        card_id=card_id,
        expected_name=_required_string(raw, "expected_name", location),
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


def _required_bool(data: dict[str, Any], key: str, location: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise MappingRuleError(f"{location}.{key} must be a boolean")
    return value


def _optional_bool(
    data: dict[str, Any], key: str, location: str, *, default: bool
) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise MappingRuleError(f"{location}.{key} must be a boolean")
    return value


def _category(stage: str | None) -> str | None:
    normalized = normalize_name(stage or "")
    if "pokemon" in normalized:
        return "pokemon"
    if normalized.endswith("energy"):
        return "energy"
    if normalized:
        return "trainer"
    return None


def _required_list(data: dict[str, Any], key: str, location: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise MappingRuleError(f"{location}.{key} must be a list")
    return value


def _required_dict(data: dict[str, Any], key: str, location: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise MappingRuleError(f"{location}.{key} must be an object")
    return value


def _reject_unknown(data: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise MappingRuleError(
            f"{location} contains unknown field(s): {', '.join(unknown)}"
        )
