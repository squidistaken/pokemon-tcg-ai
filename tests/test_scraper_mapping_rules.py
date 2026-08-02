"""Tests for the reviewed explicit card-mapping rule data layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scraper.card_index import CardIndex
from scraper.mapping_rules import MappingRuleError, MappingRuleSet
from scraper.models import RawCard


def _rule(
    rule_id: str = "legacy-draw-fallback",
    *,
    source_name: str = "Legacy Draw",
    source_set: str | None = None,
    source_number: str | None = None,
    source_rule: str | None = None,
    source_stage: str | None = "Supporter",
    source_previous_stage: str | None = None,
    targets: list[dict[str, object]] | None = None,
    active: bool = True,
    status: str = "approved",
    reviewer: str | None = "Stef",
    reviewed_at: str | None = "2026-08-02",
    family_id: str | None = None,
    allow_cross_subtype: bool = False,
) -> dict[str, object]:
    return {
        "rule_id": rule_id,
        "source_name": source_name,
        "source_set": source_set,
        "source_number": source_number,
        "source_rule": source_rule,
        "source_stage": source_stage,
        "source_previous_stage": source_previous_stage,
        "targets": targets or [{"card_id": 1213, "expected_name": "Judge"}],
        "active": active,
        "review": {
            "status": status,
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
        },
        "rationale": "Reviewed deterministic substitution.",
        "family_id": family_id,
        "allow_cross_subtype": allow_cross_subtype,
    }


def _write_fragment(
    path: Path, rules: list[dict[str, object]], version: int | bool = 1
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": version, "rules": rules}), encoding="utf-8"
    )


def test_missing_and_empty_directories_have_no_candidates(tmp_path):
    index = CardIndex()

    missing = MappingRuleSet.load(tmp_path / "missing", index)
    empty = MappingRuleSet.load(tmp_path, index)

    card = RawCard(2, "Legacy Draw", "PAL", "185", "trainer")
    assert missing.candidates_for(card) == ()
    assert empty.candidates_for(card) == ()


def test_exact_rule_precedes_name_fallback_and_normalizes_identity(tmp_path):
    _write_fragment(
        tmp_path / "b-name.json",
        [_rule(targets=[{"card_id": 1213, "expected_name": "Judge"}])],
    )
    _write_fragment(
        tmp_path / "a-exact.json",
        [
            _rule(
                "legacy-draw-pal-185",
                source_set="PAL",
                source_number="0185",
                targets=[{"card_id": 927, "expected_name": "Charmeleon"}],
                family_id="example-family",
                source_stage="Stage 1 Pokémon",
                source_previous_stage="Charmander",
            )
        ],
    )
    rules = MappingRuleSet.load(tmp_path, CardIndex())

    exact_card = RawCard(2, "Légacy Draw", " pal ", "185", "trainer")
    other_printing = RawCard(1, "Legacy Draw", "PAF", "80", "trainer")

    exact_candidates = rules.candidates_for(exact_card)
    exact_rule = rules.rule_for(exact_card)

    assert [candidate.target_id for candidate in exact_candidates] == [927]
    assert [
        candidate.target_id for candidate in rules.candidates_for(other_printing)
    ] == [1213]
    assert exact_rule is not None
    assert exact_rule.family_id == "example-family"
    assert exact_candidates[0].rule_id == "legacy-draw-pal-185"
    assert exact_candidates[0].family_id == "example-family"
    assert exact_candidates[0].source_previous_stage == "Charmander"


def test_targets_preserve_declared_fallback_order_and_provenance(tmp_path):
    _write_fragment(
        tmp_path / "rules.json",
        [
            _rule(
                targets=[
                    {"card_id": 1213, "expected_name": "Judge"},
                    {"card_id": 1181, "expected_name": "Billy & O'Nare"},
                ]
            )
        ],
    )
    rules = MappingRuleSet.load(tmp_path, CardIndex())
    card = RawCard(3, "Legacy Draw", "PAL", "0185", "trainer")

    candidates = rules.candidates_for(card)

    assert [candidate.target_id for candidate in candidates] == [1213, 1181]
    assert [candidate.target_name for candidate in candidates] == [
        "Judge",
        "Billy & O'Nare",
    ]
    assert all(candidate.kind == "mapping" for candidate in candidates)
    assert all(candidate.count == 3 for candidate in candidates)
    assert all(candidate.source_number == "185" for candidate in candidates)
    assert all(
        candidate.rationale == "Reviewed deterministic substitution."
        for candidate in candidates
    )


def test_exact_rule_uses_same_set_alias_for_limitless_and_bulbapedia(tmp_path):
    _write_fragment(
        tmp_path / "rules.json",
        [_rule(source_set="OBF", source_number="27")],
    )
    aliases = lambda value: "OBF" if value == "Obsidian Flames" else value
    rules = MappingRuleSet.load(tmp_path, CardIndex(), aliases)

    limitless = rules.candidates_for(RawCard(1, "Legacy Draw", "OBF", "27"))
    bulbapedia = rules.candidates_for(
        RawCard(1, "Legacy Draw", "Obsidian Flames", "27")
    )

    assert [candidate.target_id for candidate in limitless] == [1213]
    assert [candidate.target_id for candidate in bulbapedia] == [1213]


def test_inactive_exact_rule_blocks_a_name_fallback(tmp_path):
    _write_fragment(
        tmp_path / "rules.json",
        [
            _rule(),
            _rule(
                "rejected-printing",
                source_set="PAL",
                source_number="185",
                active=False,
                status="rejected",
                reviewer="Teun",
            ),
        ],
    )
    rules = MappingRuleSet.load(tmp_path, CardIndex())

    assert rules.candidates_for(RawCard(1, "Legacy Draw", "PAL", "185")) == ()
    assert rules.candidates_for(RawCard(1, "Legacy Draw", "PAF", "80"))


@pytest.mark.parametrize(
    ("rules", "message"),
    [
        ([_rule(), _rule()], "duplicate mapping rule_id"),
        (
            [_rule(), _rule("other")],
            "conflicting name fallback rules",
        ),
        (
            [
                _rule("one", source_set="PAL", source_number="185"),
                _rule("two", source_set="pal", source_number="0185"),
            ],
            "conflicting exact rules",
        ),
        (
            [
                _rule(
                    targets=[
                        {"card_id": 1213, "expected_name": "Judge"},
                        {"card_id": 1213, "expected_name": "Judge"},
                    ]
                )
            ],
            "repeats target Card ID 1213",
        ),
    ],
)
def test_duplicates_and_conflicts_are_rejected(tmp_path, rules, message):
    _write_fragment(tmp_path / "rules.json", rules)

    with pytest.raises(MappingRuleError, match=message):
        MappingRuleSet.load(tmp_path, CardIndex())


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        (
            _rule(targets=[{"card_id": 999999, "expected_name": "Missing"}]),
            "unknown competition Card ID",
        ),
        (
            _rule(targets=[{"card_id": 1213, "expected_name": "Not Judge"}]),
            "competition index names it 'Judge'",
        ),
        (
            _rule(active=True, status="pending"),
            "requires an approved review",
        ),
        (
            _rule(active=True, reviewer="agent-7"),
            "dated Stef or Teun review",
        ),
        (
            _rule(active=True, reviewed_at=None),
            "dated Stef or Teun review",
        ),
        (
            _rule(source_set="PAL"),
            "must specify both source_set and source_number",
        ),
    ],
)
def test_stale_unreviewed_and_incomplete_rules_are_rejected(tmp_path, rule, message):
    _write_fragment(tmp_path / "rules.json", [rule])

    with pytest.raises(MappingRuleError, match=message):
        MappingRuleSet.load(tmp_path, CardIndex())


@pytest.mark.parametrize("reviewed_at", ["tomorrow", "2026-02-30", "20260802"])
def test_active_rule_requires_a_real_iso_review_date(tmp_path, reviewed_at):
    _write_fragment(tmp_path / "rules.json", [_rule(reviewed_at=reviewed_at)])

    with pytest.raises(MappingRuleError, match="valid YYYY-MM-DD"):
        MappingRuleSet.load(tmp_path, CardIndex())


def test_ordinary_source_cannot_target_ace_spec(tmp_path):
    target = [{"card_id": 1088, "expected_name": "Prime Catcher"}]
    _write_fragment(tmp_path / "rules.json", [_rule(targets=target)])

    with pytest.raises(MappingRuleError, match="ordinary source to ACE SPEC"):
        MappingRuleSet.load(tmp_path, CardIndex())


def test_explicit_ace_spec_source_may_target_ace_spec(tmp_path):
    target = [{"card_id": 1088, "expected_name": "Prime Catcher"}]
    _write_fragment(
        tmp_path / "rules.json",
        [
            _rule(
                source_name="Old Computer",
                source_rule="ACE SPEC",
                source_stage="Item",
                targets=target,
            )
        ],
    )
    rules = MappingRuleSet.load(tmp_path, CardIndex())

    candidates = rules.candidates_for(RawCard(1, "Old Computer", "BCR", "138"))

    assert [candidate.target_id for candidate in candidates] == [1088]


def test_family_target_must_preserve_declared_stage(tmp_path):
    _write_fragment(
        tmp_path / "rules.json",
        [
            _rule(
                family_id="evolution-family",
                source_stage="Basic Pokémon",
                targets=[{"card_id": 927, "expected_name": "Charmeleon"}],
            )
        ],
    )

    with pytest.raises(MappingRuleError, match="declares source_stage 'Basic Pokémon'"):
        MappingRuleSet.load(tmp_path, CardIndex())


def test_cross_species_pokemon_rule_requires_family_id(tmp_path):
    _write_fragment(
        tmp_path / "rules.json",
        [
            _rule(
                source_name="Different Lizard",
                source_stage="Stage 1 Pokémon",
                targets=[{"card_id": 927, "expected_name": "Charmeleon"}],
            )
        ],
    )

    with pytest.raises(MappingRuleError, match="requires family_id"):
        MappingRuleSet.load(tmp_path, CardIndex())


def test_evolved_cross_species_rule_requires_previous_stage_metadata(tmp_path):
    _write_fragment(
        tmp_path / "rules.json",
        [
            _rule(
                source_name="Different Lizard",
                source_stage="Stage 1 Pokémon",
                targets=[{"card_id": 927, "expected_name": "Charmeleon"}],
                family_id="lizard-family",
                source_previous_stage=None,
            )
        ],
    )

    with pytest.raises(MappingRuleError, match="source_previous_stage"):
        MappingRuleSet.load(tmp_path, CardIndex())


def test_unknown_rule_field_is_rejected_instead_of_disabling_a_guard(tmp_path):
    rule = _rule()
    rule["source_previous_stgae"] = "Typo"
    _write_fragment(tmp_path / "rules.json", [rule])

    with pytest.raises(MappingRuleError, match="unknown field.*source_previous_stgae"):
        MappingRuleSet.load(tmp_path, CardIndex())


def test_cross_subtype_energy_requires_explicit_rule_flag(tmp_path):
    rule = _rule(
        source_name="Old Basic Energy",
        source_stage="Basic Energy",
        targets=[{"card_id": 9, "expected_name": "Boomerang Energy"}],
    )
    _write_fragment(tmp_path / "rules.json", [rule])

    with pytest.raises(MappingRuleError, match="has stage 'Special Energy'"):
        MappingRuleSet.load(tmp_path, CardIndex())

    rule["allow_cross_subtype"] = True
    _write_fragment(tmp_path / "rules.json", [rule])

    assert MappingRuleSet.load(tmp_path, CardIndex()).rules


@pytest.mark.parametrize("version", [2, True])
def test_schema_version_is_required_and_exact(tmp_path, version):
    _write_fragment(tmp_path / "rules.json", [], version=version)

    with pytest.raises(MappingRuleError, match="schema_version .*; expected 1"):
        MappingRuleSet.load(tmp_path, CardIndex())


def test_nested_fragments_are_loaded_in_stable_path_order(tmp_path):
    _write_fragment(
        tmp_path / "z" / "second.json",
        [_rule("second", source_name="Second Card")],
    )
    _write_fragment(
        tmp_path / "a" / "first.json",
        [_rule("first", source_name="First Card")],
    )

    rules = MappingRuleSet.load(tmp_path, CardIndex())

    assert [rule.rule_id for rule in rules.rules] == ["first", "second"]


def test_fingerprint_is_stable_and_tracks_material_rule_changes(tmp_path):
    path = tmp_path / "rules.json"
    base = _rule(active=False, status="pending")
    _write_fragment(path, [base])
    first = MappingRuleSet.load(tmp_path, CardIndex()).fingerprint
    assert MappingRuleSet.load(tmp_path, CardIndex()).fingerprint == first

    active = _rule()
    _write_fragment(path, [active])
    activated = MappingRuleSet.load(tmp_path, CardIndex()).fingerprint
    assert activated != first

    active["targets"] = [{"card_id": 1181, "expected_name": "Billy & O'Nare"}]
    _write_fragment(path, [active])
    retargeted = MappingRuleSet.load(tmp_path, CardIndex()).fingerprint
    assert retargeted != activated

    reviewed = _rule(
        reviewer="Teun",
        targets=[{"card_id": 1181, "expected_name": "Billy & O'Nare"}],
    )
    _write_fragment(path, [reviewed])
    assert MappingRuleSet.load(tmp_path, CardIndex()).fingerprint != retargeted
