"""Tests for the explicit card-mapping rule data layer."""

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
    source_set: str | None = "PAL",
    source_number: str | None = "185",
    source_rule: str | None = None,
    source_stage: str | None = "Supporter",
    source_previous_stage: str | None = None,
    targets: list[dict[str, object]] | None = None,
    family_id: str | None = None,
    allow_cross_subtype: bool = False,
) -> dict[str, object]:
    target_values = targets or [{"card_id": 1213, "expected_name": "Judge"}]
    normalized_targets = [
        {
            "mapping_confidence": 4,
            "rationale": "Reviewed deterministic substitution.",
            **target,
        }
        for target in target_values
    ]
    return {
        "rule_id": rule_id,
        "source_name": source_name,
        "source_set": source_set,
        "source_number": source_number,
        "source_rule": source_rule,
        "source_stage": source_stage,
        "source_previous_stage": source_previous_stage,
        "targets": normalized_targets,
        "family_id": family_id,
        "allow_cross_subtype": allow_cross_subtype,
    }


def _write_fragment(
    path: Path, rules: list[dict[str, object]], version: int | bool = 2
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


def test_exact_rule_normalizes_identity_and_unseen_printings_fail_closed(tmp_path):
    _write_fragment(
        tmp_path / "exact.json",
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
    assert rules.candidates_for(other_printing) == ()
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
                    {
                        "card_id": 1213,
                        "expected_name": "Judge",
                        "mapping_confidence": 5,
                        "rationale": "Primary target rationale.",
                    },
                    {
                        "card_id": 1181,
                        "expected_name": "Billy & O'Nare",
                        "mapping_confidence": 2,
                        "rationale": "Fallback target rationale.",
                    },
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
    assert [candidate.rationale for candidate in candidates] == [
        "Primary target rationale.",
        "Fallback target rationale.",
    ]
    assert all(candidate.confidence is None for candidate in candidates)
    assert [candidate.mapping_confidence for candidate in candidates] == [5, 2]
    assert "confidence" not in candidates[0].to_json()
    assert candidates[0].to_json()["mapping_confidence"] == 5


def test_minimum_mapping_confidence_filters_targets(tmp_path):
    _write_fragment(
        tmp_path / "rules.json",
        [
            _rule(
                targets=[
                    {
                        "card_id": 1213,
                        "expected_name": "Judge",
                        "mapping_confidence": 4,
                    },
                    {
                        "card_id": 1181,
                        "expected_name": "Billy & O'Nare",
                        "mapping_confidence": 2,
                    },
                ]
            )
        ],
    )
    card = RawCard(1, "Legacy Draw", "PAL", "185", "trainer")

    all_targets = MappingRuleSet.load(
        tmp_path, CardIndex(), minimum_mapping_confidence=1
    )
    high_confidence = MappingRuleSet.load(
        tmp_path, CardIndex(), minimum_mapping_confidence=3
    )

    assert [swap.target_id for swap in all_targets.candidates_for(card)] == [1213, 1181]
    assert [swap.target_id for swap in high_confidence.candidates_for(card)] == [1213]


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


def test_rejected_mapping_file_requires_explicit_opt_in(tmp_path):
    _write_fragment(
        tmp_path / "rejected_by_review_agent.json",
        [_rule("rejected-rule")],
    )
    card = RawCard(1, "Legacy Draw", "PAL", "185", "trainer")

    default_rules = MappingRuleSet.load(tmp_path, CardIndex())
    opted_in_rules = MappingRuleSet.load(
        tmp_path, CardIndex(), use_rejected_mappings=True
    )

    assert default_rules.candidates_for(card) == ()
    assert [swap.target_id for swap in opted_in_rules.candidates_for(card)] == [1213]


@pytest.mark.parametrize(
    ("rules", "message"),
    [
        ([_rule(), _rule()], "duplicate mapping rule_id"),
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
            _rule(source_set="PAL", source_number=None),
            "source_number must be a non-empty string",
        ),
    ],
)
def test_stale_and_incomplete_rules_are_rejected(tmp_path, rule, message):
    _write_fragment(tmp_path / "rules.json", [rule])

    with pytest.raises(MappingRuleError, match=message):
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
                source_set="BCR",
                source_number="138",
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


def test_tool_source_matches_pokemon_tool_competition_subtype(tmp_path):
    _write_fragment(
        tmp_path / "rules.json",
        [
            _rule(
                source_name="Old Tool",
                source_stage="Tool",
                targets=[{"card_id": 1159, "expected_name": "Hero's Cape"}],
            )
        ],
    )

    with pytest.raises(MappingRuleError, match="ordinary source to ACE SPEC"):
        MappingRuleSet.load(tmp_path, CardIndex())

    rule = _rule(
        source_name="Old Tool",
        source_rule="ACE SPEC",
        source_stage="Tool",
        targets=[{"card_id": 1159, "expected_name": "Hero's Cape"}],
    )
    _write_fragment(tmp_path / "rules.json", [rule])

    assert MappingRuleSet.load(tmp_path, CardIndex()).rules


@pytest.mark.parametrize("version", [3, True])
def test_schema_version_is_required_and_exact(tmp_path, version):
    _write_fragment(tmp_path / "rules.json", [], version=version)

    with pytest.raises(MappingRuleError, match="schema_version .*; expected 2"):
        MappingRuleSet.load(tmp_path, CardIndex())


def test_empty_v1_fragment_is_safe_but_nonempty_v1_fails_closed(tmp_path):
    path = tmp_path / "rules.json"
    _write_fragment(path, [], version=1)
    assert MappingRuleSet.load(tmp_path, CardIndex()).rules == ()

    _write_fragment(path, [_rule()], version=1)
    with pytest.raises(MappingRuleError, match="only an empty schema v1"):
        MappingRuleSet.load(tmp_path, CardIndex())


@pytest.mark.parametrize("mapping_confidence", [0, 6, 3.5, True, None])
def test_mapping_confidence_must_be_integer_one_through_five(
    tmp_path, mapping_confidence
):
    target = {
        "card_id": 1213,
        "expected_name": "Judge",
        "mapping_confidence": mapping_confidence,
    }
    _write_fragment(tmp_path / "rules.json", [_rule(targets=[target])])

    with pytest.raises(MappingRuleError, match="integer from 1 to 5"):
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
    base = _rule()
    _write_fragment(path, [base])
    first = MappingRuleSet.load(tmp_path, CardIndex()).fingerprint
    assert MappingRuleSet.load(tmp_path, CardIndex()).fingerprint == first

    base["targets"] = [{
        "card_id": 1181,
        "expected_name": "Billy & O'Nare",
        "mapping_confidence": 3,
        "rationale": "A different target.",
    }]
    _write_fragment(path, [base])
    retargeted = MappingRuleSet.load(tmp_path, CardIndex()).fingerprint
    assert retargeted != first

    rerationalized = _rule(
        targets=[{"card_id": 1181, "expected_name": "Billy & O'Nare"}],
    )
    _write_fragment(path, [rerationalized])
    assert MappingRuleSet.load(tmp_path, CardIndex()).fingerprint != retargeted
