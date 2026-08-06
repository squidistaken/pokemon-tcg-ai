# ruff: noqa: SLF001 - focused tests exercise deterministic work-file helpers

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import cast

import pytest

from scraper import mapping_work
from scraper.card_index import CardIndex
from scraper.card_swapper import SourceProfile
from scraper.inventory import (
    CanonicalPrinting,
    OrdinaryMatch,
    SeenCardRecord,
    SourceCounts,
    inventory_content_sha256,
    write_inventory,
)


def _profile(
    name: str,
    *,
    stage: str = "Supporter",
    previous_stage: str | None = None,
) -> SourceProfile:
    return SourceProfile(
        name=name,
        stage=stage,
        rule="n/a",
        previous_stage=previous_stage,
        hp=80 if "Pokémon" in stage else None,
        energy_type="Fire" if "Pokémon" in stage else None,
        weakness="Water" if "Pokémon" in stage else None,
        resistance=None,
        retreat=1 if "Pokémon" in stage else None,
        moves=(("Attack", "{R}", "20", ""),),
    )


def _record(
    name: str,
    set_code: str | None,
    number: str | None,
    *,
    card_id: int | None = None,
    profile: SourceProfile | None = None,
    copies: int = 1,
) -> SeenCardRecord:
    return SeenCardRecord(
        identity=CanonicalPrinting(name, set_code, number),
        match=OrdinaryMatch(
            "exact" if card_id is not None else "unresolved",
            card_id,
            name if card_id is not None else None,
            None,
            (),
        ),
        counts_by_source={"limitless": SourceCounts(1, 1, copies)},
        profile=profile,
    )


def _unresolved(
    name: str,
    set_code: str | None,
    number: str | None,
    *,
    stage: str = "Supporter",
    previous_stage: str | None = None,
    copies: int = 1,
) -> SeenCardRecord:
    return _record(
        name,
        set_code,
        number,
        profile=_profile(name, stage=stage, previous_stage=previous_stage),
        copies=copies,
    )


def _prepare(
    root: Path,
    records: list[SeenCardRecord],
    *,
    inventory_name: str = "seen_cards.jsonl.gz",
    shards: int = 1,
    hydrate_missing: bool = False,
) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    inventory = root / inventory_name
    write_inventory(inventory, records)
    if not inventory_name.endswith(".gz"):
        inventory.write_bytes(gzip.decompress(inventory.read_bytes()))
    competition = root / "cards.csv"
    competition.write_bytes(Path(CardIndex().csv_path).read_bytes())
    work_root = root / "work"
    args = argparse.Namespace(
        inventory=inventory,
        csv=competition,
        work_root=work_root,
        shards=shards,
        hydrate_missing=hydrate_missing,
    )
    assert mapping_work.prepare(args) == 0
    (run_dir,) = tuple(path for path in work_root.iterdir() if path.is_dir())
    return run_dir, inventory, competition


def _target(confidence: int, *, card_id: int = 1213) -> dict[str, object]:
    names = {
        1213: "Judge",
        1154: "Team Rocket’s Hypnotizer",
        1155: "Survival Brace",
    }
    return {
        "card_id": card_id,
        "expected_name": names[card_id],
        "mapping_confidence": confidence,
        "rationale": f"Independent confidence {confidence} rationale.",
    }


def _rule(
    rule_id: str,
    source: SeenCardRecord,
    confidence: int = 4,
    *,
    source_stage: str = "Supporter",
    card_id: int = 1213,
) -> dict[str, object]:
    return {
        "rule_id": rule_id,
        "source_name": source.identity.name,
        "source_set": source.identity.set_code,
        "source_number": source.identity.number,
        "source_stage": source_stage,
        "targets": [_target(confidence, card_id=card_id)],
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_outputs(
    run_dir: Path,
    rules: list[dict[str, object]],
    decisions: list[dict[str, object]],
    *,
    proposer: str = "proposer-one",
    reviewer: str = "reviewer-two",
    proposal_sha256: str | None = None,
) -> None:
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    shard = run["shards"][0]
    shard_id = shard["shard_id"]
    proposal_path = run_dir / "proposals" / f"{shard_id}.json"
    proposal = {
        "schema_version": 1,
        "kind": "mapping_proposals",
        "run_id": run["run_id"],
        "shard_id": shard_id,
        "shard_sha256": shard["sha256"],
        "proposer": proposer,
        "rules": rules,
    }
    _write_json(proposal_path, proposal)
    proposal_sha = mapping_work._sha256(proposal_path)
    _write_json(
        run_dir / "receipts" / f"{shard_id}.proposer.json",
        {
            "schema_version": 1,
            "kind": "completion_receipt",
            "run_id": run["run_id"],
            "shard_id": shard_id,
            "shard_sha256": shard["sha256"],
            "role": "proposer",
            "agent": proposer,
            "input_sha256": shard["sha256"],
            "records_considered": shard["record_count"],
            "positive_rules": len(rules),
            "output_sha256": proposal_sha,
        },
    )
    review_path = run_dir / "reviews" / f"{shard_id}.json"
    review = {
        "schema_version": 1,
        "kind": "mapping_reviews",
        "run_id": run["run_id"],
        "shard_id": shard_id,
        "shard_sha256": shard["sha256"],
        "proposal_sha256": proposal_sha256 or proposal_sha,
        "reviewer": reviewer,
        "decisions": decisions,
    }
    _write_json(review_path, review)
    _write_json(
        run_dir / "receipts" / f"{shard_id}.reviewer.json",
        {
            "schema_version": 1,
            "kind": "completion_receipt",
            "run_id": run["run_id"],
            "shard_id": shard_id,
            "shard_sha256": shard["sha256"],
            "role": "reviewer",
            "agent": reviewer,
            "input_sha256": proposal_sha,
            "records_considered": shard["record_count"],
            "positive_rules": sum(
                decision["status"] == "approved" for decision in decisions
            ),
            "output_sha256": mapping_work._sha256(review_path),
        },
    )


def _approved(rule_id: str, confidence: int) -> dict[str, object]:
    return {
        "rule_id": rule_id,
        "status": "approved",
        "targets": [_target(confidence)],
    }


def _rejected(rule_id: str) -> dict[str, object]:
    return {"rule_id": rule_id, "status": "rejected", "targets": []}


def test_plain_and_gzip_prepare_are_equivalent_and_problematic_only(tmp_path: Path):
    unresolved = _unresolved("Legacy Draw", "PAL", "185", copies=8)
    resolved = _record(
        "Judge",
        "SVI",
        "176",
        card_id=1213,
        profile=_profile("Judge"),
    )
    compressed_dir, compressed_inventory, _ = _prepare(
        tmp_path / "compressed", [resolved, unresolved], shards=3
    )
    plain_dir, plain_inventory, _ = _prepare(
        tmp_path / "plain",
        [resolved, unresolved],
        inventory_name="seen_cards.jsonl",
        shards=3,
    )

    compressed_run = json.loads(
        (compressed_dir / "run.json").read_text(encoding="utf-8")
    )
    plain_run = json.loads((plain_dir / "run.json").read_text(encoding="utf-8"))
    compressed_shards = sorted((compressed_dir / "shards").glob("*.json"))
    plain_shards = sorted((plain_dir / "shards").glob("*.json"))

    assert inventory_content_sha256(compressed_inventory) == inventory_content_sha256(
        plain_inventory
    )
    assert compressed_run["run_id"] == plain_run["run_id"]
    assert compressed_run["inventory_records"] == 2
    assert compressed_run["problematic_records"] == 1
    assert [path.read_bytes() for path in compressed_shards] == [
        path.read_bytes() for path in plain_shards
    ]
    shard = json.loads(compressed_shards[0].read_text(encoding="utf-8"))
    identities = [
        record["identity"]["name"]
        for unit in shard["units"]
        for record in unit["records"]
    ]
    assert identities == ["Legacy Draw"]


def test_prepare_is_deterministic_and_does_not_modify_locked_inventory(tmp_path: Path):
    records = [
        _unresolved("Alpha", "SET", "1", copies=20),
        _unresolved("Beta", "SET", "2", copies=5),
        _unresolved("Gamma", "SET", "3", copies=1),
    ]
    first, inventory, _ = _prepare(tmp_path / "first", records, shards=2)
    original_bytes = inventory.read_bytes()
    second, _, _ = _prepare(tmp_path / "second", records, shards=2)

    assert inventory.read_bytes() == original_bytes
    assert json.loads((first / "run.json").read_text())["inventory_sha256"] == (
        inventory_content_sha256(inventory)
    )
    assert [path.read_bytes() for path in sorted((first / "shards").glob("*"))] == [
        path.read_bytes() for path in sorted((second / "shards").glob("*"))
    ]


def test_supplemental_previous_stage_keeps_evolution_family_indivisible():
    basic = _unresolved("Charmander", "OBF", "26", stage="Basic Pokémon")
    evolved = _record("Charmeleon", "OBF", "27")
    supplements = {
        mapping_work._identity_key(evolved): {
            "profile": mapping_work._profile_json(
                _profile(
                    "Charmeleon",
                    stage="Stage 1 Pokémon",
                    previous_stage="Charmander",
                )
            )
        }
    }

    units = mapping_work._work_units((basic, evolved), supplements)

    assert len(units) == 1
    records = cast(list[dict[str, object]], units[0]["records"])
    identities = [cast(dict[str, object], record["identity"]) for record in records]
    assert {identity["name"] for identity in identities} == {
        "Charmander",
        "Charmeleon",
    }
    assert {record["profile_source"] for record in records} == {
        "inventory",
        "supplemental",
    }


def test_hydration_is_exact_only_cached_and_never_uses_identity_less_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    exact = _record("Iono", "PAL", "185")
    identity_less = _record("Mystery Card", None, None)
    loaded: list[tuple[str, str | None, str | None]] = []

    class ProfileLoader:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, card):
            loaded.append((card.name, card.set_code, card.number))
            return _profile(card.name)

    monkeypatch.setattr(mapping_work, "LimitlessProfileLoader", ProfileLoader)
    supplements, errors = mapping_work._hydrate_missing_profiles(
        (identity_less, exact), tmp_path
    )

    assert errors == []
    assert loaded == [("Iono", "PAL", "185")]
    assert list(supplements) == [mapping_work._identity_key(exact)]

    class NoNetworkLoader:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, _card):
            raise AssertionError("cached hydration must not fetch again")

    monkeypatch.setattr(mapping_work, "LimitlessProfileLoader", NoNetworkLoader)
    cached, cached_errors = mapping_work._hydrate_missing_profiles(
        (identity_less, exact), tmp_path
    )
    assert cached == supplements
    assert cached_errors == []


def test_hydration_rejects_wrong_card_page(monkeypatch, tmp_path: Path):
    source = _record("Iono", "PAL", "185")

    class WrongProfileLoader:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, _card):
            return _profile("Judge")

    monkeypatch.setattr(mapping_work, "LimitlessProfileLoader", WrongProfileLoader)

    supplements, errors = mapping_work._hydrate_missing_profiles((source,), tmp_path)

    assert supplements == {}
    assert len(errors) == 1
    assert "does not match source identity" in str(errors[0]["error"])
    assert not (tmp_path / "supplemental_profiles.jsonl").exists()


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (_target(4, card_id=1213) | {"card_id": 999999}, "illegal Card ID"),
        (_target(4) | {"mapping_confidence": 0}, "integer from 1 to 5"),
        (_target(4) | {"mapping_confidence": True}, "integer from 1 to 5"),
        (_target(4) | {"expected_name": "Not Judge"}, "Card ID 1213 is 'Judge'"),
    ],
)
def test_target_validation_rejects_illegal_targets_and_confidence(target, message):
    with pytest.raises(mapping_work.MappingWorkError, match=message):
        mapping_work._validate_target(target, CardIndex(), "target")


def test_tool_alias_is_accepted_but_ordinary_to_ace_spec_is_not():
    source = _unresolved("Old Tool", "OLD", "1", stage="Tool")
    records = {
        (
            "old tool",
            "old",
            "1",
        ): {
            "identity": source.identity.to_json(),
            "profile": mapping_work._profile_json(source.profile),
        }
    }
    rule = _rule("tool-rule", source, card_id=1154, source_stage="Tool")

    validated = mapping_work._validate_rule(rule, records, CardIndex(), "rule")

    assert validated["source_stage"] == "Pokémon Tool"
    with pytest.raises(
        mapping_work.MappingWorkError, match="ordinary card to ACE SPEC"
    ):
        mapping_work._validate_rule(
            _rule("ace-rule", source, card_id=1155, source_stage="Tool"),
            records,
            CardIndex(),
            "rule",
        )


def test_identity_less_source_cannot_emit_an_exact_rule(tmp_path: Path):
    source = _unresolved("Unknown Printing", None, None)
    run_dir, _, _ = _prepare(tmp_path, [source])
    _write_outputs(
        run_dir,
        [
            {
                **_rule("invented-identity", source),
                "source_set": "MADEUP",
                "source_number": "1",
            }
        ],
        [_approved("invented-identity", 4)],
    )

    with pytest.raises(mapping_work.MappingWorkError, match="not an exact printing"):
        mapping_work._validated_outputs(run_dir)


def test_empty_positive_proposal_is_valid_sparse_output(tmp_path: Path):
    run_dir, _, _ = _prepare(
        tmp_path, [_unresolved("No Defensible Mapping", "OLD", "1")]
    )
    _write_outputs(run_dir, [], [])

    run, outputs = mapping_work._validated_outputs(run_dir)
    assert run["problematic_records"] == 1
    proposal, review = outputs[0]
    assert proposal is not None
    assert review is not None
    assert proposal["validated_rules"] == []
    assert review["validated_decisions"] == []

    assert mapping_work.compile_work(argparse.Namespace(run_dir=run_dir)) == 0
    compiled = json.loads((run_dir / "compiled_rules.json").read_text(encoding="utf-8"))
    assert compiled == {"schema_version": 2, "rules": []}


def test_reviewer_must_be_independent(tmp_path: Path):
    source = _unresolved("Legacy Draw", "OLD", "1")
    run_dir, _, _ = _prepare(tmp_path, [source])
    _write_outputs(
        run_dir,
        [_rule("legacy-draw", source)],
        [_approved("legacy-draw", 4)],
        proposer="same-agent",
        reviewer="same-agent",
    )

    with pytest.raises(mapping_work.MappingWorkError, match="differ from proposer"):
        mapping_work._validated_outputs(run_dir)


def test_compile_omits_rejections_and_disagreements(
    tmp_path: Path,
):
    agreed = _unresolved("Agreed", "OLD", "1")
    rejected = _unresolved("Rejected", "OLD", "2")
    disagreement = _unresolved("Disagreement", "OLD", "3")
    run_dir, _, _ = _prepare(tmp_path, [agreed, rejected, disagreement])
    _write_outputs(
        run_dir,
        [
            _rule("agreed", agreed, 4),
            _rule("rejected", rejected, 5),
            _rule("disagreement", disagreement, 5),
        ],
        [
            _approved("agreed", 3),
            _rejected("rejected"),
            _approved("disagreement", 3),
        ],
    )

    assert mapping_work.compile_work(argparse.Namespace(run_dir=run_dir)) == 0
    compiled = json.loads((run_dir / "compiled_rules.json").read_text(encoding="utf-8"))
    rejected_compiled = json.loads(
        (run_dir / "rejected_by_review_agent.json").read_text(encoding="utf-8")
    )
    summary = json.loads((run_dir / "compile_summary.json").read_text(encoding="utf-8"))

    assert [rule["rule_id"] for rule in compiled["rules"]] == ["agreed"]
    assert [rule["rule_id"] for rule in rejected_compiled["rules"]] == ["rejected"]
    (agreed_rule,) = compiled["rules"]
    assert agreed_rule["targets"][0]["mapping_confidence"] == 3
    assert summary["agreed_rules"] == 1
    assert summary["rejected_rules"] == 1
    assert summary["reviewer_rejections"] == 1
    assert summary["disagreements"] == 1


def test_stale_inventory_and_proposal_hashes_are_rejected(tmp_path: Path):
    source = _unresolved("Legacy Draw", "OLD", "1")
    inventory_run, inventory, _ = _prepare(tmp_path / "inventory", [source])
    inventory.write_bytes(gzip.decompress(inventory.read_bytes()) + b"\n")
    with pytest.raises(mapping_work.MappingWorkError, match="inventory SHA-256"):
        mapping_work._validated_outputs(inventory_run)

    proposal_run, _, _ = _prepare(tmp_path / "proposal", [source])
    _write_outputs(
        proposal_run,
        [_rule("legacy-draw", source)],
        [_approved("legacy-draw", 4)],
        proposal_sha256="0" * 64,
    )
    with pytest.raises(mapping_work.MappingWorkError, match="proposal_sha256"):
        mapping_work._validated_outputs(proposal_run)


def test_prepare_never_overwrites_locked_inputs_after_agent_work(tmp_path: Path):
    source = _unresolved("Legacy Draw", "OLD", "1")
    run_dir, inventory, competition = _prepare(tmp_path, [source])
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    shard_path = run_dir / "shards" / "shard-001.json"
    shard_path.write_bytes(shard_path.read_bytes() + b"tampered")
    tampered = shard_path.read_bytes()
    _write_json(run_dir / "proposals" / "started.json", {})

    with pytest.raises(mapping_work.MappingWorkError, match="shard-001.*SHA-256"):
        mapping_work.prepare(
            argparse.Namespace(
                inventory=inventory,
                csv=competition,
                work_root=tmp_path / "work",
                shards=run["shard_count"],
                hydrate_missing=run["hydrate_missing"],
            )
        )

    assert shard_path.read_bytes() == tampered


def test_run_context_changes_with_sharding_and_hydration(tmp_path: Path):
    inventory = tmp_path / "seen_cards.jsonl"
    inventory.write_text("{}\n", encoding="utf-8")
    competition = tmp_path / "cards.csv"
    competition.write_text("Card ID\n", encoding="utf-8")

    base = mapping_work._context(
        inventory, competition, shard_count=24, hydrate_missing=False
    )[2]
    rebalanced = mapping_work._context(
        inventory, competition, shard_count=12, hydrate_missing=False
    )[2]
    hydrated = mapping_work._context(
        inventory, competition, shard_count=24, hydrate_missing=True
    )[2]

    assert len({base, rebalanced, hydrated}) == 3


def test_positive_target_schema_rejects_explicit_no_mapping_fields():
    target = {
        "card_id": 1213,
        "expected_name": "Judge",
        "mapping_confidence": 3,
        "rationale": "Same primary hand-disruption role.",
        "unmapped": False,
    }

    with pytest.raises(mapping_work.MappingWorkError, match="unknown.*unmapped"):
        mapping_work._validate_target(target, CardIndex(), "target")
