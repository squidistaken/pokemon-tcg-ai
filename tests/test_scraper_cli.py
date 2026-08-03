"""Card-swap command-line controls."""

from __future__ import annotations

import gzip
import json

import pytest

from scraper.__main__ import build_parser, main
from scraper.card_index import CardIndex
from scraper.card_swapper import CardSwapper, MappingCardSwapper
from scraper.inventory import read_inventory
from scraper.mapping_rules import MappingRuleError
from scraper.models import RawCard, RawDeck
from scraper.sources import NETWORK_SOURCES, SOURCES


def test_card_swapping_can_be_disabled():
    args = build_parser().parse_args(["--disable-card-swap"])

    assert args.disable_card_swap is True


def test_rejected_mappings_are_opt_in():
    default_args = build_parser().parse_args([])
    opted_in_args = build_parser().parse_args(["--use-rejected-mappings"])

    assert default_args.use_rejected_mappings is False
    assert opted_in_args.use_rejected_mappings is True


def test_production_mapping_confidence_defaults_to_every_tier():
    args = build_parser().parse_args([])

    assert args.minimum_mapping_confidence == 1


def test_all_means_every_network_source():
    assert list(NETWORK_SOURCES) == ["limitless", "bulbapedia"]
    assert "text" in SOURCES
    assert "text" not in NETWORK_SOURCES


def test_card_swapper_requires_resolve_implementation():
    class IncompleteSwapper(CardSwapper):
        pass

    assert IncompleteSwapper.__abstractmethods__ == frozenset({"resolve"})

    assert MappingCardSwapper(CardIndex()).resolve(RawCard(1, "Missing")) == ()


def test_all_strategies_fetch_once_and_create_separate_manifests(
    tmp_path, monkeypatch
):
    class FakeSource:
        calls = 0

        def iter_decks(self, **_kwargs):
            type(self).calls += 1
            yield RawDeck(
                source="fake",
                archetype="missing",
                cards=[RawCard(1, "Entirely Missing")],
            )

    monkeypatch.setitem(SOURCES, "fake", FakeSource)

    assert (
        main(
            [
                "--source",
                "fake",
                "--card-swap-strategy",
                "all",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )

    assert FakeSource.calls == 1
    for folder in ("mapping-resolved", "heuristic-resolved"):
        manifest = json.loads((tmp_path / folder / "manifest.json").read_text())
        assert manifest["schema_version"] == 3
        assert manifest["decks"] == {}
    (gap_path,) = tmp_path.glob("mapping-gaps-*.jsonl.gz")
    (gap,) = read_inventory(gap_path)
    assert gap.identity.name == "entirely missing"


def test_all_strategies_receive_the_identical_fetched_deck(
    tmp_path, monkeypatch
):
    class FakeSource:
        @staticmethod
        def iter_decks(**_kwargs):
            yield RawDeck("fake-shared", "shared")

    seen: list[int] = []

    def record(raw, *_args, **_kwargs):
        seen.append(id(raw))

    monkeypatch.setitem(SOURCES, "fake-shared", FakeSource)
    monkeypatch.setattr("scraper.__main__.process_deck", record)

    assert (
        main(
            [
                "--source",
                "fake-shared",
                "--card-swap-strategy",
                "all",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )

    assert len(seen) == 2
    assert seen[0] == seen[1]


def test_disabled_mapping_does_not_write_a_gap_report(tmp_path, monkeypatch):
    class FakeSource:
        calls = 0

        @staticmethod
        def iter_decks(**_kwargs):
            FakeSource.calls += 1
            yield RawDeck("fake-disabled", "missing", [RawCard(1, "Missing")])

    monkeypatch.setitem(SOURCES, "fake-disabled", FakeSource)

    assert (
        main(
            [
                "--source",
                "fake-disabled",
                "--card-swap-strategy",
                "mapping",
                "--disable-card-swap",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )

    assert FakeSource.calls == 1
    assert list(tmp_path.glob("mapping-gaps-*.jsonl.gz")) == []


def test_partial_source_failure_writes_completed_work_and_returns_nonzero(
    tmp_path, monkeypatch
):
    class PartialSource:
        @staticmethod
        def iter_decks(**_kwargs):
            yield RawDeck("partial", "missing", [RawCard(1, "Missing")])
            raise RuntimeError("remote failed")

    processed: list[RawDeck] = []
    monkeypatch.setitem(SOURCES, "partial", PartialSource)
    monkeypatch.setattr(
        "scraper.__main__.process_deck",
        lambda raw, *_args, **_kwargs: processed.append(raw),
    )

    assert main(["--source", "partial", "--out", str(tmp_path)]) == 1
    assert [deck.archetype for deck in processed] == ["missing"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["decks"] == {}


def _gap_checkpoint_label(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.loads(next(handle))["checkpoint_label"]


def test_mapping_gap_report_refuses_to_append_to_an_existing_file(
    tmp_path, monkeypatch
):
    class GapSource:
        @staticmethod
        def iter_decks(**_kwargs):
            yield RawDeck(
                "gap-source",
                "gap",
                [RawCard(2, "Unmapped Card")],
                external_ids={"deck": "stable"},
            )

    monkeypatch.setitem(SOURCES, "gap-source", GapSource)
    rules = tmp_path / "rules"
    rules.mkdir()
    gap = tmp_path / "gaps.jsonl.gz"
    args = [
        "--source", "gap-source", "--card-swap-strategy", "mapping",
        "--card-swap-map", str(rules), "--mapping-gap-out", str(gap),
        "--out", str(tmp_path / "decks"),
    ]

    assert main(args) == 0
    original = gap.read_bytes()
    assert main(args) == 2
    assert gap.read_bytes() == original
    (record,) = read_inventory(gap)
    assert record.counts_by_source["gap-source"].decks == 1


def test_mapping_rule_change_writes_a_new_run_specific_gap_report(
    tmp_path, monkeypatch
):
    class GapSource:
        @staticmethod
        def iter_decks(**_kwargs):
            yield RawDeck(
                "changed-gap-source",
                "gap",
                [RawCard(1, "Unmapped Card")],
                external_ids={"deck": "stable"},
            )

    monkeypatch.setitem(SOURCES, "changed-gap-source", GapSource)
    empty_rules = tmp_path / "empty-rules"
    empty_rules.mkdir()
    changed_rules = tmp_path / "changed-rules"
    changed_rules.mkdir()
    (changed_rules / "inactive.json").write_text(json.dumps({
        "schema_version": 2,
        "rules": [{
            "rule_id": "known-rejection", "source_name": "Some Other Card",
            "source_set": "PAL", "source_number": "185", "source_rule": None,
            "source_stage": "Supporter", "source_previous_stage": None,
            "targets": [{
                "card_id": 1213, "expected_name": "Judge",
                "mapping_confidence": 4,
                "rationale": "Reviewed deterministic substitution.",
            }],
            "family_id": None,
            "allow_cross_subtype": False,
        }],
    }), encoding="utf-8")
    def run(rule_dir, job_id):
        monkeypatch.setenv("SLURM_JOB_ID", job_id)
        return main([
            "--source", "changed-gap-source", "--card-swap-strategy", "mapping",
            "--card-swap-map", str(rule_dir),
            "--out", str(tmp_path / "decks"),
        ])

    assert run(empty_rules, "first") == 0
    first = tmp_path / "decks" / "mapping-gaps-first.jsonl.gz"
    assert run(changed_rules, "second") == 0
    second = tmp_path / "decks" / "mapping-gaps-second.jsonl.gz"
    assert first.exists()
    assert second.exists()
    assert _gap_checkpoint_label(first) != _gap_checkpoint_label(second)
    (record,) = read_inventory(second)
    assert record.counts_by_source["changed-gap-source"].decks == 1


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--card-swap-strategy", "heuristic"],
        ["--card-swap-strategy", "mapping", "--disable-card-swap"],
    ],
)
def test_unselected_or_disabled_mapping_does_not_load_stale_rules(
    tmp_path, monkeypatch, extra_args
):
    class EmptySource:
        @staticmethod
        def iter_decks(**_kwargs):
            return iter(())

    rules = tmp_path / "rules"
    rules.mkdir()
    (rules / "invalid.json").write_text('{"schema_version": 999, "rules": []}')
    monkeypatch.setitem(SOURCES, "empty", EmptySource)

    assert (
        main(
            [
                "--source",
                "empty",
                "--card-swap-map",
                str(rules),
                "--out",
                str(tmp_path / "decks"),
                *extra_args,
            ]
        )
        == 0
    )


def test_selected_mapping_rejects_stale_rules_at_startup(tmp_path, monkeypatch):
    class EmptySource:
        @staticmethod
        def iter_decks(**_kwargs):
            return iter(())

    rules = tmp_path / "rules"
    rules.mkdir()
    (rules / "invalid.json").write_text('{"schema_version": 999, "rules": []}')
    monkeypatch.setitem(SOURCES, "empty", EmptySource)

    with pytest.raises(MappingRuleError, match="schema_version"):
        main(
            [
                "--source",
                "empty",
                "--card-swap-strategy",
                "mapping",
                "--card-swap-map",
                str(rules),
                "--out",
                str(tmp_path / "decks"),
            ]
        )
