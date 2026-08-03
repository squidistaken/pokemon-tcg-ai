"""Tests for deterministic source-printing inventory checkpoints."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from scraper.card_index import CardIndex
from scraper.card_swapper import SourceProfile
from scraper.inventory import (
    CanonicalPrinting,
    InventoryCheckpointMismatch,
    SeenCardInventory,
    inventory_content_sha256,
    read_inventory,
    resolve_inventory_path,
)
from scraper.models import RawCard, RawDeck


def _canonicalize(_source: str, card: RawCard) -> CanonicalPrinting:
    set_codes = {"Obsidian Flames": "OBF", "Obsidian Flames expansion": "OBF"}
    return CanonicalPrinting(
        name=card.name,
        set_code=set_codes.get(card.set_code or "", card.set_code),
        number=card.number,
    )


def _deck(source: str, archetype: str, cards: list[RawCard], suffix: str) -> RawDeck:
    return RawDeck(
        source=source,
        archetype=archetype,
        cards=cards,
        url=f"https://example.test/{suffix}",
        event="Test Cup",
        event_date="2026-08-01",
        external_ids={"deck": suffix},
    )


def _profile(name: str) -> SourceProfile:
    return SourceProfile(
        name=name,
        stage="Stage 1 Pokémon",
        rule="n/a",
        previous_stage="Charmander",
        hp=90,
        energy_type="Fire",
        weakness="Water",
        resistance=None,
        retreat=2,
        moves=(("Flare", "RC", "50", ""),),
    )


def test_canonicalization_failure_does_not_partially_mutate_inventory():
    def fail_on_second(_source: str, card: RawCard) -> CanonicalPrinting:
        if card.name == "Broken":
            raise RuntimeError("set catalogue unavailable")
        return CanonicalPrinting(card.name, card.set_code, card.number)

    inventory = SeenCardInventory(CardIndex(), fail_on_second)
    deck = _deck(
        "limitless",
        "Atomic",
        [RawCard(1, "Working", "SET", "1"), RawCard(1, "Broken", "SET", "2")],
        "atomic",
    )

    with pytest.raises(RuntimeError, match="set catalogue unavailable"):
        inventory.observe(deck)

    assert inventory.records == ()
    assert inventory.observation_keys == frozenset()


def test_aggregates_cross_source_spellings_counts_matches_and_examples(
    tmp_path: Path,
):
    loaded: list[RawCard] = []

    def load_profile(card: RawCard) -> SourceProfile:
        loaded.append(card)
        return _profile(card.name)

    inventory = SeenCardInventory(
        CardIndex(), _canonicalize, load_profile, example_limit=2
    )
    inventory.observe(
        _deck(
            "bulbapedia",
            "Deck B",
            [
                RawCard(
                    2,
                    "Charmeleon",
                    "Obsidian Flames",
                    "27",
                    "pokemon",
                ),
                RawCard(
                    1,
                    "Charmeleon",
                    "Obsidian Flames",
                    "27",
                    "pokemon",
                ),
            ],
            "b",
        )
    )
    inventory.observe(
        _deck(
            "limitless",
            "Deck A",
            [RawCard(3, "Charmeleon", "OBF", "27", "pokemon")],
            "a",
        )
    )

    (record,) = inventory.records
    assert record.identity == CanonicalPrinting("Charmeleon", "OBF", "27")
    assert record.match.method == "ambiguous"
    assert record.match.card_id is None
    assert record.match.candidate_ids == tuple(sorted(record.match.candidate_ids))
    assert len(record.match.candidate_ids) > 1
    assert record.counts_by_source["bulbapedia"].to_json() == {
        "decks": 1,
        "lines": 2,
        "copies": 3,
    }
    assert record.counts_by_source["limitless"].to_json() == {
        "decks": 1,
        "lines": 1,
        "copies": 3,
    }
    assert {spelling.set_code for spelling in record.spellings} == {
        "OBF",
        "Obsidian Flames",
    }
    assert {example.archetype for example in record.examples} == {"Deck A", "Deck B"}
    assert loaded == [RawCard(2, "Charmeleon", "OBF", "27", "pokemon")]
    assert record.profile == _profile("Charmeleon")
    assert record.metadata_error is None

    checkpoint = tmp_path / "seen_cards.jsonl.gz"
    inventory.write(checkpoint)
    (restored,) = read_inventory(checkpoint)
    assert restored.to_json() == record.to_json()


def test_records_metadata_error_without_stopping_discovery():
    def fail(_card: RawCard) -> SourceProfile:
        raise RuntimeError("bad response")

    inventory = SeenCardInventory(CardIndex(), _canonicalize, fail)
    inventory.observe(
        _deck(
            "limitless",
            "Unknown",
            [RawCard(1, "Definitely Missing", "BAD", "1")],
            "missing",
        )
    )

    (record,) = inventory.records
    assert record.match.method == "unresolved"
    assert record.metadata_error == "RuntimeError: bad response"
    assert record.profile is None


def test_write_is_deterministic_sorted_gzip_jsonl(tmp_path: Path):
    first = SeenCardInventory(CardIndex(), _canonicalize)
    first.observe(
        _deck(
            "limitless",
            "Zed",
            [RawCard(1, "Z Card"), RawCard(1, "A Card")],
            "z",
        )
    )
    second = SeenCardInventory(CardIndex(), _canonicalize)
    second.observe(
        _deck(
            "limitless",
            "Zed",
            [RawCard(1, "A Card"), RawCard(1, "Z Card")],
            "z",
        )
    )
    first_path = tmp_path / "first.jsonl.gz"
    second_path = tmp_path / "second.jsonl.gz"

    first.write(first_path)
    second.write(second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    with gzip.open(first_path, "rt", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle]
    assert lines[0]["record_type"] == "header"
    assert [line["identity"]["name"] for line in lines[1:]] == [
        "A Card",
        "Z Card",
    ]
    assert all(line["schema_version"] == 1 for line in lines)


def test_plain_and_gzip_inventory_reads_and_hashes_are_equivalent(tmp_path: Path):
    compressed = tmp_path / "seen_cards.jsonl.gz"
    plain = tmp_path / "seen_cards.jsonl"
    inventory = SeenCardInventory(CardIndex(), _canonicalize)
    inventory.observe(_deck("limitless", "One", [RawCard(1, "Iono")], "one"))
    inventory.write(compressed)
    plain.write_bytes(gzip.decompress(compressed.read_bytes()))

    assert read_inventory(plain) == read_inventory(compressed)
    assert inventory_content_sha256(plain) == inventory_content_sha256(compressed)
    assert resolve_inventory_path(tmp_path / "seen_cards") == compressed


def test_inventory_reader_uses_magic_bytes_instead_of_suffix(tmp_path: Path):
    compressed = tmp_path / "compressed.jsonl"
    plain = tmp_path / "plain.jsonl.gz"
    inventory = SeenCardInventory(CardIndex(), _canonicalize)
    inventory.observe(_deck("limitless", "One", [RawCard(1, "Iono")], "one"))
    actual = tmp_path / "actual.jsonl.gz"
    inventory.write(actual)
    payload = gzip.decompress(actual.read_bytes())
    compressed.write_bytes(actual.read_bytes())
    plain.write_bytes(payload)

    assert read_inventory(compressed) == read_inventory(plain)


def test_inventory_path_resolver_falls_back_to_available_form(tmp_path: Path):
    plain = tmp_path / "seen_cards.jsonl"
    plain.write_text("", encoding="utf-8")

    assert resolve_inventory_path(tmp_path / "seen_cards.jsonl.gz") == plain


def test_inventory_path_resolver_rejects_mismatched_copies(tmp_path: Path):
    plain = tmp_path / "seen_cards.jsonl"
    compressed = tmp_path / "seen_cards.jsonl.gz"
    plain.write_text("plain\n", encoding="utf-8")
    with gzip.open(compressed, "wt", encoding="utf-8") as handle:
        handle.write("different\n")

    with pytest.raises(ValueError, match="plain and gzip inventory copies differ"):
        resolve_inventory_path(tmp_path / "seen_cards")


def test_failed_atomic_replace_preserves_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint = tmp_path / "seen_cards.jsonl.gz"
    checkpoint.write_bytes(b"previous checkpoint")
    inventory = SeenCardInventory(CardIndex(), _canonicalize)
    inventory.observe(_deck("limitless", "One", [RawCard(1, "Iono")], "one"))

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr("scraper.inventory.os.replace", fail_replace)

    with pytest.raises(OSError, match="disk unavailable"):
        inventory.write(checkpoint)

    assert checkpoint.read_bytes() == b"previous checkpoint"
    assert list(tmp_path.glob(".seen_cards.jsonl.gz.*.tmp")) == []


def test_existing_checkpoint_can_be_loaded_merged_and_rewritten(tmp_path: Path):
    checkpoint = tmp_path / "seen_cards.jsonl.gz"
    initial = SeenCardInventory(CardIndex(), _canonicalize)
    initial.observe(
        _deck(
            "limitless",
            "One",
            [RawCard(2, "Iono", "PAL", "185", "trainer")],
            "one",
        )
    )
    initial.write(checkpoint)

    resumed = SeenCardInventory(CardIndex(), _canonicalize)
    resumed.merge_file(checkpoint)
    resumed.observe(
        _deck(
            "bulbapedia",
            "Two",
            [RawCard(3, "Iono", "PAL", "185", "trainer")],
            "two",
        )
    )
    resumed.write(checkpoint)

    (record,) = read_inventory(checkpoint)
    assert record.counts_by_source["limitless"].copies == 2
    assert record.counts_by_source["bulbapedia"].copies == 3
    assert {example.archetype for example in record.examples} == {"One", "Two"}


def test_restart_replay_skips_decks_already_counted_in_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "seen_cards.jsonl.gz"
    first_deck = _deck(
        "limitless",
        "One",
        [RawCard(2, "Iono", "PAL", "185", "trainer")],
        "one",
    )
    second_deck = _deck(
        "limitless",
        "Two",
        [RawCard(3, "Iono", "PAL", "185", "trainer")],
        "two",
    )
    initial = SeenCardInventory(CardIndex(), _canonicalize)
    assert initial.observe(first_deck) is True
    initial.write(checkpoint)

    resumed = SeenCardInventory(CardIndex(), _canonicalize)
    resumed.merge_file(checkpoint)
    resumed.merge_file(checkpoint)
    assert resumed.observe(first_deck) is False
    assert resumed.observe(second_deck) is True
    resumed.write(checkpoint)

    replayed = SeenCardInventory(CardIndex(), _canonicalize)
    replayed.merge_file(checkpoint)
    assert replayed.observe(first_deck) is False
    assert replayed.observe(second_deck) is False
    (record,) = replayed.records
    assert record.counts_by_source["limitless"].to_json() == {
        "decks": 2,
        "lines": 2,
        "copies": 5,
    }
    assert len(replayed.observation_keys) == 2


def test_read_rejects_invalid_schema(tmp_path: Path):
    path = tmp_path / "broken.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write('{"schema_version":999}\n')

    with pytest.raises(ValueError, match="invalid inventory line 1"):
        read_inventory(path)


def test_negative_example_limit_is_rejected():
    with pytest.raises(ValueError, match="example_limit"):
        SeenCardInventory(CardIndex(), _canonicalize, example_limit=-1)


def test_metadata_errors_are_retried_after_resume(tmp_path: Path):
    checkpoint = tmp_path / "seen_cards.jsonl.gz"
    failing = SeenCardInventory(
        CardIndex(),
        _canonicalize,
        lambda _card: (_ for _ in ()).throw(RuntimeError("temporary")),
    )
    failing.observe(
        _deck("limitless", "One", [RawCard(1, "Charmeleon", "OBF", "27")], "1")
    )
    failing.write(checkpoint)

    resumed = SeenCardInventory(
        CardIndex(), _canonicalize, lambda card: _profile(card.name)
    )
    resumed.merge_file(checkpoint)

    assert resumed.refresh_metadata_errors() == 1
    assert resumed.records[0].profile == _profile("Charmeleon")
    assert resumed.records[0].metadata_error is None


def test_checkpoint_label_prevents_stale_gap_inventory_merge(tmp_path: Path):
    checkpoint = tmp_path / "gaps.jsonl.gz"
    old = SeenCardInventory(
        CardIndex(), _canonicalize, checkpoint_label="mapping-rules:old"
    )
    old.observe(_deck("limitless", "One", [RawCard(1, "Missing")], "1"))
    old.write(checkpoint)

    current = SeenCardInventory(
        CardIndex(), _canonicalize, checkpoint_label="mapping-rules:new"
    )

    with pytest.raises(InventoryCheckpointMismatch, match="different"):
        current.merge_file(checkpoint)
