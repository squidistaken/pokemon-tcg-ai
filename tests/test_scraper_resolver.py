"""Tests for resolve_deck, including the card-swapper fallback path."""

from __future__ import annotations

import csv

from scraper.card_index import CardIndex
from scraper.card_swapper import CardSwapper
from scraper.models import RawCard, RawDeck
from scraper.resolver import resolve_deck

_HEADER = [
    "Card ID",
    "Card Name",
    "Expansion",
    "Collection No.",
    "Stage (Pokémon)/Type (Energy and Trainer)",
    "Rule",
    "Category",
    "Previous stage",
]


def _row(card_id, name, set_code, stage, prev="n/a", rule="n/a"):
    return {
        "Card ID": str(card_id),
        "Card Name": name,
        "Expansion": set_code,
        "Collection No.": str(card_id),
        "Stage (Pokémon)/Type (Energy and Trainer)": stage,
        "Rule": rule,
        "Category": "n/a",
        "Previous stage": prev,
    }


def _make_index(tmp_path, rows):
    csv_path = tmp_path / "cards.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_HEADER)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return CardIndex(csv_path=str(csv_path))


def test_resolve_deck_without_swapper_behaves_as_before(tmp_path):
    index = _make_index(tmp_path, [_row(1, "Foo Basic", "TST", "Basic Pokémon")])
    raw = RawDeck(
        source="test",
        archetype="test-deck",
        cards=[
            RawCard(count=1, name="Foo Basic"),
            RawCard(count=1, name="Totally Missing Card"),
        ],
    )

    resolved = resolve_deck(raw, index)

    assert resolved.ids == [1]
    assert [c.name for c in resolved.unresolved_cards] == ["Totally Missing Card"]
    assert resolved.swaps == []


def test_swapper_rescues_an_otherwise_unresolved_trainer_card(tmp_path):
    index = _make_index(tmp_path, [_row(1, "Real Substitute Supporter", "TST", "Supporter")])
    swapper = CardSwapper(index, mapping={"Missing Supporter": "Real Substitute Supporter"})
    raw = RawDeck(
        source="test",
        archetype="test-deck",
        cards=[RawCard(count=2, name="Missing Supporter")],
    )

    resolved = resolve_deck(raw, index, swapper)

    assert resolved.ids == [1, 1]
    assert resolved.unresolved_cards == []
    assert resolved.swaps == [("Missing Supporter", "Real Substitute Supporter")]


def test_evolution_chain_guard_blocks_a_pokemon_swap(tmp_path):
    index = _make_index(
        tmp_path,
        [
            _row(1, "Foo Evolved", "TST", "Stage 1 Pokémon", prev="Foo Base"),
            _row(2, "Unrelated Basic", "TST", "Basic Pokémon"),
        ],
    )
    swapper = CardSwapper(index, mapping={"Foo Base": "Unrelated Basic"})
    raw = RawDeck(
        source="test",
        archetype="test-deck",
        cards=[
            RawCard(count=1, name="Foo Evolved"),
            RawCard(count=1, name="Foo Base"),
        ],
    )

    resolved = resolve_deck(raw, index, swapper)

    # "Foo Evolved" needs its real "Foo Base", not the swapper's substitute —
    # the guard must leave "Foo Base" unresolved rather than silently swap it.
    assert resolved.ids == [1]
    assert [c.name for c in resolved.unresolved_cards] == ["Foo Base"]
    assert resolved.swaps == []


def test_evolution_chain_guard_does_not_block_an_unrelated_swap(tmp_path):
    index = _make_index(
        tmp_path,
        [
            _row(1, "Foo Evolved", "TST", "Stage 1 Pokémon", prev="Foo Base"),
            _row(2, "Foo Base", "TST", "Basic Pokémon"),
            _row(3, "Real Substitute Supporter", "TST", "Supporter"),
        ],
    )
    swapper = CardSwapper(index, mapping={"Missing Supporter": "Real Substitute Supporter"})
    raw = RawDeck(
        source="test",
        archetype="test-deck",
        cards=[
            RawCard(count=1, name="Foo Evolved"),
            RawCard(count=1, name="Foo Base"),
            RawCard(count=1, name="Missing Supporter"),
        ],
    )

    resolved = resolve_deck(raw, index, swapper)

    assert sorted(resolved.ids) == [1, 2, 3]
    assert resolved.unresolved_cards == []
    assert resolved.swaps == [("Missing Supporter", "Real Substitute Supporter")]
