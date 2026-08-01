"""Tests for deck resolution and same-name variant selection."""

from __future__ import annotations

import csv

from scraper.card_index import CardIndex
from scraper.card_swapper import HeuristicCardSwapper, SourceProfile
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
        writer = csv.DictWriter(f, fieldnames=_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return CardIndex(csv_path=str(csv_path))


def _charmeleon_profile() -> SourceProfile:
    return SourceProfile(
        name="Charmeleon",
        stage="Stage 1 Pokémon",
        rule="n/a",
        previous_stage="Charmander",
        hp=90,
        energy_type="Fire",
        weakness="Water",
        resistance="none",
        retreat=2,
        moves=(("Heat Tackle", "RR", "70", "This Pokémon also does 20 damage to itself."),),
    )


def test_resolve_deck_without_swapper_behaves_as_before(tmp_path):
    index = _make_index(tmp_path, [_row(1, "Foo Basic", "TST", "Basic Pokémon")])
    raw = RawDeck(
        source="test",
        archetype="test-deck",
        cards=[RawCard(1, "Foo Basic"), RawCard(1, "Totally Missing Card")],
    )

    resolved = resolve_deck(raw, index)

    assert resolved.ids == [1]
    assert [card.name for card in resolved.unresolved_cards] == [
        "Totally Missing Card"
    ]
    assert resolved.swaps == []


def test_cross_name_card_remains_unresolved():
    index = CardIndex()
    profile = SourceProfile(
        name="Iono",
        stage="Supporter",
        rule="n/a",
        previous_stage=None,
        hp=None,
        energy_type=None,
        weakness=None,
        resistance=None,
        retreat=None,
        moves=(("", "", "", "Each player shuffles their hand and draws cards."),),
    )
    raw = RawDeck("test", "iono", [RawCard(2, "Iono", "PAF", "80", "trainer")])

    resolved = resolve_deck(
        raw, index, HeuristicCardSwapper(index, lambda _card: profile)
    )

    assert resolved.ids == []
    assert resolved.unresolved_cards == raw.cards
    assert resolved.swaps == []


def test_copy_cap_is_checked_after_variant_selection():
    index = CardIndex()
    raw = RawDeck(
        "test",
        "charizard",
        [
            RawCard(4, "Charmeleon", "ASC", "21", "pokemon"),
            RawCard(2, "Charmeleon", "OBF", "27", "pokemon"),
        ],
    )

    resolved = resolve_deck(
        raw, index, HeuristicCardSwapper(index, lambda _card: _charmeleon_profile())
    )

    assert resolved.ids == [927] * 4
    assert resolved.unresolved_cards == [raw.cards[1]]
    assert resolved.swap_failures


def test_missing_card_does_not_create_a_false_guard_failure():
    index = CardIndex()
    raw = RawDeck(
        "test",
        "charizard",
        [
            RawCard(2, "Charmeleon", "OBF", "27", "pokemon"),
            RawCard(1, "Entirely Missing", "ZZZ", "1", "trainer"),
        ],
    )

    resolved = resolve_deck(
        raw, index, HeuristicCardSwapper(index, lambda _card: _charmeleon_profile())
    )

    assert resolved.ids == [927, 927]
    assert resolved.unresolved_cards == [raw.cards[1]]
    assert resolved.swap_failures == []


def test_exact_legal_energy_is_never_swapped():
    index = CardIndex()
    raw = RawDeck(
        "test",
        "energy",
        [RawCard(1, "Team Rocket’s Energy", "DRI", "182", "energy")],
    )

    resolved = resolve_deck(
        raw,
        index,
        HeuristicCardSwapper(
            index, lambda _card: (_ for _ in ()).throw(AssertionError())
        ),
    )

    assert resolved.ids == [15]
    assert resolved.swaps == []


def test_unique_same_name_printing_has_variant_provenance():
    index = CardIndex()
    raw = RawDeck(
        "test", "judge", [RawCard(2, "Judge", "PAF", "228", "trainer")]
    )

    resolved = resolve_deck(raw, index)

    assert resolved.ids == [1213, 1213]
    assert resolved.swaps[0].source_set == "PAF"
    assert resolved.swaps[0].target_id == 1213
    assert resolved.swaps[0].kind == "variant"


def test_obf_charmeleon_uses_closest_legal_variant():
    index = CardIndex()
    raw = RawDeck(
        "test",
        "charizard",
        [RawCard(2, "Charmeleon", "OBF", "27", "pokemon")],
    )

    resolved = resolve_deck(
        raw, index, HeuristicCardSwapper(index, lambda _card: _charmeleon_profile())
    )

    assert resolved.ids == [927, 927]
    assert resolved.swaps[0].target_id == 927
    assert resolved.swaps[0].kind == "variant"
