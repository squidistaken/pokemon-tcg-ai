"""Tests for deck resolution and same-name variant selection."""

from __future__ import annotations

import csv
from typing import override

from scraper.card_index import CardIndex
from scraper.card_swapper import CardSwapper, HeuristicCardSwapper, SourceProfile
from scraper.models import CardSwap, RawCard, RawDeck
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


def test_family_mapping_backtracks_to_preserve_existing_evolution_chain(tmp_path):
    index = _make_index(
        tmp_path,
        [
            _row(1, "Magnemite", "TST", "Basic Pokémon"),
            _row(2, "Pikachu", "TST", "Basic Pokémon"),
            _row(3, "Raichu", "TST", "Stage 1 Pokémon", prev="Pikachu"),
        ],
    )

    class FamilySwapper(CardSwapper):
        @override
        def resolve(self, card: RawCard) -> tuple[CardSwap, ...]:
            targets = (1, 2) if card.name == "Larva" else (3,)
            return tuple(
                CardSwap(
                    source_name=card.name,
                    source_set=card.set_code,
                    source_number=card.number,
                    count=card.count,
                    target_id=target,
                    target_name=index.by_id[target].name,
                    kind="mapping",
                    confidence=1.0,
                    rationale="reviewed family",
                    rule_id=f"rule-{card.name}",
                    family_id="family-1",
                    source_previous_stage="Larva" if card.name == "Moth" else None,
                )
                for target in targets
            )

    raw = RawDeck(
        "test",
        "family",
        [RawCard(2, "Larva"), RawCard(2, "Moth")],
    )

    resolved = resolve_deck(raw, index, FamilySwapper())

    assert resolved.ids == [2, 2, 3, 3]
    assert [swap.target_id for swap in resolved.swaps] == [2, 3]
    assert resolved.swap_failures == []


def test_family_mapping_fails_closed_when_no_coherent_assignment_exists(tmp_path):
    index = _make_index(
        tmp_path,
        [
            _row(1, "Magnemite", "TST", "Basic Pokémon"),
            _row(2, "Raichu", "TST", "Stage 1 Pokémon", prev="Pikachu"),
        ],
    )

    class BrokenFamilySwapper(CardSwapper):
        @override
        def resolve(self, card: RawCard) -> tuple[CardSwap, ...]:
            target = 1 if card.name == "Larva" else 2
            return (
                CardSwap(
                    source_name=card.name,
                    source_set=None,
                    source_number=None,
                    count=card.count,
                    target_id=target,
                    target_name=index.by_id[target].name,
                    kind="mapping",
                    confidence=1.0,
                    rationale="reviewed family",
                    family_id="family-1",
                    source_previous_stage="Larva" if card.name == "Moth" else None,
                ),
            )

    raw = RawDeck("test", "family", [RawCard(2, "Larva"), RawCard(2, "Moth")])

    resolved = resolve_deck(raw, index, BrokenFamilySwapper())

    assert resolved.ids == []
    assert resolved.unresolved_cards == raw.cards
    assert resolved.swap_failures


def test_unrelated_exact_pokemon_cannot_satisfy_a_mapping_family(tmp_path):
    index = _make_index(
        tmp_path,
        [
            _row(1, "Magnemite", "TST", "Basic Pokémon"),
            _row(2, "Pikachu", "TST", "Basic Pokémon"),
            _row(3, "Raichu", "TST", "Stage 1 Pokémon", prev="Pikachu"),
        ],
    )

    class MixedFamilySwapper(CardSwapper):
        @override
        def resolve(self, card: RawCard) -> tuple[CardSwap, ...]:
            target = 1 if card.name == "Larva" else 3
            return (
                CardSwap(
                    card.name,
                    None,
                    None,
                    card.count,
                    target,
                    index.by_id[target].name,
                    "mapping",
                    1.0,
                    "reviewed family",
                    family_id="family-1",
                    source_stage=(
                        "Basic Pokémon"
                        if card.name == "Larva"
                        else "Stage 1 Pokémon"
                    ),
                    source_previous_stage=(
                        "Larva" if card.name == "Moth" else None
                    ),
                ),
            )

    raw = RawDeck(
        "test",
        "family",
        [RawCard(1, "Pikachu", "TST", "2"), RawCard(1, "Larva"), RawCard(1, "Moth")],
    )

    resolved = resolve_deck(raw, index, MixedFamilySwapper())

    assert resolved.ok is False
    assert resolved.ids == [2]
    assert resolved.swap_failures


def test_family_mapping_supports_a_reviewed_rare_candy_style_chain(tmp_path):
    index = _make_index(
        tmp_path,
        [
            _row(1, "Squirtle", "TST", "Basic Pokémon"),
            _row(2, "Wartortle", "TST", "Stage 1 Pokémon", prev="Squirtle"),
            _row(3, "Blastoise", "TST", "Stage 2 Pokémon", prev="Wartortle"),
        ],
    )

    class CandyFamilySwapper(CardSwapper):
        @override
        def resolve(self, card: RawCard) -> tuple[CardSwap, ...]:
            target = 1 if card.name == "Larva" else 3
            return (
                CardSwap(
                    card.name,
                    None,
                    None,
                    card.count,
                    target,
                    index.by_id[target].name,
                    "mapping",
                    1.0,
                    "reviewed family",
                    family_id="family-1",
                    source_stage=(
                        "Basic Pokémon"
                        if card.name == "Larva"
                        else "Stage 2 Pokémon"
                    ),
                    source_previous_stage=(
                        "Cocoon" if card.name == "Moth King" else None
                    ),
                ),
            )

    raw = RawDeck(
        "test", "family", [RawCard(2, "Larva"), RawCard(2, "Moth King")]
    )

    resolved = resolve_deck(raw, index, CandyFamilySwapper())

    assert resolved.ok is True
    assert resolved.ids == [1, 1, 3, 3]


def test_ordered_fallback_skips_copy_cap_conflict():
    index = CardIndex()

    class ChoiceSwapper(CardSwapper):
        @override
        def resolve(self, card: RawCard) -> tuple[CardSwap, ...]:
            return tuple(
                CardSwap(
                    card.name,
                    None,
                    None,
                    card.count,
                    target,
                    index.by_id[target].name,
                    "mapping",
                    1.0,
                    "ordered fallback",
                )
                for target in (1213, 1181)
            )

    judge = index.by_id[1213]
    raw = RawDeck(
        "test",
        "fallback",
        [
            RawCard(4, judge.name, judge.set_code, judge.number),
            RawCard(1, "Legacy Draw"),
        ],
    )

    resolved = resolve_deck(raw, index, ChoiceSwapper())

    assert resolved.ok is True
    assert resolved.swaps[-1].target_id == 1181


def test_ordered_fallback_skips_extra_ace_spec():
    index = CardIndex()

    class ChoiceSwapper(CardSwapper):
        @override
        def resolve(self, card: RawCard) -> tuple[CardSwap, ...]:
            return tuple(
                CardSwap(
                    card.name,
                    None,
                    None,
                    card.count,
                    target,
                    index.by_id[target].name,
                    "mapping",
                    1.0,
                    "ordered fallback",
                )
                for target in (10, 1213)
            )

    ace = index.by_id[1088]
    raw = RawDeck(
        "test",
        "fallback",
        [RawCard(1, ace.name, ace.set_code, ace.number), RawCard(1, "Old ACE")],
    )

    resolved = resolve_deck(raw, index, ChoiceSwapper())

    assert resolved.ok is True
    assert resolved.swaps[-1].target_id == 1213
