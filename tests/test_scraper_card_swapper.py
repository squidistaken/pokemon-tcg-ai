"""Tests for gameplay-profile matching of same-name card variants."""

from __future__ import annotations

import csv

import pytest

from scraper.card_index import CardIndex
from scraper.card_swapper import (
    HeuristicCardSwapper,
    LimitlessProfileLoader,
    SimilarityConfig,
    SourceProfile,
    _text_similarity,
    parse_limitless_profile,
)
from scraper.models import RawCard


def _pokemon_html(
    name: str,
    *,
    hp: int,
    attack: str,
    cost: str,
    damage: int,
    effect: str,
    retreat: int,
) -> str:
    return f"""
    <div class="card-text">
      <div class="card-text-section">
        <p class="card-text-title"><span class="card-text-name">{name}</span> - Fire - {hp} HP</p>
        <p class="card-text-type">Pokémon - Stage 1 - Evolves from Charmander</p>
      </div>
      <div class="card-text-section">
        <div class="card-text-attack">
          <p class="card-text-attack-info"><span class="ptcg-symbol">{cost}</span> {attack} {damage}</p>
          <p class="card-text-attack-effect">{effect}</p>
        </div>
      </div>
      <div class="card-text-section"><p class="card-text-wrr">
        Weakness: Water<br>Resistance: none<br>Retreat: {retreat}<br>
      </p></div>
    </div>
    """


def _trainer_html(name: str, subtype: str, effect: str) -> str:
    return f"""
    <div class="card-text">
      <div class="card-text-section">
        <p class="card-text-title"><span class="card-text-name">{name}</span></p>
        <p class="card-text-type">Trainer - {subtype}</p>
      </div>
      <div class="card-text-section">{effect}</div>
    </div>
    """


CHARMELEON_HTML = _pokemon_html(
    "Charmeleon",
    hp=90,
    attack="Heat Tackle",
    cost="RR",
    damage=70,
    effect="This Pokémon also does 20 damage to itself.",
    retreat=2,
)

DRILBUR_HTML = """
<div class="card-text">
  <div class="card-text-section">
    <p class="card-text-title"><span class="card-text-name">Drilbur</span> - Fighting - 70 HP</p>
    <p class="card-text-type">Pokémon - Basic</p>
  </div>
  <div class="card-text-section">
    <div class="card-text-attack">
      <p class="card-text-attack-info"><span class="ptcg-symbol">C</span> Call for Family</p>
      <p class="card-text-attack-effect">Search your deck for up to 2 Basic Pokémon and put them onto your Bench. Then, shuffle your deck.</p>
    </div>
    <div class="card-text-attack">
      <p class="card-text-attack-info"><span class="ptcg-symbol">CCC</span> Dig Claws 50</p>
    </div>
  </div>
  <div class="card-text-section"><p class="card-text-wrr">
    Weakness: Grass<br>Resistance: none<br>Retreat: 2<br>
  </p></div>
</div>
"""


@pytest.mark.parametrize(
    ("incoming", "html", "expected_target_id"),
    [
        (RawCard(2, "Charmeleon", "OBF", "27", "pokemon"), CHARMELEON_HTML, 927),
        (RawCard(2, "Drilbur", "PBL", "46", "pokemon"), DRILBUR_HTML, None),
        (
            RawCard(2, "Iono", "PAF", "80", "trainer"),
            _trainer_html(
                "Iono",
                "Supporter",
                "Each player shuffles their hand and puts it on the bottom of their deck. "
                "Each player draws a card for each of their remaining Prize cards.",
            ),
            None,
        ),
        (
            RawCard(4, "Nest Ball", "PAF", "84", "trainer"),
            _trainer_html(
                "Nest Ball",
                "Item",
                "Search your deck for a Basic Pokémon and put it onto your Bench.",
            ),
            None,
        ),
    ],
)
def test_profile_resolution(incoming, html, expected_target_id):
    swapper = HeuristicCardSwapper(
        CardIndex(), profile_loader=lambda _card: parse_limitless_profile(html)
    )

    decisions = swapper.resolve(incoming)

    assert (decisions[0].target_id if decisions else None) == expected_target_id
    if decisions:
        assert decisions[0].source_set == incoming.set_code
        assert decisions[0].source_number == incoming.number
        assert decisions[0].count == incoming.count
        assert decisions[0].kind == "variant"


def test_profile_loader_fetches_each_printing_once(tmp_path):
    class FakeClient:
        def __init__(self):
            self.urls: list[str] = []

        def get_text(self, url: str) -> str:
            self.urls.append(url)
            return CHARMELEON_HTML

    client = FakeClient()
    loader = LimitlessProfileLoader(client=client, cache_dir=tmp_path)
    card = RawCard(2, "Charmeleon", "OBF", "027", "pokemon")

    assert loader(card) == loader(card)
    assert client.urls == ["https://limitlesstcg.com/cards/OBF/27"]
    assert (tmp_path / "OBF-27.html").exists()


def test_profile_loader_resolves_a_full_expansion_name(tmp_path):
    class FakeClient:
        def __init__(self):
            self.urls: list[str] = []

        def get_text(self, url: str) -> str:
            self.urls.append(url)
            if url == "https://limitlesstcg.com/cards":
                return '<a href="/cards/OBF">Obsidian Flames OBF</a>'
            return CHARMELEON_HTML

    client = FakeClient()
    loader = LimitlessProfileLoader(client=client, cache_dir=tmp_path)

    profile = loader(RawCard(2, "Charmeleon", "Obsidian Flames", "27", "pokemon"))

    assert profile is not None
    assert profile.name == "Charmeleon"
    assert client.urls == [
        "https://limitlesstcg.com/cards",
        "https://limitlesstcg.com/cards/OBF/27",
    ]


def test_profile_name_must_match_the_incoming_card():
    swapper = HeuristicCardSwapper(
        CardIndex(),
        profile_loader=lambda _card: parse_limitless_profile(
            CHARMELEON_HTML.replace("Charmeleon", "Different Pokémon")
        ),
    )

    assert not swapper.resolve(RawCard(1, "Charmeleon", "OBF", "27"))


def test_equal_safe_scores_choose_a_deterministic_card_id(tmp_path):
    path = tmp_path / "cards.csv"
    fieldnames = [
        "Card ID",
        "Card Name",
        "Expansion",
        "Collection No.",
        "Stage (Pokémon)/Type (Energy and Trainer)",
        "Rule",
        "Previous stage",
    ]
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "Card ID": card_id,
                    "Card Name": "Shared Stadium",
                    "Expansion": set_code,
                    "Collection No.": card_id,
                    "Stage (Pokémon)/Type (Energy and Trainer)": "Stadium",
                    "Rule": "n/a",
                    "Previous stage": "n/a",
                }
                for card_id, set_code in [(9, "TWO"), (3, "ONE")]
            ]
        )
    index = CardIndex(str(path))
    profile = SourceProfile(
        name="Shared Stadium",
        stage="Stadium",
        rule="n/a",
        previous_stage=None,
        hp=None,
        energy_type=None,
        weakness=None,
        resistance=None,
        retreat=None,
        moves=(("", "", "", ""),),
    )

    decisions = HeuristicCardSwapper(index, lambda _card: profile).resolve(
        RawCard(1, "Shared Stadium", "SRC", "1", "trainer")
    )

    assert decisions[0].target_id == 3


def test_text_sequence_weight_is_configurable():
    left = "Search your deck for a Basic Pokémon and put it onto your Bench."
    reordered = "Put a Basic Pokémon onto your Bench after searching your deck."

    token_only = _text_similarity(
        left, reordered, SimilarityConfig(text_sequence_weight=0.0)
    )
    sequence_only = _text_similarity(
        left, reordered, SimilarityConfig(text_sequence_weight=1.0)
    )

    assert token_only > sequence_only


def test_acceptance_thresholds_are_configurable():
    swapper = HeuristicCardSwapper(
        CardIndex(),
        profile_loader=lambda _card: parse_limitless_profile(CHARMELEON_HTML),
        config=SimilarityConfig(pokemon_min_score=0.99),
    )

    assert not swapper.resolve(RawCard(2, "Charmeleon", "OBF", "27", "pokemon"))


@pytest.mark.parametrize(
    "config",
    [
        SimilarityConfig(text_sequence_weight=0.0),
        SimilarityConfig(text_sequence_weight=1.0),
    ],
)
def test_boundary_weights_are_valid(config):
    assert 0.0 <= config.text_sequence_weight <= 1.0


def test_invalid_similarity_config_is_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        SimilarityConfig(text_sequence_weight=1.1)
