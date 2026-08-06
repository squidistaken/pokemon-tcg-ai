from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import isclose
from pathlib import Path
from typing import Protocol, cast, override

from bs4 import BeautifulSoup

from .card_index import CardIndex, CardProfile, normalize_name, normalize_number
from .http import HttpClient
from .mapping_rules import MappingRuleSet, SetCanonicalizer
from .models import CardSwap, RawCard

CARD_URL = "https://limitlesstcg.com/cards/{set_code}/{number}"
SET_INDEX_URL = "https://limitlesstcg.com/cards"
DEFAULT_CACHE_DIR = Path("outputs/card_swap_cache")
DEFAULT_MAPPING_RULE_DIR = Path(__file__).with_name("card_mappings")

_TYPE_CODES = {
    "grass": "g",
    "fire": "r",
    "water": "w",
    "lightning": "l",
    "psychic": "p",
    "fighting": "f",
    "darkness": "d",
    "dark": "d",
    "metal": "m",
    "colorless": "c",
}
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "up",
    "you",
    "your",
}


@dataclass(frozen=True)
class SimilarityConfig:
    """Weights and acceptance thresholds for gameplay-profile similarity."""

    text_sequence_weight: float = 0.40
    move_name_weight: float = 0.15
    move_cost_weight: float = 0.30
    move_damage_weight: float = 0.20
    move_effect_weight: float = 0.35
    pokemon_text_weight: float = 0.30
    pokemon_hp_weight: float = 0.15
    pokemon_retreat_weight: float = 0.10
    pokemon_moves_weight: float = 0.30
    pokemon_weakness_weight: float = 0.075
    pokemon_resistance_weight: float = 0.075
    pokemon_min_score: float = 0.55
    pokemon_min_move_score: float = 0.40
    non_pokemon_min_score: float = 0.90

    def __post_init__(self) -> None:
        values = vars(self)
        if any(not 0.0 <= value <= 1.0 for value in values.values()):
            raise ValueError("similarity weights and thresholds must be between 0 and 1")
        if not isclose(
            self.move_name_weight
            + self.move_cost_weight
            + self.move_damage_weight
            + self.move_effect_weight,
            1.0,
        ):
            raise ValueError("move similarity weights must sum to 1")
        if not isclose(
            self.pokemon_text_weight
            + self.pokemon_hp_weight
            + self.pokemon_retreat_weight
            + self.pokemon_moves_weight
            + self.pokemon_weakness_weight
            + self.pokemon_resistance_weight,
            1.0,
        ):
            raise ValueError("Pokémon profile weights must sum to 1")


DEFAULT_SIMILARITY_CONFIG = SimilarityConfig()


@dataclass(frozen=True)
class SourceProfile:
    """Gameplay fields parsed from one source card page."""

    name: str
    stage: str
    rule: str
    previous_stage: str | None
    hp: int | None
    energy_type: str | None
    weakness: str | None
    resistance: str | None
    retreat: int | None
    moves: tuple[tuple[str, str, str, str], ...]

    @property
    def searchable_text(self) -> str:
        return " ".join(part for move in self.moves for part in move if _present(part))


@dataclass(frozen=True)
class RankedCandidate:
    card_id: int
    name: str
    score: float


class ProfileLoader(Protocol):
    def __call__(self, card: RawCard) -> SourceProfile | None: ...


class CardSwapper(ABC):
    """Strategy interface for resolving an otherwise unresolved card line."""

    @abstractmethod
    def resolve(self, card: RawCard) -> tuple[CardSwap, ...]:
        """Return ordered competition-legal candidates for one source card."""


class MappingCardSwapper(CardSwapper):
    """Resolve from versioned, approved rules without heuristic fallback."""

    def __init__(
        self,
        index: CardIndex,
        rules_dir: Path | str = DEFAULT_MAPPING_RULE_DIR,
        set_canonicalizer: SetCanonicalizer | None = None,
        *,
        use_rejected_mappings: bool = False,
        minimum_mapping_confidence: int = 1,
    ):
        self.index = index
        self.rules = MappingRuleSet.load(
            rules_dir,
            index,
            set_canonicalizer,
            use_rejected_mappings=use_rejected_mappings,
            minimum_mapping_confidence=minimum_mapping_confidence,
        )

    @override
    def resolve(self, card: RawCard) -> tuple[CardSwap, ...]:
        try:
            return self.rules.candidates_for(card)
        except Exception:  # noqa: BLE001 - metadata failure leaves the card unresolved
            return ()


def _present(value: str | None) -> bool:
    return bool(value and normalize_name(value) not in {"", "n a", "none"})


def _optional_int(value: str | None) -> int | None:
    match = re.search(r"\d+", value or "")
    return int(match.group()) if match else None


def _type_code(value: str | None) -> str | None:
    normalized = normalize_name(value or "")
    if normalized in {"", "n a", "none"}:
        return None
    braced = re.findall(r"\{([A-Za-z])\}", value or "")
    if braced:
        return "".join(letter.lower() for letter in braced)
    return _TYPE_CODES.get(normalized, normalized)


def _rule(value: str | None) -> str:
    normalized = normalize_name(value or "")
    return "" if normalized in {"", "n a", "none"} else normalized


def _tokens(value: str) -> set[str]:
    return set(normalize_name(value).split()) - _STOP_WORDS - {"n", "a"}


def _text_similarity(
    left: str,
    right: str,
    config: SimilarityConfig = DEFAULT_SIMILARITY_CONFIG,
) -> float:
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    dice = (
        2 * len(left_tokens & right_tokens) / (len(left_tokens) + len(right_tokens))
        if left_tokens or right_tokens
        else 1.0
    )
    sequence = SequenceMatcher(None, normalize_name(left), normalize_name(right)).ratio()
    return (1.0 - config.text_sequence_weight) * dice + config.text_sequence_weight * sequence


def _numeric_similarity(left: int | None, right: int | None, scale: int) -> float:
    if left is None and right is None:
        return 1.0
    if left is None or right is None:
        return 0.0
    return max(0.0, 1.0 - abs(left - right) / scale)


def _cost_symbols(value: str) -> Counter[str]:
    symbols = re.findall(r"\{([A-Za-z])\}", value)
    if symbols:
        return Counter(symbol.lower() for symbol in symbols)
    translated = value.replace("●", "C")
    return Counter(char.lower() for char in translated if char.upper() in "GRWLPFDMC")


def _cost_similarity(left: str, right: str) -> float:
    left_cost, right_cost = _cost_symbols(left), _cost_symbols(right)
    total = sum(left_cost.values()) + sum(right_cost.values())
    if not total:
        return 1.0
    return 2 * sum((left_cost & right_cost).values()) / total


def _move_similarity(
    left: tuple[str, str, str, str],
    right: tuple[str, str, str, str],
    config: SimilarityConfig,
) -> float:
    return (
        config.move_name_weight * _text_similarity(left[0], right[0], config)
        + config.move_cost_weight * _cost_similarity(left[1], right[1])
        + config.move_damage_weight
        * _numeric_similarity(_optional_int(left[2]), _optional_int(right[2]), 100)
        + config.move_effect_weight * _text_similarity(left[3], right[3], config)
    )


def _moves_similarity(
    source: tuple[tuple[str, str, str, str], ...],
    target: list[tuple[str, str, str, str]],
    config: SimilarityConfig,
) -> float:
    source_moves = [move for move in source if any(_present(part) for part in move)]
    target_moves = [move for move in target if any(_present(part) for part in move)]
    if not source_moves and not target_moves:
        return 1.0
    if not source_moves or not target_moves:
        return 0.0
    return sum(
        max(_move_similarity(move, candidate, config) for candidate in target_moves)
        for move in source_moves
    ) / max(len(source_moves), len(target_moves))


def _compatible(source: SourceProfile, target: CardProfile) -> bool:
    return (
        normalize_name(source.name) == normalize_name(target.name)
        and source.stage == target.stage
        and _rule(source.rule) == _rule(target.rule)
        and _type_code(source.energy_type) == _type_code(target.energy_type)
        and normalize_name(source.previous_stage or "")
        == normalize_name(target.previous_stage or "")
    )


def _profile_score(
    source: SourceProfile,
    target: CardProfile,
    config: SimilarityConfig,
) -> float:
    if "Pokémon" not in source.stage:
        return _text_similarity(source.searchable_text, target.searchable_text, config)
    return (
        config.pokemon_text_weight
        * _text_similarity(source.searchable_text, target.searchable_text, config)
        + config.pokemon_hp_weight * _numeric_similarity(source.hp, target.hp, 100)
        + config.pokemon_retreat_weight
        * _numeric_similarity(source.retreat, target.retreat, 4)
        + config.pokemon_moves_weight
        * _moves_similarity(source.moves, target.moves, config)
        + config.pokemon_weakness_weight
        * float(_type_code(source.weakness) == _type_code(target.weakness))
        + config.pokemon_resistance_weight
        * float(_type_code(source.resistance) == _type_code(target.resistance))
    )


def rank_candidates(
    source: SourceProfile,
    index: CardIndex,
    config: SimilarityConfig = DEFAULT_SIMILARITY_CONFIG,
) -> tuple[RankedCandidate, ...]:
    """Rank compatible same-name competition variants by gameplay similarity."""
    ranked = [
        RankedCandidate(
            profile.card_id,
            profile.name,
            _profile_score(source, profile, config),
        )
        for profile in index.profiles.values()
        if _compatible(source, profile)
    ]
    return tuple(sorted(ranked, key=lambda item: (-item.score, item.card_id)))


def parse_limitless_profile(html: str) -> SourceProfile:
    """Parse the gameplay fields exposed by a Limitless card page."""
    soup = BeautifulSoup(html, "lxml")
    card_text = soup.select_one(".card-text")
    if card_text is None:
        raise ValueError("Limitless response has no card text")
    name_node = card_text.select_one(".card-text-name")
    type_node = card_text.select_one(".card-text-type")
    title_node = card_text.select_one(".card-text-title")
    if name_node is None or type_node is None or title_node is None:
        raise ValueError("Limitless response is missing card identity fields")

    name = name_node.get_text(" ", strip=True)
    type_text = type_node.get_text(" ", strip=True)
    if type_text.startswith("Pokémon"):
        match = re.search(r"\b(Basic|Stage\s+[12])\b", type_text)
        if match is None:
            raise ValueError("Limitless Pokémon response has no evolution stage")
        stage_name = match.group(1)
        stage = "Basic Pokémon" if stage_name == "Basic" else f"{stage_name} Pokémon"
    elif type_text.startswith("Trainer"):
        stage = type_text.split("-")[-1].strip()
    elif type_text.startswith("Energy"):
        energy_kind = type_text.split("-")[-1].strip()
        stage = energy_kind if energy_kind.endswith("Energy") else f"{energy_kind} Energy"
    else:
        raise ValueError(f"unknown Limitless card type {type_text!r}")

    title_text = title_node.get_text(" ", strip=True)
    hp_match = re.search(r"(\d+)\s+HP", title_text)
    type_match = re.search(r"-\s*([^-]+?)\s*-\s*\d+\s+HP", title_text)
    previous_match = re.search(r"Evolves from\s+(.+)$", type_text)
    all_text = card_text.get_text(" ", strip=True)
    if "ACE SPEC" in all_text:
        rule = "ACE SPEC"
    elif "mega" in normalize_name(name).split() and normalize_name(name).endswith(" ex"):
        rule = "Mega Pokémon ex"
    elif normalize_name(name).endswith(" ex"):
        rule = "Pokémon ex"
    else:
        rule = "n/a"

    wrr_node = card_text.select_one(".card-text-wrr")
    wrr = wrr_node.get_text(" ", strip=True) if wrr_node else ""
    weakness = re.search(r"Weakness:\s*([^\s]+)", wrr)
    resistance = re.search(r"Resistance:\s*([^\s]+)", wrr)
    retreat = re.search(r"Retreat:\s*(\d+)", wrr)

    moves: list[tuple[str, str, str, str]] = []
    for attack in card_text.select(".card-text-attack"):
        info_node = attack.select_one(".card-text-attack-info")
        if info_node is None:
            continue
        cost_node = info_node.select_one(".ptcg-symbol")
        cost = cost_node.get_text("", strip=True) if cost_node else ""
        info = info_node.get_text(" ", strip=True)
        rest = info[len(cost) :].strip() if cost and info.startswith(cost) else info
        damage_match = re.search(r"\s+(\d+[+x×-]?)$", rest)
        damage = damage_match.group(1) if damage_match else ""
        move_name = rest[: damage_match.start()].strip() if damage_match else rest
        effect_node = attack.select_one(".card-text-attack-effect")
        effect = effect_node.get_text(" ", strip=True) if effect_node else ""
        moves.append((move_name, cost, damage, effect))

    for ability in card_text.select(".card-text-ability"):
        moves.append((ability.get_text(" ", strip=True), "", "", ""))

    if not moves:
        sections = [
            section
            for section in card_text.select(".card-text-section")
            if "card-text-artist"
            not in cast(list[str], section.get("class", []))
            and section.select_one(".card-text-title") is None
            and section.select_one(".card-text-wrr") is None
        ]
        effect = " ".join(section.get_text(" ", strip=True) for section in sections)
        moves.append(("", "", "", effect))

    return SourceProfile(
        name=name,
        stage=stage,
        rule=rule,
        previous_stage=previous_match.group(1).strip() if previous_match else None,
        hp=int(hp_match.group(1)) if hp_match else None,
        energy_type=type_match.group(1).strip() if type_match else None,
        weakness=weakness.group(1) if weakness else None,
        resistance=resistance.group(1) if resistance else None,
        retreat=int(retreat.group(1)) if retreat else None,
        moves=tuple(moves),
    )


class LimitlessProfileLoader:
    """Fetch each missing source printing once and cache the ignored HTML locally."""

    def __init__(
        self,
        client: HttpClient | None = None,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
    ):
        self.client = client or HttpClient(min_interval=1.0)
        self.cache_dir = Path(cache_dir)
        self._profiles: dict[tuple[str, str], SourceProfile] = {}
        self._set_codes: dict[str, str] | None = None

    def _load_set_codes(self) -> dict[str, str]:
        if self._set_codes is not None:
            return self._set_codes
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / "sets.html"
        if path.exists():
            html = path.read_text(encoding="utf-8")
        else:
            html = self.client.get_text(SET_INDEX_URL)
            path.write_text(html, encoding="utf-8")
        codes: dict[str, str] = {}
        soup = BeautifulSoup(html, "lxml")
        for link in soup.select('a[href^="/cards/"]'):
            href = str(link.get("href") or "")
            match = re.fullmatch(r"/cards/([A-Za-z0-9]+)", href)
            if match is None:
                continue
            code = match.group(1).upper()
            label = link.get_text(" ", strip=True)
            if normalize_name(label).endswith(f" {normalize_name(code)}"):
                label = label[: -len(code)].strip()
            codes.setdefault(normalize_name(label), code)
            codes.setdefault(normalize_name(code), code)
        self._set_codes = codes
        return codes

    def _set_code(self, value: str) -> str | None:
        if re.fullmatch(r"[A-Za-z0-9]{2,5}", value.strip()):
            return value.upper()
        return self._load_set_codes().get(normalize_name(value))

    def canonical_set_code(self, value: str | None) -> str | None:
        """Resolve a source set code or full expansion name to Limitless's code."""
        if value is None or not value.strip():
            return None
        return self._set_code(value)

    def __call__(self, card: RawCard) -> SourceProfile | None:
        number = normalize_number(card.number)
        if not card.set_code or not number:
            return None
        set_code = self._set_code(card.set_code)
        if set_code is None:
            return None
        key = (set_code, number)
        if key in self._profiles:
            return self._profiles[key]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{key[0]}-{key[1]}.html"
        if path.exists():
            html = path.read_text(encoding="utf-8")
        else:
            html = self.client.get_text(
                CARD_URL.format(set_code=key[0], number=key[1])
            )
            path.write_text(html, encoding="utf-8")
        profile = parse_limitless_profile(html)
        self._profiles[key] = profile
        return profile


class HeuristicCardSwapper(CardSwapper):
    """Resolve ambiguous same-name printings from their gameplay profiles."""

    def __init__(
        self,
        index: CardIndex,
        profile_loader: ProfileLoader | None = None,
        config: SimilarityConfig | None = None,
    ):
        self.index = index
        self.profile_loader = profile_loader or LimitlessProfileLoader()
        self.config = config or DEFAULT_SIMILARITY_CONFIG

    @override
    def resolve(self, card: RawCard) -> tuple[CardSwap, ...]:
        """Return a clear same-name profile match, or no decision when ambiguous."""
        if not self.index.candidates_for_name(card.name):
            return ()
        try:
            source = self.profile_loader(card)
        except Exception:  # noqa: BLE001 - a metadata failure leaves the card unresolved
            return ()
        if source is None or normalize_name(source.name) != normalize_name(card.name):
            return ()
        ranked = rank_candidates(source, self.index, self.config)
        if not ranked:
            return ()
        best = ranked[0]
        min_score = (
            self.config.pokemon_min_score
            if "Pokémon" in source.stage
            else self.config.non_pokemon_min_score
        )
        if best.score < min_score:
            return ()
        move_score = _moves_similarity(
            source.moves,
            self.index.profiles[best.card_id].moves,
            self.config,
        )
        if "Pokémon" in source.stage and move_score < self.config.pokemon_min_move_score:
            return ()
        return (
            CardSwap(
                source_name=card.name,
                source_set=card.set_code,
                source_number=normalize_number(card.number),
                count=max(0, card.count),
                target_id=best.card_id,
                target_name=best.name,
                kind="variant",
                confidence=round(best.score, 4),
                rationale=(
                    f"same-name gameplay profile score {best.score:.3f}; "
                    f"move score {move_score:.3f}"
                ),
            ),
        )
