"""Pure-Python subset of the competition observation API used for inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class AreaType(IntEnum):
    """Card zones defined by the competition observation schema."""

    DECK = 1
    HAND = 2
    DISCARD = 3
    ACTIVE = 4
    BENCH = 5
    PRIZE = 6
    STADIUM = 7
    ENERGY = 8
    TOOL = 9
    PRE_EVOLUTION = 10
    PLAYER = 11
    LOOKING = 12


class OptionType(IntEnum):
    """Selection option types defined by the competition observation schema."""

    NUMBER = 0
    YES = 1
    NO = 2
    CARD = 3
    TOOL_CARD = 4
    ENERGY_CARD = 5
    ENERGY = 6
    PLAY = 7
    ATTACH = 8
    EVOLVE = 9
    ABILITY = 10
    DISCARD = 11
    RETREAT = 12
    ATTACK = 13
    END = 14
    SKILL = 15
    SPECIAL_CONDITION = 16


@dataclass(frozen=True)
class Card:
    id: int
    serial: int
    playerIndex: int


@dataclass(frozen=True)
class Pokemon:
    id: int
    serial: int
    hp: int
    maxHp: int
    appearThisTurn: bool
    energies: list[int]
    energyCards: list[Card]
    tools: list[Card]
    preEvolution: list[Card]


@dataclass(frozen=True)
class PlayerState:
    active: list[Pokemon | None]
    bench: list[Pokemon]
    benchMax: int
    deckCount: int
    discard: list[Card]
    prize: list[Card | None]
    handCount: int
    hand: list[Card] | None
    poisoned: bool
    burned: bool
    asleep: bool
    paralyzed: bool
    confused: bool


@dataclass(frozen=True)
class State:
    turn: int
    turnActionCount: int
    yourIndex: int
    firstPlayer: int
    supporterPlayed: bool
    stadiumPlayed: bool
    energyAttached: bool
    retreated: bool
    result: int
    stadium: list[Card]
    looking: list[Card | None] | None
    players: list[PlayerState]


@dataclass(frozen=True)
class Option:
    type: int
    number: int | None = None
    area: int | None = None
    index: int | None = None
    playerIndex: int | None = None
    toolIndex: int | None = None
    energyIndex: int | None = None
    count: int | None = None
    inPlayArea: int | None = None
    inPlayIndex: int | None = None
    attackId: int | None = None
    cardId: int | None = None
    serial: int | None = None
    specialConditionType: int | None = None


@dataclass(frozen=True)
class SelectData:
    type: int
    context: int
    minCount: int
    maxCount: int
    remainDamageCounter: int
    remainEnergyCost: int
    option: list[Option]
    deck: list[Card] | None
    contextCard: Card | None
    effect: Card | None


@dataclass(frozen=True)
class Observation:
    select: SelectData | None
    logs: list[Mapping[str, Any]]
    current: State | None
    search_begin_input: str | None = None


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _card(value: Mapping[str, Any] | None) -> Card | None:
    if value is None:
        return None
    return Card(
        id=int(value["id"]),
        serial=int(value["serial"]),
        playerIndex=int(value["playerIndex"]),
    )


def _cards(values: list[Mapping[str, Any]]) -> list[Card]:
    return [card for value in values if (card := _card(value)) is not None]


def _pokemon(value: Mapping[str, Any] | None) -> Pokemon | None:
    if value is None:
        return None
    return Pokemon(
        id=int(value["id"]),
        serial=int(value["serial"]),
        hp=int(value["hp"]),
        maxHp=int(value["maxHp"]),
        appearThisTurn=bool(value["appearThisTurn"]),
        energies=[int(energy) for energy in value["energies"]],
        energyCards=_cards(value["energyCards"]),
        tools=_cards(value["tools"]),
        preEvolution=_cards(value["preEvolution"]),
    )


def _player(value: Mapping[str, Any]) -> PlayerState:
    hand = value["hand"]
    return PlayerState(
        active=[_pokemon(pokemon) for pokemon in value["active"]],
        bench=[
            pokemon
            for entry in value["bench"]
            if (pokemon := _pokemon(entry)) is not None
        ],
        benchMax=int(value["benchMax"]),
        deckCount=int(value["deckCount"]),
        discard=_cards(value["discard"]),
        prize=[_card(card) for card in value["prize"]],
        handCount=int(value["handCount"]),
        hand=None if hand is None else _cards(hand),
        poisoned=bool(value["poisoned"]),
        burned=bool(value["burned"]),
        asleep=bool(value["asleep"]),
        paralyzed=bool(value["paralyzed"]),
        confused=bool(value["confused"]),
    )


def _state(value: Mapping[str, Any] | None) -> State | None:
    if value is None:
        return None
    looking = value["looking"]
    return State(
        turn=int(value["turn"]),
        turnActionCount=int(value["turnActionCount"]),
        yourIndex=int(value["yourIndex"]),
        firstPlayer=int(value["firstPlayer"]),
        supporterPlayed=bool(value["supporterPlayed"]),
        stadiumPlayed=bool(value["stadiumPlayed"]),
        energyAttached=bool(value["energyAttached"]),
        retreated=bool(value["retreated"]),
        result=int(value["result"]),
        stadium=_cards(value["stadium"]),
        looking=None if looking is None else [_card(card) for card in looking],
        players=[_player(player) for player in value["players"]],
    )


def _option(value: Mapping[str, Any]) -> Option:
    return Option(
        type=int(value["type"]),
        number=_optional_int(value.get("number")),
        area=_optional_int(value.get("area")),
        index=_optional_int(value.get("index")),
        playerIndex=_optional_int(value.get("playerIndex")),
        toolIndex=_optional_int(value.get("toolIndex")),
        energyIndex=_optional_int(value.get("energyIndex")),
        count=_optional_int(value.get("count")),
        inPlayArea=_optional_int(value.get("inPlayArea")),
        inPlayIndex=_optional_int(value.get("inPlayIndex")),
        attackId=_optional_int(value.get("attackId")),
        cardId=_optional_int(value.get("cardId")),
        serial=_optional_int(value.get("serial")),
        specialConditionType=_optional_int(value.get("specialConditionType")),
    )


def _select(value: Mapping[str, Any] | None) -> SelectData | None:
    if value is None:
        return None
    deck = value["deck"]
    return SelectData(
        type=int(value["type"]),
        context=int(value["context"]),
        minCount=int(value["minCount"]),
        maxCount=int(value["maxCount"]),
        remainDamageCounter=int(value["remainDamageCounter"]),
        remainEnergyCost=int(value["remainEnergyCost"]),
        option=[_option(option) for option in value["option"]],
        deck=None if deck is None else _cards(deck),
        contextCard=_card(value["contextCard"]),
        effect=_card(value["effect"]),
    )


def to_observation_class(value: Mapping[str, Any]) -> Observation:
    """Convert Kaggle's observation mapping into inference dataclasses."""
    return Observation(
        select=_select(value["select"]),
        logs=list(value.get("logs", [])),
        current=_state(value["current"]),
        search_begin_input=value.get("search_begin_input"),
    )
