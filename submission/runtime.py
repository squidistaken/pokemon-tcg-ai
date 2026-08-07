"""Torch-only Kaggle inference runtime for the structured PPO checkpoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntEnum
from itertools import pairwise
from typing import Any, ClassVar

import torch
from torch import nn
from torch.nn import functional as nn_functional

if globals().get("__package__"):
    from .cg_api import (
        AreaType,
        Observation,
        Option,
        OptionType,
        PlayerState,
        Pokemon,
        SelectData,
        State,
    )
else:
    from cg_api import (
        AreaType,
        Observation,
        Option,
        OptionType,
        PlayerState,
        Pokemon,
        SelectData,
        State,
    )


def checkpoint_state_dict(payload: object) -> Mapping[str, torch.Tensor]:
    """Return tensors from a versioned or legacy actor-critic checkpoint."""
    state_dict: object = payload
    if isinstance(payload, Mapping) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Checkpoint does not contain a non-empty state dict.")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise ValueError("Checkpoint state dict must map string keys to tensors.")
    return state_dict


class OptionReferenceResolver:
    """Resolve engine option references to concrete card and attack IDs."""

    @staticmethod
    def resolve_target_pokemon(
        state: State,
        option: Option,
        agent_seat: int,
    ) -> Pokemon | None:
        """
        Return the in-play Pokemon an option acts on, mirroring the training
        resolver. ``target_id`` carries only the target's card ID, which is
        shared by every copy of that card; this is what lets the option row
        carry that instance's live state.
        """
        owner_index = (
            option.playerIndex if option.playerIndex is not None else agent_seat
        )
        if option.type in (
            OptionType.TOOL_CARD,
            OptionType.ENERGY_CARD,
            OptionType.ENERGY,
        ):
            return OptionReferenceResolver._pokemon_at(
                state, owner_index, option.area, option.index
            )
        if option.type in (OptionType.ATTACH, OptionType.EVOLVE):
            return OptionReferenceResolver._pokemon_at(
                state, owner_index, option.inPlayArea, option.inPlayIndex
            )
        return None

    @staticmethod
    def resolve(
        state: State,
        select: SelectData,
        option: Option,
        agent_seat: int,
    ) -> tuple[int, int, int]:
        """Return ``(card_id, target_id, attack_id)`` for one option."""
        option_type = option.type
        owner_index = (
            option.playerIndex if option.playerIndex is not None else agent_seat
        )
        if option_type == OptionType.CARD:
            return (
                OptionReferenceResolver._card_id_at(
                    state, select, owner_index, option.area, option.index
                ),
                0,
                0,
            )
        if option_type in (
            OptionType.TOOL_CARD,
            OptionType.ENERGY_CARD,
            OptionType.ENERGY,
        ):
            pokemon = OptionReferenceResolver._pokemon_at(
                state, owner_index, option.area, option.index
            )
            if pokemon is None:
                return 0, 0, 0
            attachment_index = (
                option.toolIndex
                if option_type == OptionType.TOOL_CARD
                else option.energyIndex
            )
            attachments = (
                pokemon.tools
                if option_type == OptionType.TOOL_CARD
                else pokemon.energyCards
            )
            return (
                OptionReferenceResolver._card_in_list(attachments, attachment_index),
                pokemon.id,
                0,
            )
        if option_type in (OptionType.PLAY, OptionType.ABILITY, OptionType.DISCARD):
            return (
                OptionReferenceResolver._card_id_at(
                    state,
                    select,
                    owner_index,
                    option.area or AreaType.HAND,
                    option.index,
                ),
                0,
                0,
            )
        if option_type in (OptionType.ATTACH, OptionType.EVOLVE):
            played = OptionReferenceResolver._card_id_at(
                state, select, owner_index, option.area, option.index
            )
            target = OptionReferenceResolver._pokemon_at(
                state, owner_index, option.inPlayArea, option.inPlayIndex
            )
            return played, target.id if target is not None else 0, 0
        if option_type == OptionType.ATTACK:
            return 0, 0, option.attackId if option.attackId is not None else 0
        if option_type == OptionType.SKILL:
            return option.cardId if option.cardId is not None else 0, 0, 0
        return 0, 0, 0

    @staticmethod
    def _card_id_at(
        state: State,
        select: SelectData,
        player_index: int,
        area: AreaType | None,
        index: int | None,
    ) -> int:
        if area is None or index is None:
            return 0
        player = state.players[player_index]
        if area == AreaType.HAND:
            return (
                OptionReferenceResolver._card_in_list(player.hand, index)
                if player.hand is not None
                else 0
            )
        if area == AreaType.DISCARD:
            return OptionReferenceResolver._card_in_list(player.discard, index)
        if area in (AreaType.ACTIVE, AreaType.BENCH):
            pokemon = OptionReferenceResolver._pokemon_at(
                state, player_index, area, index
            )
            return pokemon.id if pokemon is not None else 0
        if area == AreaType.PRIZE:
            return OptionReferenceResolver._card_in_list(player.prize, index)
        if area == AreaType.STADIUM:
            return OptionReferenceResolver._card_in_list(state.stadium, index)
        if area == AreaType.DECK:
            return (
                OptionReferenceResolver._card_in_list(select.deck, index)
                if select.deck is not None
                else 0
            )
        if area == AreaType.LOOKING:
            return (
                OptionReferenceResolver._card_in_list(state.looking, index)
                if state.looking is not None
                else 0
            )
        return 0

    @staticmethod
    def _pokemon_at(
        state: State,
        player_index: int,
        area: AreaType | None,
        index: int | None,
    ) -> Pokemon | None:
        if area is None or index is None:
            return None
        player = state.players[player_index]
        if area == AreaType.ACTIVE:
            slots = player.active
        elif area == AreaType.BENCH:
            slots = player.bench
        else:
            return None
        return slots[index] if 0 <= index < len(slots) else None

    @staticmethod
    def _card_in_list(cards: Sequence[Any], index: int | None) -> int:
        if index is None or not 0 <= index < len(cards):
            return 0
        card = cards[index]
        return card.id if card is not None else 0


class StructuredObservationEncoder:
    """Torch-only equivalent of the training structured encoder."""

    GLOBAL_FEATURE_COUNT = 41
    OPTION_CATEGORICAL_COUNT = 4
    OPTION_SCALAR_COUNT = 6
    OPTION_TARGET_FEATURE_COUNT = 7
    POKEMON_FEATURE_COUNT = 20
    ENERGY_TYPE_COUNT = 12

    def __init__(
        self,
        max_options: int,
        bench_cap: int = 8,
        hand_cap: int = 60,
        discard_cap: int = 60,
        prize_cap: int = 6,
        deck_cap: int = 60,
        looking_cap: int = 60,
        energy_cap: int = 60,
        evolution_cap: int = 2,
    ) -> None:
        self._max_options = max_options
        self._bench_cap = bench_cap
        self._hand_cap = hand_cap
        self._discard_cap = discard_cap
        self._prize_cap = prize_cap
        self._deck_cap = deck_cap
        self._looking_cap = looking_cap
        self._energy_cap = energy_cap
        self._evolution_cap = evolution_cap
        self._pokemon_rows = 2 * (1 + bench_cap)

    def encode(
        self,
        observation: Observation,
        agent_seat: int,
        already_chosen_option_count: int,
    ) -> dict[str, Any]:
        """Encode one raw engine observation into nested torch tensors."""
        state = observation.current
        if state is None:
            raise ValueError("observation.current must be set")
        select = observation.select
        return {
            "globals": self._encode_globals(
                state, select, agent_seat, already_chosen_option_count
            ),
            "select_cats": self._encode_select_cats(select),
            "context_card_ids": self._encode_context_cards(select),
            "stadium_id": torch.tensor(
                [state.stadium[0].id if state.stadium else 0], dtype=torch.int64
            ),
            "options": self._encode_options(state, select, agent_seat),
            "pokemon": self._encode_pokemon(state, agent_seat),
            "my": self._encode_my_zones(state, agent_seat),
            "opp": self._encode_opp_zones(state, agent_seat),
            "select_deck": self._encode_id_list(
                [card.id for card in select.deck]
                if select is not None and select.deck is not None
                else [],
                self._deck_cap,
            ),
            "looking": self._encode_id_list(
                [card.id if card is not None else 0 for card in state.looking]
                if state.looking is not None
                else [],
                self._looking_cap,
            ),
        }

    @staticmethod
    def _encode_globals(
        state: State,
        select: SelectData | None,
        agent_seat: int,
        already_chosen_option_count: int,
    ) -> torch.Tensor:
        first_player = (
            -1.0 if state.firstPlayer == -1 else float(state.firstPlayer == agent_seat)
        )
        features = [
            float(state.turn),
            float(state.turnActionCount),
            float(agent_seat),
            first_player,
            float(state.supporterPlayed),
            float(state.stadiumPlayed),
            float(state.energyAttached),
            float(state.retreated),
        ]
        if select is None:
            features += [0.0] * 9
        else:
            features += [
                1.0,
                float(select.minCount),
                float(select.maxCount),
                float(len(select.option)),
                float(already_chosen_option_count),
                float(select.remainDamageCounter),
                float(select.remainEnergyCost),
                float(select.deck is not None),
                float(state.looking is not None),
            ]
        for player in (state.players[agent_seat], state.players[1 - agent_seat]):
            active = player.active[0] if player.active else None
            features += [
                float(player.deckCount),
                float(player.handCount),
                float(len(player.prize)),
                float(len(player.bench)),
                float(player.benchMax),
                float(bool(player.active)),
                float(bool(player.active) and active is None),
                float(player.poisoned),
                float(player.burned),
                float(player.asleep),
                float(player.paralyzed),
                float(player.confused),
            ]
        if len(features) != StructuredObservationEncoder.GLOBAL_FEATURE_COUNT:
            raise RuntimeError(f"Expected 41 global features, got {len(features)}")
        return torch.tensor(features, dtype=torch.float32)

    @staticmethod
    def _encode_select_cats(select: SelectData | None) -> torch.Tensor:
        if select is None:
            return torch.zeros(2, dtype=torch.int64)
        return torch.tensor(
            [int(select.type) + 1, int(select.context) + 1], dtype=torch.int64
        )

    @staticmethod
    def _encode_context_cards(select: SelectData | None) -> torch.Tensor:
        if select is None:
            return torch.zeros(2, dtype=torch.int64)
        return torch.tensor(
            [
                select.contextCard.id if select.contextCard is not None else 0,
                select.effect.id if select.effect is not None else 0,
            ],
            dtype=torch.int64,
        )

    @staticmethod
    def _shifted_category(value: IntEnum | None) -> int:
        return int(value) + 1 if value is not None else 0

    @staticmethod
    def _float_or_absent(value: int | None) -> float:
        return float(value) if value is not None else -1.0

    def _encode_options(
        self, state: State, select: SelectData | None, agent_seat: int
    ) -> dict[str, torch.Tensor]:
        n_slots = self._max_options + 1
        entries = {
            "card_id": torch.zeros(n_slots, dtype=torch.int64),
            "target_id": torch.zeros(n_slots, dtype=torch.int64),
            "attack_id": torch.zeros(n_slots, dtype=torch.int64),
            "owner": torch.zeros(n_slots, dtype=torch.int64),
            "cats": torch.zeros(
                (n_slots, self.OPTION_CATEGORICAL_COUNT), dtype=torch.int64
            ),
            "scalars": torch.full(
                (n_slots, self.OPTION_SCALAR_COUNT), -1.0, dtype=torch.float32
            ),
            "target_state": torch.zeros(
                (n_slots, self.OPTION_TARGET_FEATURE_COUNT), dtype=torch.float32
            ),
        }
        options = select.option if select is not None else []
        if len(options) > self._max_options:
            raise ValueError(
                f"Selection has {len(options)} options, exceeding {self._max_options}"
            )
        for slot, option in enumerate(options):
            entries["cats"][slot] = torch.tensor(
                [
                    int(option.type) + 1,
                    self._shifted_category(option.area),
                    self._shifted_category(option.inPlayArea),
                    self._shifted_category(option.specialConditionType),
                ]
            )
            if option.playerIndex is not None:
                entries["owner"][slot] = 1 if option.playerIndex == agent_seat else 2
            entries["scalars"][slot] = torch.tensor(
                [
                    self._float_or_absent(option.number),
                    self._float_or_absent(option.count),
                    self._float_or_absent(option.index),
                    self._float_or_absent(option.toolIndex),
                    self._float_or_absent(option.energyIndex),
                    self._float_or_absent(option.inPlayIndex),
                ]
            )
            card_id, target_id, attack_id = OptionReferenceResolver.resolve(
                state, select, option, agent_seat
            )
            entries["card_id"][slot] = card_id
            entries["target_id"][slot] = target_id
            entries["attack_id"][slot] = attack_id
            target = OptionReferenceResolver.resolve_target_pokemon(
                state, option, agent_seat
            )
            if target is not None:
                owner = (
                    option.playerIndex
                    if option.playerIndex is not None
                    else agent_seat
                )
                is_active = any(
                    entry is target for entry in state.players[owner].active
                )
                entries["target_state"][slot] = torch.tensor(
                    [
                        1.0,
                        float(target.hp),
                        float(target.maxHp),
                        target.hp / target.maxHp if target.maxHp > 0 else 0.0,
                        float(len(target.energyCards)),
                        float(len(target.tools)),
                        1.0 if is_active else 0.0,
                    ]
                )
        return entries

    def _encode_pokemon(self, state: State, agent_seat: int) -> dict[str, torch.Tensor]:
        entries = {
            "card_id": torch.zeros(self._pokemon_rows, dtype=torch.int64),
            "tool_id": torch.zeros(self._pokemon_rows, dtype=torch.int64),
            "energy_card_ids": torch.zeros(
                (self._pokemon_rows, self._energy_cap), dtype=torch.int64
            ),
            "pre_evolution_ids": torch.zeros(
                (self._pokemon_rows, self._evolution_cap), dtype=torch.int64
            ),
            "features": torch.zeros(
                (self._pokemon_rows, self.POKEMON_FEATURE_COUNT),
                dtype=torch.float32,
            ),
            "mask": torch.zeros(self._pokemon_rows, dtype=torch.bool),
        }
        for side, player_index in enumerate((agent_seat, 1 - agent_seat)):
            player = state.players[player_index]
            base_row = side * (1 + self._bench_cap)
            if player.active:
                entries["mask"][base_row] = True
                self._fill_pokemon_row(
                    entries, player.active[0], base_row, is_active=True
                )
            for offset, pokemon in enumerate(player.bench[: self._bench_cap]):
                row = base_row + 1 + offset
                entries["mask"][row] = True
                self._fill_pokemon_row(entries, pokemon, row, is_active=False)
        return entries

    def _fill_pokemon_row(
        self,
        entries: dict[str, torch.Tensor],
        pokemon: Pokemon | None,
        row: int,
        *,
        is_active: bool,
    ) -> None:
        features = entries["features"]
        features[row, 4] = float(is_active)
        if pokemon is None:
            return
        entries["card_id"][row] = pokemon.id
        if pokemon.tools:
            entries["tool_id"][row] = pokemon.tools[0].id
        for column, card in enumerate(pokemon.energyCards[: self._energy_cap]):
            entries["energy_card_ids"][row, column] = card.id
        for column, card in enumerate(pokemon.preEvolution[: self._evolution_cap]):
            entries["pre_evolution_ids"][row, column] = card.id
        features[row, 0] = float(pokemon.hp)
        features[row, 1] = float(pokemon.maxHp)
        features[row, 2] = pokemon.hp / pokemon.maxHp if pokemon.maxHp > 0 else 0.0
        features[row, 3] = float(pokemon.appearThisTurn)
        features[row, 5] = float(len(pokemon.tools))
        features[row, 6] = float(len(pokemon.energyCards))
        features[row, 7] = float(len(pokemon.preEvolution))
        for energy in pokemon.energies:
            energy_index = int(energy)
            if 0 <= energy_index < self.ENERGY_TYPE_COUNT:
                features[row, 8 + energy_index] += 1.0

    def _encode_my_zones(
        self, state: State, agent_seat: int
    ) -> dict[str, torch.Tensor]:
        player = state.players[agent_seat]
        entries = self._encode_public_zones(player)
        hand_ids = [card.id for card in player.hand] if player.hand is not None else []
        hand = self._stage_id_list(hand_ids, self._hand_cap)
        entries["hand_ids"] = hand["ids"]
        entries["hand_mask"] = hand["mask"]
        # Match the training Composite insertion order: hand, discard, prize.
        return {
            "hand_ids": entries["hand_ids"],
            "hand_mask": entries["hand_mask"],
            "discard_ids": entries["discard_ids"],
            "discard_mask": entries["discard_mask"],
            "prize_ids": entries["prize_ids"],
            "prize_mask": entries["prize_mask"],
        }

    def _encode_opp_zones(
        self, state: State, agent_seat: int
    ) -> dict[str, torch.Tensor]:
        return self._encode_public_zones(state.players[1 - agent_seat])

    def _encode_public_zones(self, player: PlayerState) -> dict[str, torch.Tensor]:
        discard = self._stage_id_list(
            [card.id for card in player.discard], self._discard_cap
        )
        prize = self._stage_id_list(
            [card.id if card is not None else 0 for card in player.prize],
            self._prize_cap,
        )
        return {
            "discard_ids": discard["ids"],
            "discard_mask": discard["mask"],
            "prize_ids": prize["ids"],
            "prize_mask": prize["mask"],
        }

    @staticmethod
    def _stage_id_list(card_ids: list[int], capacity: int) -> dict[str, torch.Tensor]:
        card_ids = card_ids[:capacity]
        ids = torch.zeros(capacity, dtype=torch.int64)
        mask = torch.zeros(capacity, dtype=torch.bool)
        if card_ids:
            ids[: len(card_ids)] = torch.tensor(card_ids, dtype=torch.int64)
            mask[: len(card_ids)] = True
        return {"ids": ids, "mask": mask}

    def _encode_id_list(
        self, card_ids: list[int], capacity: int
    ) -> dict[str, torch.Tensor]:
        return self._stage_id_list(card_ids, capacity)


class StructuredObsAdapter(nn.Module):
    """Torch-only equivalent of the trained structured observation adapter."""

    _card_static: torch.Tensor
    _attack_static: torch.Tensor
    _card_categories: torch.Tensor
    _card_attack_ids: torch.Tensor
    _select_category_offsets: torch.Tensor
    _option_category_offsets: torch.Tensor
    _card_category_offsets: torch.Tensor
    _global_scales: torch.Tensor
    _pokemon_feature_scales: torch.Tensor
    _option_target_scales: torch.Tensor

    CATEGORY_VOCAB_SIZE = 64
    SELECT_CATEGORY_FIELD_COUNT = 2
    OPTION_CATEGORY_FIELD_COUNT = 4
    CARD_CATEGORY_FIELD_COUNT = 4
    OWNER_VALUE_COUNT = 3
    OPTION_SCALAR_SCALE = 60.0
    #: Mirrors the training adapter's OPTION_TARGET_SCALES.
    OPTION_TARGET_SCALES = (1.0, 400.0, 400.0, 1.0, 16.0, 2.0, 1.0)
    GAME_SCALES = (50.0, 20.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    SELECT_SCALES = (1.0, 6.0, 6.0, 96.0, 6.0, 30.0, 5.0, 1.0, 1.0)
    PLAYER_SCALES = (
        60.0,
        15.0,
        6.0,
        8.0,
        8.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    POKEMON_FEATURE_SCALES = (
        400.0,
        400.0,
        1.0,
        1.0,
        1.0,
        2.0,
        16.0,
        2.0,
    ) + (4.0,) * 12
    ZONE_PAIRS: ClassVar[dict[str, tuple[tuple[str, str], ...]]] = {
        "my": (
            ("hand_ids", "hand_mask"),
            ("discard_ids", "discard_mask"),
            ("prize_ids", "prize_mask"),
        ),
        "opp": (
            ("discard_ids", "discard_mask"),
            ("prize_ids", "prize_mask"),
        ),
        "select_deck": (("ids", "mask"),),
        "looking": (("ids", "mask"),),
    }

    def __init__(
        self,
        state_dict: Mapping[str, torch.Tensor],
        card_embed_dim: int,
        attack_embed_dim: int,
        category_embed_dim: int,
        zone_pooling: str = "mean",
        emit_option_tokens: bool = False,
        pokemon_seat_split: bool = False,
        option_target_state: bool = False,
    ) -> None:
        super().__init__()
        if zone_pooling not in ("mean", "mean_max_sum"):
            raise ValueError(f"Unsupported zone_pooling: {zone_pooling}")
        self._zone_pooling = zone_pooling
        self._emit_option_tokens = emit_option_tokens
        self._pokemon_seat_split = pokemon_seat_split
        self._option_target_state = option_target_state
        card_static = state_dict["backbone.adapter._card_static"]
        attack_static = state_dict["backbone.adapter._attack_static"]
        card_categories = state_dict["backbone.adapter._card_categories"]
        card_attack_ids = state_dict["backbone.adapter._card_attack_ids"]
        self.register_buffer("_card_static", torch.zeros_like(card_static))
        self.register_buffer("_attack_static", torch.zeros_like(attack_static))
        self.register_buffer("_card_categories", torch.zeros_like(card_categories))
        self.register_buffer("_card_attack_ids", torch.zeros_like(card_attack_ids))
        self.register_buffer(
            "_select_category_offsets",
            torch.zeros(self.SELECT_CATEGORY_FIELD_COUNT, dtype=torch.int64),
        )
        self.register_buffer(
            "_option_category_offsets",
            torch.zeros(self.OPTION_CATEGORY_FIELD_COUNT, dtype=torch.int64),
        )
        self.register_buffer(
            "_card_category_offsets",
            torch.zeros(self.CARD_CATEGORY_FIELD_COUNT, dtype=torch.int64),
        )
        self.register_buffer(
            "_global_scales",
            torch.tensor(
                self.GAME_SCALES + self.SELECT_SCALES + 2 * self.PLAYER_SCALES,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "_pokemon_feature_scales",
            torch.tensor(self.POKEMON_FEATURE_SCALES, dtype=torch.float32),
        )
        self.register_buffer(
            "_option_target_scales",
            torch.tensor(self.OPTION_TARGET_SCALES, dtype=torch.float32),
            persistent=False,
        )
        self._card_embedding = nn.Embedding(
            card_static.shape[0], card_embed_dim, padding_idx=0
        )
        self._attack_embedding = nn.Embedding(
            attack_static.shape[0], attack_embed_dim, padding_idx=0
        )
        field_count = (
            self.SELECT_CATEGORY_FIELD_COUNT
            + self.OPTION_CATEGORY_FIELD_COUNT
            + self.CARD_CATEGORY_FIELD_COUNT
        )
        self._category_embedding = nn.Embedding(
            field_count * self.CATEGORY_VOCAB_SIZE, category_embed_dim
        )
        # Per-entity projection + masked-mean pooling, detected from the saved
        # weights rather than the config: checkpoints trained before the pooling
        # path carry no projection tensors, and must keep loading as the flat
        # padded encoding they were trained with. Widths come from the weights
        # for the same reason -- the Kaggle sandbox has no obs_spec to size them.
        card_proj_weight = state_dict.get("backbone.adapter._card_proj.weight")
        self._pool = card_proj_weight is not None
        if self._pool:
            assert card_proj_weight is not None
            option_weight = state_dict["backbone.adapter._option_encoder.weight"]
            pokemon_weight = state_dict["backbone.adapter._pokemon_encoder.weight"]
            entity_dim = card_proj_weight.shape[0]
            #: Width of one per-entity encoding, or None on the legacy flat
            #: path. A token-sequence backbone sizes its entity projections
            #: from this (the training adapter's ``_entity_dim``).
            self.entity_dim: int | None = int(entity_dim)
            self._card_proj: nn.Module = nn.Linear(
                card_proj_weight.shape[1], entity_dim
            )
            self._option_encoder: nn.Module = nn.Linear(
                option_weight.shape[1], entity_dim
            )
            self._pokemon_encoder: nn.Module = nn.Linear(
                pokemon_weight.shape[1], entity_dim
            )
        else:
            self.entity_dim = None
            self._card_proj = nn.Identity()
            self._option_encoder = nn.Identity()
            self._pokemon_encoder = nn.Identity()

    def encode_groups(
        self, observation: Mapping[str, Any], group_names: Sequence[str]
    ) -> list[torch.Tensor]:
        """Encode each group separately, in ``group_names`` order."""
        parts: list[torch.Tensor] = []
        for name in group_names:
            value = observation[name]
            if name == "globals":
                parts.append(value / self._global_scales)
            elif name == "select_cats":
                parts.append(self._embed_categories(value, self._select_category_offsets))
            elif name in ("context_card_ids", "stadium_id"):
                parts.append(self._encode_card_ids(value))
            elif name == "options":
                parts.append(self._encode_options(value))
            elif name == "pokemon":
                parts.append(self._encode_pokemon(value))
            else:
                parts.append(self._encode_zone_group(name, value))
        return parts

    def encode_entity_tokens(
        self, observation: Mapping[str, Any], groups: Sequence[str]
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode selected groups as unpooled per-entity token sequences.

        Mirrors the training adapter's ``encode_entity_tokens``: one token per
        option row, per Pokemon row, per card in a zone, taken from the same
        per-entity encoders the pooled path uses. Padding slots stay in place
        (the slot count is fixed) and are reported through the mask.

        :return: ``{name: (tokens, valid_mask)}``.
        """
        if not self._pool:
            raise ValueError(
                "Per-entity tokens need the adapter's entity encoders, which this "
                "checkpoint was trained without."
            )
        tokens: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name in groups:
            if name in ("globals", "select_cats"):
                raise ValueError(
                    f"Group '{name}' is a single feature vector with no entity axis."
                )
            value = observation[name]
            if name == "options":
                tokens[name] = (
                    self._option_encoder(self._option_rows(value)),
                    self._option_validity(value),
                )
            elif name == "pokemon":
                tokens[name] = (
                    self._pokemon_encoder(self._pokemon_rows(value)),
                    value["mask"],
                )
            elif name in ("context_card_ids", "stadium_id"):
                tokens[name] = (self._card_proj(self._card_repr(value)), value != 0)
            else:
                card_parts: list[torch.Tensor] = []
                mask_parts: list[torch.Tensor] = []
                for ids_name, mask_name in self.ZONE_PAIRS[name]:
                    card_parts.append(
                        self._card_proj(self._card_repr(value[ids_name]))
                    )
                    mask_parts.append(value[mask_name])
                tokens[name] = (
                    torch.cat(card_parts, dim=-2),
                    torch.cat(mask_parts, dim=-1),
                )
        return tokens

    def forward(
        self, observation: Mapping[str, Any], group_names: Sequence[str]
    ) -> torch.Tensor:
        return torch.cat(self.encode_groups(observation, group_names), dim=-1)

    def _embed_categories(
        self, values: torch.Tensor, field_offsets: torch.Tensor
    ) -> torch.Tensor:
        """Embed categorical fields through the shared table, flattening the result."""
        return self._flatten_rows(self._category_embedding(values + field_offsets))

    def _card_repr(self, card_ids: torch.Tensor) -> torch.Tensor:
        """
        Learned + static + categorical + attack-pool card representation.

        Mirrors the training adapter's ``_card_repr``.
        """
        attack_ids = self._card_attack_ids[card_ids]
        return torch.cat(
            [
                self._card_embedding(card_ids),
                self._card_static[card_ids],
                self._embed_categories(self._card_categories[card_ids], self._card_category_offsets),
                self._masked_mean(self._attack_repr(attack_ids), attack_ids != 0),
            ],
            dim=-1,
        )

    def _attack_repr(self, attack_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self._attack_embedding(attack_ids), self._attack_static[attack_ids]],
            dim=-1,
        )

    @staticmethod
    def _flatten_rows(table: torch.Tensor) -> torch.Tensor:
        return table.reshape(*table.shape[:-2], -1)

    @staticmethod
    def _masked_mean(reprs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(torch.float32).unsqueeze(-1)
        return (reprs * weights).sum(dim=-2) / weights.sum(dim=-2).clamp(min=1.0)

    def _masked_pool(self, reprs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Collapse a padded set of entity encodings into a fixed-width summary.

        Mirrors the training adapter's ``_masked_pool``: under ``"mean"``
        this is the plain centroid; under ``"mean_max_sum"`` the max and a
        capacity-normalized sum are concatenated alongside it.
        """
        mean = self._masked_mean(reprs, mask)
        if self._zone_pooling == "mean":
            return mean
        occupied = mask.unsqueeze(-1)
        maximum = reprs.masked_fill(~occupied, torch.finfo(reprs.dtype).min).amax(dim=-2)
        maximum = torch.where(mask.any(dim=-1, keepdim=True), maximum, torch.zeros_like(maximum))
        total = (reprs * occupied.to(reprs.dtype)).sum(dim=-2) / reprs.shape[-2]
        return torch.cat([mean, maximum, total], dim=-1)

    @staticmethod
    def _option_validity(options: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """
        A slot is real when the encoder wrote an option type into it, plus
        the final slot, which is the always-present synthetic stop action.

        Mirrors the training adapter's ``_option_validity``: this cannot be
        ``card_id != 0``, because ``card_id`` comes from
        ``OptionReferenceResolver.resolve``, which returns 0 for
        YES/NO/NUMBER/RETREAT/END options -- they have no associated card, so
        a selection offering only those would look like every slot is
        padding, and the pooled ``options`` vector would be exactly zero.
        """
        # `!= 0` allocates a new tensor rather than viewing `cats`, so writing
        # the stop slot below does not mutate the caller's input.
        validity = options["cats"][..., 0] != 0
        validity[..., -1] = True
        return validity

    def segment_ids(self, name: str, observation: Mapping[str, Any]) -> torch.Tensor:
        """
        Rederive one group's per-slot segment ids from its own slot count.

        The training adapter registers these as non-persistent buffers (see
        ``StructuredObsAdapter._register_group``/``group_segment_ids``): they
        carry no trained information and are rederived identically from
        ``obs_spec`` on every construction, so they are never in a checkpoint
        state dict. This class has no ``obs_spec`` to read, but the same
        values fall straight out of the observation's own tensor shapes,
        which is what the caps on :class:`StructuredObservationEncoder`
        ultimately determine anyway. A backbone adds a learned embedding
        indexed by this id to each slot's per-entity token, to restore
        identity that the shared per-entity encoders otherwise erase (which
        seat a Pokemon belongs to, which zone a card sits in, whether an
        option slot is real or the synthetic stop action).

        :param name: Group name; a key of :attr:`ZONE_PAIRS`, or one of
            ``context_card_ids``/``stadium_id``/``options``/``pokemon``.
        :param observation: The full nested observation, keyed by group name.
        :return: Int64 tensor of shape ``(n_slots,)``.
        """
        value = observation[name]
        if name == "context_card_ids":
            return torch.arange(value.shape[-1], dtype=torch.int64)
        if name == "stadium_id":
            return torch.zeros(value.shape[-1], dtype=torch.int64)
        if name == "options":
            n_slots = value["card_id"].shape[-1]
            ids = torch.zeros(n_slots, dtype=torch.int64)
            ids[-1] = 1
            return ids
        if name == "pokemon":
            rows = value["card_id"].shape[-1]
            half = rows // 2
            return torch.cat(
                [
                    torch.zeros(half, dtype=torch.int64),
                    torch.ones(rows - half, dtype=torch.int64),
                ]
            )
        pairs = self.ZONE_PAIRS[name]
        return torch.cat(
            [
                torch.full((value[ids_name].shape[-1],), segment, dtype=torch.int64)
                for segment, (ids_name, _) in enumerate(pairs)
            ]
        )

    def _encode_card_ids(self, card_ids: torch.Tensor) -> torch.Tensor:
        return self._flatten_rows(self._card_proj(self._card_repr(card_ids)))

    def _option_rows(self, options: Mapping[str, torch.Tensor]) -> torch.Tensor:
        scalars = options["scalars"]
        scaled = torch.where(
            scalars < 0.0,
            scalars.new_full((), -1.0),
            scalars / self.OPTION_SCALAR_SCALE,
        )
        return torch.cat(
            [
                self._card_repr(options["card_id"]),
                self._card_repr(options["target_id"]),
                self._attack_repr(options["attack_id"]),
                nn_functional.one_hot(options["owner"], self.OWNER_VALUE_COUNT).to(
                    torch.float32
                ),
                self._embed_categories(options["cats"], self._option_category_offsets),
                scaled,
            ]
            + (
                [options["target_state"] / self._option_target_scales]
                if self._option_target_state
                else []
            ),
            dim=-1,
        )


    def _pokemon_rows(self, pokemon: Mapping[str, torch.Tensor]) -> torch.Tensor:
        energy_ids = pokemon["energy_card_ids"]
        evolution_ids = pokemon["pre_evolution_ids"]
        return torch.cat(
            [
                self._card_repr(pokemon["card_id"]),
                self._card_repr(pokemon["tool_id"]),
                self._masked_mean(self._card_repr(energy_ids), energy_ids != 0),
                self._masked_mean(self._card_repr(evolution_ids), evolution_ids != 0),
                pokemon["features"] / self._pokemon_feature_scales,
                pokemon["mask"].to(torch.float32).unsqueeze(-1),
            ],
            dim=-1,
        )

    def _encode_options(self, options: Mapping[str, torch.Tensor]) -> torch.Tensor:
        rows = self._option_rows(options)
        if self._pool:
            return self._masked_mean(
                self._option_encoder(rows), self._option_validity(options)
            )
        return self._flatten_rows(rows)

    def _encode_pokemon(self, pokemon: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """
        Mirrors the training adapter's ``_encode_pokemon``, including the
        ``pokemon_seat_split`` layout: the row axis is agent active + bench
        then opponent active + bench, so the two seats are its two halves.
        """
        rows = self._pokemon_rows(pokemon)
        if not self._pool:
            return self._flatten_rows(rows)
        encoded = self._pokemon_encoder(rows)
        mask = pokemon["mask"]
        if not self._pokemon_seat_split:
            return self._masked_pool(encoded, mask)
        half = encoded.shape[-2] // 2
        return torch.cat(
            [
                self._masked_pool(encoded[..., :half, :], mask[..., :half]),
                self._masked_pool(encoded[..., half:, :], mask[..., half:]),
            ],
            dim=-1,
        )

    def _encode_zone_group(
        self, name: str, zones: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for ids_name, mask_name in self.ZONE_PAIRS[name]:
            mask = zones[mask_name]
            capacity = mask.shape[-1]
            reprs = self._card_proj(self._card_repr(zones[ids_name]))
            parts.append(
                self._masked_pool(reprs, mask) if self._pool
                else self._masked_mean(reprs, mask)
            )
            parts.append(mask.to(torch.float32).sum(dim=-1, keepdim=True) / capacity)
        return torch.cat(parts, dim=-1)


def _activation(name: str) -> type[nn.Module]:
    activations = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
    }
    try:
        return activations[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported activation: {name}") from error


def _mlp(
    in_features: int,
    hidden: Sequence[int],
    out_features: int,
    activation: str,
) -> nn.Sequential:
    widths = [in_features, *hidden, out_features]
    layers: list[nn.Module] = []
    activation_class = _activation(activation)
    for index, (input_width, output_width) in enumerate(pairwise(widths)):
        layers.append(nn.Linear(input_width, output_width))
        if index < len(widths) - 2:
            layers.append(activation_class())
    return nn.Sequential(*layers)


_DEFAULT_IN_KEYS = (
    ["observation", "globals"],
    ["observation", "select_cats"],
    ["observation", "context_card_ids"],
    ["observation", "stadium_id"],
    ["observation", "options"],
    ["observation", "pokemon"],
    ["observation", "my"],
    ["observation", "opp"],
    ["observation", "select_deck"],
    ["observation", "looking"],
)


def _group_names(backbone_config: Mapping[str, Any]) -> list[str]:
    """Structured group each in-key names, in the order the trunk consumes."""
    raw_in_keys = backbone_config.get("in_keys", _DEFAULT_IN_KEYS)
    return [
        key[-1] if isinstance(key, Sequence) and not isinstance(key, str) else key
        for key in raw_in_keys
    ]


def _head_needs_option_repr(config: Mapping[str, Any]) -> bool:
    """
    Whether the checkpoint's policy head scores per-option tokens.

    Read off the head's own class rather than sniffed from a state-dict key:
    the two pointer heads name their parameters differently, so any single key
    to look for silently misses one of them and the runtime then builds a
    backbone that emits nothing for it to score.

    :param config: The checkpoint's embedded model config.
    :return: True when the head cannot run on ``state_repr`` alone.
    """
    head_class = _POLICY_HEADS.get(str(config["head"]["_target_"]))
    return bool(head_class is not None and head_class.requires_option_repr)


def _build_adapter(
    state_dict: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
) -> StructuredObsAdapter:
    """Rebuild the trained observation adapter from the embedded widths."""
    adapter_config = config.get("adapter", {})
    return StructuredObsAdapter(
        state_dict,
        card_embed_dim=int(adapter_config.get("card_embed_dim", 8)),
        attack_embed_dim=int(adapter_config.get("attack_embed_dim", 8)),
        category_embed_dim=int(adapter_config.get("category_embed_dim", 4)),
        zone_pooling=str(adapter_config.get("zone_pooling", "mean")),
        pokemon_seat_split=bool(adapter_config.get("pokemon_seat_split", False)),
        option_target_state=bool(adapter_config.get("option_target_state", False)),
    )


class MLPBackbone(nn.Module):
    """
    Structured adapter followed by the trained MLP trunk.

    By default this trunk emits ``state_repr`` only. Set the checkpoint's
    ``backbone.option_tokens`` to also emit per-option tokens as
    ``option_repr`` -- one ``Linear`` projection per option row plus that
    group's stop-slot segment embedding (mirrors
    :class:`TransformerBackbone`'s cheap, non-attention option path), for
    pairing with :class:`PointerPolicyHead`.
    """

    def __init__(
        self,
        state_dict: Mapping[str, torch.Tensor],
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        adapter_config = config.get("adapter", {})
        backbone_config = config["backbone"]
        # Mirrors build_actor_critic: the head decides whether per-option tokens
        # are needed at all, and the trunk's own option_tokens decides which of
        # the two sources supplies them. Derived rather than assumed because
        # older checkpoints predate the pointer head entirely and must keep
        # loading exactly as they were trained.
        emit_option_tokens = _head_needs_option_repr(config) and not bool(
            backbone_config.get("option_tokens", False)
        )
        self.adapter = StructuredObsAdapter(
            state_dict,
            card_embed_dim=int(adapter_config.get("card_embed_dim", 8)),
            attack_embed_dim=int(adapter_config.get("attack_embed_dim", 8)),
            category_embed_dim=int(adapter_config.get("category_embed_dim", 4)),
            zone_pooling=str(adapter_config.get("zone_pooling", "mean")),
            emit_option_tokens=emit_option_tokens,
            pokemon_seat_split=bool(adapter_config.get("pokemon_seat_split", False)),
            option_target_state=bool(adapter_config.get("option_target_state", False)),
        )
        self.adapter_option_tokens = emit_option_tokens
        self.group_names = _group_names(backbone_config)
        input_dim = state_dict["backbone.mlp.0.weight"].shape[1]
        self.mlp = _mlp(
            input_dim,
            [int(value) for value in backbone_config["num_cells"]],
            int(config["embed_dim"]),
            str(backbone_config.get("activation", "tanh")),
        )
        self.option_tokens = bool(backbone_config.get("option_tokens", False))
        # Either path supplies the pointer head: the trunk projects the option
        # rows itself, or the adapter hands them over unprojected. Training
        # rejects both at once, so at most one is ever set.
        self.produces_option_repr = self.option_tokens or self.adapter_option_tokens
        if self.option_tokens:
            option_weight = state_dict["backbone.option_projection.weight"]
            self.option_projection: nn.Module = nn.Linear(
                option_weight.shape[1], option_weight.shape[0]
            )
            # Persisted only once the training code grew this parameter (see
            # StructuredObsAdapter.segment_ids); absent on checkpoints from
            # before then, which this backbone must still load and serve
            # exactly as they were trained, i.e. with no segment identity
            # added to the option tokens at all.
            segment_weight = state_dict.get("backbone.option_segment_embedding")
            if segment_weight is not None:
                self.option_segment_embedding: nn.Parameter | None = nn.Parameter(
                    torch.zeros_like(segment_weight)
                )
            else:
                self.option_segment_embedding = None

    def forward(
        self, observation: Mapping[str, Any]
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        state_repr = self.mlp(self.adapter(observation, self.group_names))
        if self.adapter_option_tokens:
            option_rows, _ = self.adapter.encode_entity_tokens(
                observation, ["options"]
            )["options"]
            return state_repr, option_rows
        if not self.option_tokens:
            return state_repr
        option_rows, _ = self.adapter.encode_entity_tokens(observation, ["options"])[
            "options"
        ]
        option_repr = self.option_projection(option_rows)
        if self.option_segment_embedding is not None:
            segment_ids = self.adapter.segment_ids("options", observation)
            option_repr = option_repr + self.option_segment_embedding[segment_ids]
        return state_repr, option_repr


class TransformerBackbone(nn.Module):
    """
    Structured adapter followed by the trained self-attention trunk.

    Torch ships :class:`nn.TransformerEncoder`, so this reuses the very module
    the training backbone builds (same submodule names, hence the same
    checkpoint keys) rather than reimplementing attention. What it does port is
    everything around it: one token per observation group, the learned type
    embeddings, the opt-in per-entity token groups, the three readouts, and the
    per-option tokens a pointer head scores.

    Widths come from the saved weights, not the config, for the same reason
    :class:`MLPBackbone` reads its input width there: the Kaggle sandbox has no
    observation spec to size the per-group projections from.
    """

    def __init__(
        self,
        state_dict: Mapping[str, torch.Tensor],
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.adapter = _build_adapter(state_dict, config)
        backbone_config = config["backbone"]
        self.group_names = _group_names(backbone_config)
        out_features = int(config["embed_dim"])
        self.out_features = out_features
        self.pooling = str(backbone_config.get("pooling", "mean"))
        if self.pooling not in ("mean", "cls", "attention"):
            raise ValueError(f"Unknown transformer pooling '{self.pooling}'")
        self.token_groups = [
            str(name) for name in backbone_config.get("token_groups") or ()
        ]
        self.option_tokens = bool(backbone_config.get("option_tokens", False))
        self.produces_option_repr = self.option_tokens
        self.encoded_option_repr = bool(
            backbone_config.get("encoded_option_repr", False)
        )
        self.replace_pooled = bool(backbone_config.get("replace_pooled", False))

        # For every name in token_groups, replace_pooled drops that group's
        # single pooled encode_groups() token (and its token_projections
        # entry / token_type_embedding row) from the sequence, so the
        # per-entity tokens become the only route by which that group
        # reaches the trunk. False (the default) keeps one pooled token per
        # registered group, same as before this flag existed.
        dropped_pooled = (
            frozenset(self.token_groups) if self.replace_pooled else frozenset()
        )
        self._dropped_pooled_groups = dropped_pooled
        self.pooled_group_names = [
            name for name in self.group_names if name not in dropped_pooled
        ]
        self.token_projections = nn.ModuleList(
            [
                nn.Linear(
                    state_dict[f"backbone.token_projections.{index}.weight"].shape[1],
                    out_features,
                )
                for index in range(len(self.pooled_group_names))
            ]
        )
        self.token_type_embedding = nn.Parameter(
            torch.zeros(len(self.pooled_group_names), out_features)
        )

        self._needs_entity_tokens = sorted(
            set(self.token_groups) | ({"options"} if self.option_tokens else set())
        )
        entity_dim = self.adapter.entity_dim
        if self._needs_entity_tokens and entity_dim is None:
            raise ValueError(
                "Per-entity tokens need the adapter's entity encoders, which this "
                "checkpoint was trained without."
            )
        self.entity_projections = nn.ModuleDict(
            {
                name: nn.Linear(int(entity_dim or 0), out_features)
                for name in self._needs_entity_tokens
            }
        )
        self.entity_type_embedding = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(out_features))
                for name in self._needs_entity_tokens
            }
        )
        # One learned vector per segment id (StructuredObsAdapter.
        # segment_ids), added to that slot's projected token -- the stop
        # slot's own identity, and the seat/zone identity a pooled group
        # vector otherwise erases. Persisted only once the training code
        # grew this parameter; built per-name from whichever of these keys
        # the checkpoint actually has, so an older checkpoint trained before
        # it existed still loads (and serves) exactly as it was trained,
        # with no segment identity added to its entity tokens at all.
        self.entity_segment_embedding = nn.ParameterDict(
            {
                name: nn.Parameter(
                    torch.zeros_like(
                        state_dict[f"backbone.entity_segment_embedding.{name}"]
                    )
                )
                for name in self._needs_entity_tokens
                if f"backbone.entity_segment_embedding.{name}" in state_dict
            }
        )

        if self.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(out_features))
        elif self.pooling == "attention":
            self.pool_query = nn.Parameter(torch.zeros(out_features))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_features,
            nhead=int(backbone_config.get("num_heads", 4)),
            dim_feedforward=int(backbone_config.get("ff_dim", 256)),
            dropout=float(backbone_config.get("dropout", 0.0)),
            activation=str(backbone_config.get("activation", "gelu")),
            norm_first=bool(backbone_config.get("norm_first", False)),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(backbone_config.get("num_layers", 1)),
            norm=(
                nn.LayerNorm(out_features)
                if bool(backbone_config.get("final_norm", False))
                else None
            ),
            enable_nested_tensor=False,
        )

    def _project_entity_tokens(
        self,
        name: str,
        tokens: torch.Tensor,
        observation: Mapping[str, Any],
    ) -> torch.Tensor:
        """Project one group's raw per-entity encodings and add its identity."""
        projected = (
            self.entity_projections[name](tokens) + self.entity_type_embedding[name]
        )
        if name in self.entity_segment_embedding:
            segment_ids = self.adapter.segment_ids(name, observation)
            projected = projected + self.entity_segment_embedding[name][segment_ids]
        return projected

    def forward(
        self, observation: Mapping[str, Any]
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the observation into ``state_repr`` (and per-option tokens).

        :return: ``state_repr``, or ``(state_repr, option_repr)`` when the
            checkpoint was trained with ``option_tokens``.
        """
        # encode_groups always computes every registered group's pooled
        # vector; replace_pooled only changes which of those get projected
        # and concatenated below, not whether the adapter computes them.
        group_vectors = self.adapter.encode_groups(observation, self.group_names)
        kept_vectors = [
            vector
            for name, vector in zip(self.group_names, group_vectors, strict=True)
            if name not in self._dropped_pooled_groups
        ]
        tokens = torch.stack(
            [
                projection(vector)
                for projection, vector in zip(
                    self.token_projections, kept_vectors, strict=True
                )
            ],
            dim=-2,
        )
        tokens = tokens + self.token_type_embedding
        valid = tokens.new_ones(tokens.shape[:-1], dtype=torch.bool)

        entity_tokens: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if self._needs_entity_tokens:
            entity_tokens = self.adapter.encode_entity_tokens(
                observation, self._needs_entity_tokens
            )
        # Offsets recorded while assembling the sequence (not from a static
        # formula) because pooling="cls" prepends its token *after* this
        # loop, shifting every later index by one (handled below), and
        # replace_pooled changes how many pooled tokens precede the entity
        # tokens.
        entity_offsets: dict[str, int] = {}
        for name in self.token_groups:
            group_tokens, group_valid = entity_tokens[name]
            entity_offsets[name] = tokens.shape[-2]
            projected = self._project_entity_tokens(name, group_tokens, observation)
            tokens = torch.cat([tokens, projected], dim=-2)
            valid = torch.cat([valid, group_valid], dim=-1)

        if self.pooling == "cls":
            cls = self.cls_token.expand(*tokens.shape[:-2], 1, tokens.shape[-1])
            tokens = torch.cat([cls, tokens], dim=-2)
            valid = torch.cat([valid.new_ones(*valid.shape[:-1], 1), valid], dim=-1)
            entity_offsets = {
                name: offset + 1 for name, offset in entity_offsets.items()
            }

        # The encoder takes a single leading batch dim; flatten any extra ones
        # and restore them once the token dimension is pooled away. Serving
        # passes one unbatched observation, which flattens to batch 1.
        batch_shape = tokens.shape[:-2]
        flat_tokens = tokens.reshape(-1, tokens.shape[-2], tokens.shape[-1])
        flat_valid = valid.reshape(-1, valid.shape[-1])
        # Only the per-entity path introduces padding; mirror training and
        # decide from config, so an unpadded sequence takes the same
        # (mask-free) attention path it took while training.
        padding_mask = ~flat_valid if self.token_groups else None
        encoded = self.encoder(flat_tokens, src_key_padding_mask=padding_mask)

        if self.pooling == "cls":
            pooled = encoded[:, 0, :]
        elif self.pooling == "attention":
            scores = (encoded @ self.pool_query) / (self.out_features**0.5)
            scores = scores.masked_fill(~flat_valid, float("-inf"))
            pooled = (scores.softmax(dim=-1).unsqueeze(-1) * encoded).sum(dim=-2)
        else:
            weights = flat_valid.to(encoded.dtype).unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=-2) / weights.sum(dim=-2).clamp(
                min=1.0
            )
        state_repr = pooled.reshape(*batch_shape, self.out_features)

        if not self.option_tokens:
            return state_repr
        if self.encoded_option_repr:
            n_options = entity_tokens["options"][0].shape[-2]
            start = entity_offsets["options"]
            option_repr = encoded[:, start : start + n_options, :].reshape(
                *batch_shape, n_options, self.out_features
            )
        else:
            option_rows, _ = entity_tokens["options"]
            option_repr = self._project_entity_tokens(
                "options", option_rows, observation
            )
        return state_repr, option_repr


class LinearPolicyHead(nn.Module):
    """Linear action-logit head with checkpoint-compatible names."""

    #: This head reads only ``state_repr``.
    requires_option_repr = False

    def __init__(
        self,
        in_features: int,
        n_actions: int,
        head_config: Mapping[str, Any],
        state_dict: Mapping[str, torch.Tensor],
    ) -> None:
        del head_config, state_dict
        super().__init__()
        self.linear = nn.Linear(in_features, n_actions)

    def forward(
        self,
        state_repr: torch.Tensor,
        option_repr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del option_repr
        return self.linear(state_repr)


class PointerPolicyHead(nn.Module):
    """
    Pointer head: score each per-option token against a state-derived query.

    Scoring an option's own representation, rather than the slot it happens to
    occupy, is what lets the policy learn *what* an option does. The option
    table already carries one row per action slot (including the synthetic
    stop), so scoring every row yields exactly the expected logits.
    """

    #: This head cannot run on ``state_repr`` alone.
    requires_option_repr = True

    def __init__(
        self,
        in_features: int,
        n_actions: int,
        head_config: Mapping[str, Any],
        state_dict: Mapping[str, torch.Tensor],
    ) -> None:
        del head_config
        super().__init__()
        self.n_actions = n_actions
        # The query is projected to the option width, which equals in_features
        # only when the tokens come from a trunk emitting them at its own
        # width. Off the adapter they are entity_dim wide instead, so the
        # width is measured rather than assumed.
        self.option_dim = int(state_dict["policy_head.query.weight"].shape[0])
        self.query = nn.Linear(in_features, self.option_dim)
        self._scale = float(self.option_dim) ** 0.5

    def forward(
        self,
        state_repr: torch.Tensor,
        option_repr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if option_repr is None:
            raise ValueError("PointerPolicyHead needs per-option tokens.")
        if option_repr.shape[-2] != self.n_actions:
            raise ValueError(
                f"option_repr has {option_repr.shape[-2]} slots but the action space "
                f"has {self.n_actions}."
            )
        query = self.query(state_repr).unsqueeze(-2)
        return (option_repr * query).sum(dim=-1) / self._scale


class PointerHead(nn.Module):
    """
    Pointer head scoring ``[state_repr, option_repr_i]`` with a shared MLP.

    The dot-product :class:`PointerPolicyHead` above compresses that comparison
    into a single scaled product; this variant runs a shared two-layer scorer
    over the concatenation instead, which is the training default. The synthetic
    stop action has no option row to score, so its logit comes from a separate
    state-only branch rather than from a row of the option table.

    Widths come from the saved weights rather than the config: ``scorer.0``
    consumes ``state_repr`` and one option token side by side, so its input
    width minus ``in_features`` is the option width the backbone must emit.

    :param in_features: Width of the incoming ``state_repr``.
    :param n_actions: Size of the action space; the last index is stop.
    :param head_config: Trained head config, read for the hidden activation.
    :param state_dict: Checkpoint weights, measured for the layer widths.
    """

    #: This head cannot run on ``state_repr`` alone.
    requires_option_repr = True

    def __init__(
        self,
        in_features: int,
        n_actions: int,
        head_config: Mapping[str, Any],
        state_dict: Mapping[str, torch.Tensor],
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.n_option_slots = n_actions - 1
        scorer_input = state_dict["policy_head.scorer.0.weight"].shape[1]
        self.option_dim = int(scorer_input) - in_features
        if self.option_dim <= 0:
            raise ValueError(
                f"PointerHead scorer takes {scorer_input} inputs, which leaves no "
                f"option width alongside a {in_features}-wide state."
            )
        # Depth measured from the weights, not from head_config["num_cells"]:
        # training defaults that key to None and derives a single hidden layer
        # from in_features, so a checkpoint that took the default records
        # layers the config does not mention.
        hidden = []
        index = 0
        while f"policy_head.scorer.{index + 2}.weight" in state_dict:
            hidden.append(int(state_dict[f"policy_head.scorer.{index}.weight"].shape[0]))
            index += 2
        activation = str(head_config.get("activation", "tanh"))
        self.scorer = _mlp(in_features + self.option_dim, hidden, 1, activation)
        self.stop_scorer = _mlp(in_features, hidden, 1, activation)

    def forward(
        self,
        state_repr: torch.Tensor,
        option_repr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if option_repr is None:
            raise ValueError("PointerHead needs per-option tokens.")
        if option_repr.shape[-2] < self.n_option_slots:
            raise ValueError(
                f"option_repr has {option_repr.shape[-2]} slots but the action space "
                f"needs {self.n_option_slots} scored options."
            )
        options = option_repr[..., : self.n_option_slots, :]
        broadcast_state = state_repr.unsqueeze(-2).expand(
            *state_repr.shape[:-1], self.n_option_slots, state_repr.shape[-1]
        )
        option_logits = self.scorer(
            torch.cat([broadcast_state, options], dim=-1)
        ).squeeze(-1)
        return torch.cat([option_logits, self.stop_scorer(state_repr)], dim=-1)


class ValueHead(nn.Module):
    """Checkpoint-compatible critic, retained only for strict loading."""

    def __init__(self, in_features: int, num_cells: Sequence[int]) -> None:
        super().__init__()
        self.mlp = _mlp(in_features, num_cells, 1, "tanh")


#: Training backbones this runtime can rebuild, by their config ``_target_``.
_BACKBONES: dict[str, type[nn.Module]] = {
    "src.models.mlp.MLPBackbone": MLPBackbone,
    "src.models.transformer.TransformerBackbone": TransformerBackbone,
}
#: Training policy heads this runtime can rebuild, by their config ``_target_``.
_POLICY_HEADS: dict[str, type[nn.Module]] = {
    "src.models.heads.LinearPolicyHead": LinearPolicyHead,
    "src.models.heads.PointerPolicyHead": PointerPolicyHead,
    "src.models.heads.PointerHead": PointerHead,
}


class ActorCritic(nn.Module):
    """Checkpoint-compatible torch-only actor-critic."""

    def __init__(
        self,
        state_dict: Mapping[str, torch.Tensor],
        config: Mapping[str, Any],
        max_options: int,
    ) -> None:
        super().__init__()
        backbone_target = config["backbone"].get("_target_")
        head_target = config["head"].get("_target_")
        if backbone_target not in _BACKBONES:
            raise ValueError(
                f"Kaggle runtime cannot rebuild backbone {backbone_target!r}; "
                f"supported: {sorted(_BACKBONES)}"
            )
        if head_target not in _POLICY_HEADS:
            raise ValueError(
                f"Kaggle runtime cannot rebuild policy head {head_target!r}; "
                f"supported: {sorted(_POLICY_HEADS)}"
            )
        embed_dim = int(config["embed_dim"])
        self.backbone = _BACKBONES[backbone_target](state_dict, config)
        # Checked before the head is built, not after: a pointer head sizes its
        # projections against option tokens, so on a trunk that emits none it
        # would fail first on its own missing weights and bury the real reason.
        head_class = _POLICY_HEADS[head_target]
        if head_class.requires_option_repr and not self.backbone.produces_option_repr:
            raise ValueError(
                f"Head {head_class.__name__} needs per-option tokens, but "
                f"backbone {type(self.backbone).__name__} does not emit them."
            )
        self.policy_head = head_class(
            embed_dim, max_options + 1, config["head"], state_dict
        )
        self.value_head = ValueHead(
            embed_dim, [int(value) for value in config["value_head"]["num_cells"]]
        )

    def policy_logits(self, observation: Mapping[str, Any]) -> torch.Tensor:
        """Action logits for one observation, skipping the critic."""
        encoded = self.backbone(observation)
        if self.backbone.produces_option_repr:
            state_repr, option_repr = encoded
        else:
            state_repr, option_repr = encoded, None
        return self.policy_head(state_repr, option_repr)


class Policy:
    """
    Torch-only greedy or sampling policy used by the Kaggle entry point.

    Both modes score one legal action at a time and re-encode the partial-pick
    count before the next choice, matching the selection sequence seen during
    training. Greedy mode is deterministic; sample mode draws without
    replacement from the masked learned distribution.
    """

    def __init__(self, payload: object, config: Mapping[str, Any]) -> None:
        env_config = config["env"]
        if env_config.get("encoder", "structured") != "structured":
            raise ValueError("Kaggle runtime supports only the structured encoder")
        self.max_options = int(env_config["max_options"])
        inference_config = config.get("inference", {})
        if not isinstance(inference_config, Mapping):
            raise TypeError("inference config must be a mapping")
        self.action_selection = str(
            inference_config.get("action_selection", "sample")
        )
        if self.action_selection not in {"greedy", "sample"}:
            raise ValueError(
                "inference.action_selection must be either 'greedy' or 'sample'"
            )
        state_dict = checkpoint_state_dict(payload)
        self.encoder = StructuredObservationEncoder(max_options=self.max_options)
        self.model = ActorCritic(
            state_dict, config["model"], max_options=self.max_options
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, observation: Observation) -> list[int]:
        select = observation.select
        state = observation.current
        if select is None or state is None:
            raise ValueError("Policy requires a current selection and game state")
        n_options = len(select.option)
        min_count = select.minCount
        max_count = select.maxCount
        if not 0 <= min_count <= max_count <= n_options:
            raise ValueError(
                "Invalid selection bounds: expected "
                f"0 <= minCount ({min_count}) <= maxCount ({max_count}) "
                f"<= option count ({n_options})."
            )
        if n_options > self.max_options:
            raise ValueError(
                f"Selection offers {n_options} options but the checkpoint supports "
                f"only {self.max_options}."
            )
        if max_count == 0:
            return []

        picks: list[int] = []
        stop_index = self.max_options
        mask = torch.zeros(self.max_options + 1, dtype=torch.bool)
        mask[:n_options] = True
        while len(picks) < max_count:
            if len(picks) >= min_count:
                mask[stop_index] = True
            encoded = self.encoder.encode(observation, state.yourIndex, len(picks))
            logits = self.model.policy_logits(encoded)
            chosen = self._select_action(logits, mask)
            if chosen == stop_index:
                break
            picks.append(chosen)
            mask[chosen] = False
        return picks

    def _select_action(self, logits: torch.Tensor, mask: torch.Tensor) -> int:
        """Choose one legal action according to the configured serving mode."""
        if logits.ndim != 1 or logits.shape != mask.shape:
            raise ValueError(
                "Expected matching one-dimensional logits/mask, got "
                f"{tuple(logits.shape)} and {tuple(mask.shape)}."
            )
        if not bool(mask.any()):
            raise ValueError("Cannot select an action from an empty mask.")
        masked_logits = logits.masked_fill(~mask, -torch.inf)
        if self.action_selection == "greedy":
            return int(torch.argmax(masked_logits).item())
        probabilities = torch.softmax(masked_logits, dim=0)
        return int(torch.multinomial(probabilities, 1).item())
