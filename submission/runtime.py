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
    POKEMON_FEATURE_COUNT = 20
    ENERGY_TYPE_COUNT = 12

    def __init__(
        self,
        max_options: int,
        bench_cap: int = 8,
        hand_cap: int = 30,
        discard_cap: int = 60,
        prize_cap: int = 6,
        deck_cap: int = 60,
        looking_cap: int = 60,
        energy_cap: int = 40,
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
    _select_category_offsets: torch.Tensor
    _option_category_offsets: torch.Tensor
    _global_scales: torch.Tensor
    _pokemon_feature_scales: torch.Tensor

    CATEGORY_VOCAB_SIZE = 64
    OWNER_VALUE_COUNT = 3
    OPTION_SCALAR_SCALE = 60.0
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
    ) -> None:
        super().__init__()
        if zone_pooling not in ("mean", "mean_max_sum"):
            raise ValueError(f"Unsupported zone_pooling: {zone_pooling}")
        self._zone_pooling = zone_pooling
        self._emit_option_tokens = emit_option_tokens
        card_static = state_dict["backbone.adapter._card_static"]
        attack_static = state_dict["backbone.adapter._attack_static"]
        self.register_buffer("_card_static", torch.zeros_like(card_static))
        self.register_buffer("_attack_static", torch.zeros_like(attack_static))
        self.register_buffer(
            "_select_category_offsets", torch.zeros(2, dtype=torch.int64)
        )
        self.register_buffer(
            "_option_category_offsets", torch.zeros(4, dtype=torch.int64)
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
        self._card_embedding = nn.Embedding(
            card_static.shape[0], card_embed_dim, padding_idx=0
        )
        self._attack_embedding = nn.Embedding(
            attack_static.shape[0], attack_embed_dim, padding_idx=0
        )
        self._category_embedding = nn.Embedding(
            6 * self.CATEGORY_VOCAB_SIZE, category_embed_dim
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
            self._card_proj = nn.Identity()
            self._option_encoder = nn.Identity()
            self._pokemon_encoder = nn.Identity()

    def forward(
        self, observation: Mapping[str, Any], group_names: Sequence[str]
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        parts: list[torch.Tensor] = []
        option_tokens: torch.Tensor | None = None
        for name in group_names:
            value = observation[name]
            if name == "globals":
                parts.append(value / self._global_scales)
            elif name == "select_cats":
                parts.append(
                    self._flatten_rows(
                        self._category_embedding(value + self._select_category_offsets)
                    )
                )
            elif name in ("context_card_ids", "stadium_id"):
                parts.append(self._encode_card_ids(value))
            elif name == "options":
                parts.append(self._encode_options(value))
                if self._emit_option_tokens:
                    option_tokens = self._option_encoder(self._option_rows(value))
            elif name == "pokemon":
                parts.append(self._encode_pokemon(value))
            else:
                parts.append(self._encode_zone_group(name, value))
        state = torch.cat(parts, dim=-1)
        if option_tokens is None:
            return state
        return state, option_tokens

    def _card_repr(self, card_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self._card_embedding(card_ids), self._card_static[card_ids]], dim=-1
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
        """Set pooling mirroring the training-side adapter's ``zone_pooling``."""
        mean = self._masked_mean(reprs, mask)
        if self._zone_pooling == "mean":
            return mean
        occupied = mask.unsqueeze(-1)
        maximum = reprs.masked_fill(~occupied, torch.finfo(reprs.dtype).min).amax(dim=-2)
        maximum = torch.where(
            mask.any(dim=-1, keepdim=True), maximum, torch.zeros_like(maximum)
        )
        total = (reprs * occupied.to(reprs.dtype)).sum(dim=-2) / reprs.shape[-2]
        return torch.cat([mean, maximum, total], dim=-1)

    def _encode_card_ids(self, card_ids: torch.Tensor) -> torch.Tensor:
        return self._flatten_rows(self._card_proj(self._card_repr(card_ids)))

    def _option_rows(self, options: Mapping[str, torch.Tensor]) -> torch.Tensor:
        scalars = options["scalars"]
        scaled = torch.where(
            scalars < 0.0,
            scalars.new_full((), -1.0),
            scalars / self.OPTION_SCALAR_SCALE,
        )
        rows = torch.cat(
            [
                self._card_repr(options["card_id"]),
                self._card_repr(options["target_id"]),
                self._attack_repr(options["attack_id"]),
                nn_functional.one_hot(options["owner"], self.OWNER_VALUE_COUNT).to(
                    torch.float32
                ),
                self._category_embedding(
                    options["cats"] + self._option_category_offsets
                ).flatten(-2),
                scaled,
            ],
            dim=-1,
        )
        return rows

    def _encode_options(self, options: Mapping[str, torch.Tensor]) -> torch.Tensor:
        rows = self._option_rows(options)
        if self._pool:
            return self._masked_mean(
                self._option_encoder(rows), options["card_id"] != 0
            )
        return self._flatten_rows(rows)

    def _encode_pokemon(self, pokemon: Mapping[str, torch.Tensor]) -> torch.Tensor:
        energy_ids = pokemon["energy_card_ids"]
        evolution_ids = pokemon["pre_evolution_ids"]
        rows = torch.cat(
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
        if self._pool:
            return self._masked_pool(self._pokemon_encoder(rows), pokemon["mask"])
        return self._flatten_rows(rows)

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


class MLPBackbone(nn.Module):
    """Structured adapter followed by the trained MLP trunk."""

    def __init__(
        self,
        state_dict: Mapping[str, torch.Tensor],
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        adapter_config = config.get("adapter", {})
        # Both of these are read off the checkpoint rather than assumed: older
        # checkpoints predate the pointer head and the richer zone pooling, and
        # must keep loading exactly as they were trained.
        emit_option_tokens = "policy_head.scorer.0.weight" in state_dict
        self.adapter = StructuredObsAdapter(
            state_dict,
            card_embed_dim=int(adapter_config.get("card_embed_dim", 8)),
            attack_embed_dim=int(adapter_config.get("attack_embed_dim", 8)),
            category_embed_dim=int(adapter_config.get("category_embed_dim", 4)),
            zone_pooling=str(adapter_config.get("zone_pooling", "mean")),
            emit_option_tokens=emit_option_tokens,
        )
        self.produces_option_repr = emit_option_tokens
        backbone_config = config["backbone"]
        raw_in_keys = backbone_config.get(
            "in_keys",
            [
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
            ],
        )
        self.group_names = [
            key[-1] if isinstance(key, Sequence) and not isinstance(key, str) else key
            for key in raw_in_keys
        ]
        input_dim = state_dict["backbone.mlp.0.weight"].shape[1]
        self.mlp = _mlp(
            input_dim,
            [int(value) for value in backbone_config["num_cells"]],
            int(config["embed_dim"]),
            str(backbone_config.get("activation", "tanh")),
        )

    def forward(
        self, observation: Mapping[str, Any]
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        encoded = self.adapter(observation, self.group_names)
        if self.produces_option_repr:
            state_features, option_tokens = encoded
            return self.mlp(state_features), option_tokens
        return self.mlp(encoded)


class LinearPolicyHead(nn.Module):
    """Linear action-logit head with checkpoint-compatible names."""

    requires_option_repr = False

    def __init__(self, in_features: int, n_actions: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, n_actions)

    def forward(
        self, state_repr: torch.Tensor, option_repr: torch.Tensor | None = None
    ) -> torch.Tensor:
        del option_repr
        return self.linear(state_repr)


class PointerHead(nn.Module):
    """
    Per-option scoring head with checkpoint-compatible names.

    Mirrors :class:`src.models.heads.PointerHead`: one shared scorer over
    ``[state_repr, option_repr_i]`` for every option slot, plus a state-only
    branch for the synthetic stop action.
    """

    requires_option_repr = True

    def __init__(
        self,
        in_features: int,
        n_actions: int,
        option_dim: int,
        num_cells: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        self.n_option_slots = n_actions - 1
        self.scorer = _mlp(in_features + option_dim, num_cells, 1, activation)
        self.stop_scorer = _mlp(in_features, num_cells, 1, activation)

    def forward(
        self, state_repr: torch.Tensor, option_repr: torch.Tensor | None = None
    ) -> torch.Tensor:
        if option_repr is None:
            raise ValueError("PointerHead requires per-option tokens; got None.")
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


class ActorCritic(nn.Module):
    """Checkpoint-compatible torch-only actor-critic."""

    def __init__(
        self,
        state_dict: Mapping[str, torch.Tensor],
        config: Mapping[str, Any],
        max_options: int,
    ) -> None:
        super().__init__()
        if config["backbone"].get("_target_") != "src.models.mlp.MLPBackbone":
            raise ValueError("Kaggle runtime supports only MLPBackbone checkpoints")
        head_target = config["head"].get("_target_")
        supported_heads = {
            "src.models.heads.LinearPolicyHead",
            "src.models.heads.PointerHead",
        }
        if head_target not in supported_heads:
            raise ValueError(
                f"Kaggle runtime supports only {sorted(supported_heads)} checkpoints, "
                f"got {head_target}"
            )
        embed_dim = int(config["embed_dim"])
        self.backbone = MLPBackbone(state_dict, config)
        if head_target == "src.models.heads.PointerHead":
            head_config = config["head"]
            # The option width is read off the trained scorer rather than
            # recomputed, since the sandbox has no obs_spec to size it from.
            scorer_in = state_dict["policy_head.scorer.0.weight"].shape[1]
            self.policy_head: nn.Module = PointerHead(
                embed_dim,
                max_options + 1,
                option_dim=scorer_in - embed_dim,
                num_cells=[int(value) for value in head_config.get("num_cells", [embed_dim])],
                activation=str(head_config.get("activation", "tanh")),
            )
        else:
            self.policy_head = LinearPolicyHead(embed_dim, max_options + 1)
        self.value_head = ValueHead(
            embed_dim, [int(value) for value in config["value_head"]["num_cells"]]
        )

    def policy_logits(self, observation: Mapping[str, Any]) -> torch.Tensor:
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
