import logging

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Binary, Composite, Unbounded

from cg.api import AreaType, Observation, Option, OptionType, Pokemon, SelectData, State
from .observation_encoder import ObservationEncoder

logger = logging.getLogger(__name__)


class StructuredObservationEncoder(ObservationEncoder):
    """
    Encoder mapping an engine observation to a structured, padded TensorDict.

    This is the observation contract for model construction (see
    ``docs/torchrl_environment.md``): instead of a flat feature vector, the
    encoder emits card identities as raw integer IDs (for model-side
    embedding lookup), per-option features aligned index-for-index with the
    environment's action mask, a per-Pokemon feature table for both boards,
    and padded/masked card-ID tables for every visible zone. No modelling
    decisions (normalization, embeddings, aggregation) are made here.

    Conventions, applied everywhere:

    - Card/attack ID ``0`` means "none / padding / face-down / unknown"
      (real engine IDs start at 1).
    - Categorical integer fields store ``enum value + 1`` so ``0`` means
      "absent"; they are meant for embedding lookup, not arithmetic.
    - ``-1.0`` marks absent entries in float scalar fields where ``0`` is a
      valid value (e.g. option indices).
    - Everything is agent-relative: "my" precedes "opponent", and owner
      fields use 1 for the agent and 2 for the opponent.
    - Zones longer than their cap are truncated with a one-time warning.

    Internally, per-element writes are staged into preallocated NumPy
    buffers (reset at the start of the method that owns them) and converted
    to torch tensors with a single ``torch.from_numpy(...).clone()`` per
    output field. The clone is required because the buffers are reused
    across calls to :meth:`encode` while TorchRL keeps references to
    previously returned tensordicts (e.g. inside a rollout); without it,
    later steps would silently overwrite earlier ones.
    """

    GLOBAL_FEATURES = 41
    OPTION_CATEGORICALS = 4
    OPTION_SCALARS = 6
    POKEMON_FEATURES = 20
    ENERGY_TYPES = 12

    def __init__(
            self,
            max_options: int = 96,
            bench_cap: int = 8,
            hand_cap: int = 30,
            discard_cap: int = 60,
            prize_cap: int = 6,
            deck_cap: int = 60,
            looking_cap: int = 60,
            energy_cap: int = 16,
            evolution_cap: int = 2,
    ) -> None:
        """
        :param max_options: Padded option-space size of the paired
            environment (stop action excluded); the option table has
            ``max_options + 1`` rows so row ``i`` matches action ``i``.
        :param bench_cap: Padded bench size per player.
        :param hand_cap: Padded size of the agent's hand table.
        :param discard_cap: Padded size of each discard-pile table.
        :param prize_cap: Padded size of each prize table.
        :param deck_cap: Padded size of the deck-search table.
        :param looking_cap: Padded size of the "looking" card table.
        :param energy_cap: Padded number of attached energy cards per Pokemon.
        :param evolution_cap: Padded number of pre-evolution cards per Pokemon.
        """
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
        self._warned_zones: set[str] = set()

        n_slots = max_options + 1
        self._np_option_card_id = np.zeros(n_slots, dtype=np.int64)
        self._np_option_target_id = np.zeros(n_slots, dtype=np.int64)
        self._np_option_attack_id = np.zeros(n_slots, dtype=np.int64)
        self._np_option_owner = np.zeros(n_slots, dtype=np.int64)
        self._np_option_cats = np.zeros((n_slots, self.OPTION_CATEGORICALS), dtype=np.int64)
        self._np_option_scalars = np.full((n_slots, self.OPTION_SCALARS), -1.0, dtype=np.float32)

        self._np_pokemon_card_id = np.zeros(self._pokemon_rows, dtype=np.int64)
        self._np_pokemon_tool_id = np.zeros(self._pokemon_rows, dtype=np.int64)
        self._np_pokemon_energy_ids = np.zeros((self._pokemon_rows, energy_cap), dtype=np.int64)
        self._np_pokemon_pre_evolution_ids = np.zeros((self._pokemon_rows, evolution_cap), dtype=np.int64)
        self._np_pokemon_features = np.zeros((self._pokemon_rows, self.POKEMON_FEATURES), dtype=np.float32)
        self._np_pokemon_mask = np.zeros(self._pokemon_rows, dtype=np.bool_)

        self._np_hand_ids = np.zeros(hand_cap, dtype=np.int64)
        self._np_hand_mask = np.zeros(hand_cap, dtype=np.bool_)
        self._np_my_discard_ids = np.zeros(discard_cap, dtype=np.int64)
        self._np_my_discard_mask = np.zeros(discard_cap, dtype=np.bool_)
        self._np_my_prize_ids = np.zeros(prize_cap, dtype=np.int64)
        self._np_my_prize_mask = np.zeros(prize_cap, dtype=np.bool_)
        self._np_opp_discard_ids = np.zeros(discard_cap, dtype=np.int64)
        self._np_opp_discard_mask = np.zeros(discard_cap, dtype=np.bool_)
        self._np_opp_prize_ids = np.zeros(prize_cap, dtype=np.int64)
        self._np_opp_prize_mask = np.zeros(prize_cap, dtype=np.bool_)
        self._np_deck_ids = np.zeros(deck_cap, dtype=np.int64)
        self._np_deck_mask = np.zeros(deck_cap, dtype=np.bool_)
        self._np_looking_ids = np.zeros(looking_cap, dtype=np.int64)
        self._np_looking_mask = np.zeros(looking_cap, dtype=np.bool_)

    @property
    def pokemon_rows(self) -> int:
        """
        Number of rows in the Pokemon table (both players, active + bench).

        :return: ``2 * (1 + bench_cap)``; the agent's rows come first.
        """
        return self._pokemon_rows

    def spec(self) -> Composite:
        """
        Build the TorchRL spec of the encoded observation.

        :return: Composite spec matching the TensorDict from :meth:`encode`.
        """
        n_action_slots = self._max_options + 1
        return Composite(
            globals=Unbounded(shape=(self.GLOBAL_FEATURES,), dtype=torch.float32),
            select_cats=Unbounded(shape=(2,), dtype=torch.int64),
            context_card_ids=Unbounded(shape=(2,), dtype=torch.int64),
            stadium_id=Unbounded(shape=(1,), dtype=torch.int64),
            options=Composite(
                card_id=Unbounded(shape=(n_action_slots,), dtype=torch.int64),
                target_id=Unbounded(shape=(n_action_slots,), dtype=torch.int64),
                attack_id=Unbounded(shape=(n_action_slots,), dtype=torch.int64),
                owner=Unbounded(shape=(n_action_slots,), dtype=torch.int64),
                cats=Unbounded(shape=(n_action_slots, self.OPTION_CATEGORICALS), dtype=torch.int64),
                scalars=Unbounded(shape=(n_action_slots, self.OPTION_SCALARS), dtype=torch.float32),
            ),
            pokemon=Composite(
                card_id=Unbounded(shape=(self._pokemon_rows,), dtype=torch.int64),
                tool_id=Unbounded(shape=(self._pokemon_rows,), dtype=torch.int64),
                energy_card_ids=Unbounded(shape=(self._pokemon_rows, self._energy_cap), dtype=torch.int64),
                pre_evolution_ids=Unbounded(shape=(self._pokemon_rows, self._evolution_cap), dtype=torch.int64),
                features=Unbounded(shape=(self._pokemon_rows, self.POKEMON_FEATURES), dtype=torch.float32),
                mask=Binary(n=self._pokemon_rows, dtype=torch.bool),
            ),
            my=Composite(
                hand_ids=Unbounded(shape=(self._hand_cap,), dtype=torch.int64),
                hand_mask=Binary(n=self._hand_cap, dtype=torch.bool),
                discard_ids=Unbounded(shape=(self._discard_cap,), dtype=torch.int64),
                discard_mask=Binary(n=self._discard_cap, dtype=torch.bool),
                prize_ids=Unbounded(shape=(self._prize_cap,), dtype=torch.int64),
                prize_mask=Binary(n=self._prize_cap, dtype=torch.bool),
            ),
            opp=Composite(
                discard_ids=Unbounded(shape=(self._discard_cap,), dtype=torch.int64),
                discard_mask=Binary(n=self._discard_cap, dtype=torch.bool),
                prize_ids=Unbounded(shape=(self._prize_cap,), dtype=torch.int64),
                prize_mask=Binary(n=self._prize_cap, dtype=torch.bool),
            ),
            select_deck=Composite(
                ids=Unbounded(shape=(self._deck_cap,), dtype=torch.int64),
                mask=Binary(n=self._deck_cap, dtype=torch.bool),
            ),
            looking=Composite(
                ids=Unbounded(shape=(self._looking_cap,), dtype=torch.int64),
                mask=Binary(n=self._looking_cap, dtype=torch.bool),
            ),
        )

    def encode(self, observation: Observation, agent_seat: int, chosen_count: int) -> TensorDict:
        """
        Encode an observation from the agent's perspective.

        :param observation: Current engine observation (``current`` must be set).
        :param agent_seat: Player index (0 or 1) of the agent.
        :param chosen_count: Number of options already picked in an ongoing
            multi-select accumulation.
        :return: TensorDict matching :meth:`spec`, batch size ``()``.
        """
        state = observation.current
        select = observation.select
        return TensorDict(
            {
                "globals": self._encode_globals(state, select, agent_seat, chosen_count),
                "select_cats": self._encode_select_cats(select),
                "context_card_ids": self._encode_context_cards(select),
                "stadium_id": torch.tensor(
                    [state.stadium[0].id if len(state.stadium) > 0 else 0], dtype=torch.int64
                ),
                "options": self._encode_options(state, select, agent_seat),
                "pokemon": self._encode_pokemon(state, agent_seat),
                "my": self._encode_my_zones(state, agent_seat),
                "opp": self._encode_opp_zones(state, agent_seat),
                "select_deck": self._encode_id_list(
                    [card.id for card in select.deck] if select is not None and select.deck is not None else [],
                    self._np_deck_ids,
                    self._np_deck_mask,
                    "select_deck",
                ),
                "looking": self._encode_id_list(
                    [card.id if card is not None else 0 for card in state.looking]
                    if state.looking is not None
                    else [],
                    self._np_looking_ids,
                    self._np_looking_mask,
                    "looking",
                ),
            },
            batch_size=torch.Size(()),
        )

    def _encode_globals(
            self,
            state: State,
            select: SelectData | None,
            agent_seat: int,
            chosen_count: int,
    ) -> torch.Tensor:
        """
        Encode scalar game and selection context as raw float values.

        Layout: 8 turn/flag entries, 9 selection entries, then 12 entries per
        player (agent first); see ``docs/torchrl_environment.md`` for the exact
        index table. Values are unnormalized; scaling is a model-side choice.

        :param state: Current engine state.
        :param select: Current selection, or None on a terminal observation.
        :param agent_seat: Player index of the agent.
        :param chosen_count: Picks already accumulated in a multi-select.
        :return: Float32 tensor of shape ``(GLOBAL_FEATURES,)``.
        """
        if state.firstPlayer == -1:
            first_player_flag = -1.0
        else:
            first_player_flag = 1.0 if state.firstPlayer == agent_seat else 0.0
        features: list[float] = [
            float(state.turn),
            float(state.turnActionCount),
            float(agent_seat),
            first_player_flag,
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
                float(chosen_count),
                float(select.remainDamageCounter),
                float(select.remainEnergyCost),
                1.0 if select.deck is not None else 0.0,
                1.0 if state.looking is not None else 0.0,
            ]
        for player in (state.players[agent_seat], state.players[1 - agent_seat]):
            active = player.active[0] if len(player.active) > 0 else None
            features += [
                float(player.deckCount),
                float(player.handCount),
                float(len(player.prize)),
                float(len(player.bench)),
                float(player.benchMax),
                1.0 if len(player.active) > 0 else 0.0,
                1.0 if len(player.active) > 0 and active is None else 0.0,
                float(player.poisoned),
                float(player.burned),
                float(player.asleep),
                float(player.paralyzed),
                float(player.confused),
            ]
        return torch.tensor(features, dtype=torch.float32)

    def _encode_select_cats(self, select: SelectData | None) -> torch.Tensor:
        """
        Encode the selection type and context as shifted categorical IDs.

        :param select: Current selection, or None on a terminal observation.
        :return: Int64 tensor ``[type + 1, context + 1]`` (0 when absent).
        """
        if select is None:
            return torch.zeros(2, dtype=torch.int64)
        return torch.tensor([int(select.type) + 1, int(select.context) + 1], dtype=torch.int64)

    def _encode_context_cards(self, select: SelectData | None) -> torch.Tensor:
        """
        Encode the selection's context and effect card identities.

        :param select: Current selection, or None on a terminal observation.
        :return: Int64 tensor ``[contextCard.id, effect.id]`` (0 when absent).
        """
        if select is None:
            return torch.zeros(2, dtype=torch.int64)
        context_id = select.contextCard.id if select.contextCard is not None else 0
        effect_id = select.effect.id if select.effect is not None else 0
        return torch.tensor([context_id, effect_id], dtype=torch.int64)

    def _encode_options(self, state: State, select: SelectData | None, agent_seat: int) -> TensorDict:
        """
        Encode the option list into per-slot tensors aligned with the action mask.

        Row ``i`` describes action ``i``; rows beyond the option count
        (including the stop slot) stay zero. Card references
        (area/index/playerIndex triples) are resolved to concrete card IDs
        against the current state so the model never has to dereference zones.

        Per-element writes are staged into preallocated NumPy buffers
        (reset to their default value at the start of this call) and
        converted to torch tensors with one copy each at the end.

        :param state: Current engine state.
        :param select: Current selection, or None on a terminal observation.
        :param agent_seat: Player index of the agent.
        :return: TensorDict with option tables of ``max_options + 1`` rows.
        """
        card_id = self._np_option_card_id
        target_id = self._np_option_target_id
        attack_id = self._np_option_attack_id
        owner = self._np_option_owner
        cats = self._np_option_cats
        scalars = self._np_option_scalars
        card_id.fill(0)
        target_id.fill(0)
        attack_id.fill(0)
        owner.fill(0)
        cats.fill(0)
        scalars.fill(-1.0)
        options = select.option if select is not None else []
        for slot, option in enumerate(options[: self._max_options]):
            cats[slot, 0] = int(option.type) + 1
            cats[slot, 1] = int(option.area) + 1 if option.area is not None else 0
            cats[slot, 2] = int(option.inPlayArea) + 1 if option.inPlayArea is not None else 0
            cats[slot, 3] = (
                int(option.specialConditionType) + 1 if option.specialConditionType is not None else 0
            )
            if option.playerIndex is not None:
                owner[slot] = 1 if option.playerIndex == agent_seat else 2
            scalars[slot, 0] = float(option.number) if option.number is not None else -1.0
            scalars[slot, 1] = float(option.count) if option.count is not None else -1.0
            scalars[slot, 2] = float(option.index) if option.index is not None else -1.0
            scalars[slot, 3] = float(option.toolIndex) if option.toolIndex is not None else -1.0
            scalars[slot, 4] = float(option.energyIndex) if option.energyIndex is not None else -1.0
            scalars[slot, 5] = float(option.inPlayIndex) if option.inPlayIndex is not None else -1.0
            card_id[slot], target_id[slot], attack_id[slot] = self._resolve_option_ids(
                state, select, option, agent_seat
            )
        return TensorDict(
            {
                "card_id": torch.from_numpy(card_id).clone(),
                "target_id": torch.from_numpy(target_id).clone(),
                "attack_id": torch.from_numpy(attack_id).clone(),
                "owner": torch.from_numpy(owner).clone(),
                "cats": torch.from_numpy(cats).clone(),
                "scalars": torch.from_numpy(scalars).clone(),
            },
            batch_size=torch.Size(()),
        )

    def _resolve_option_ids(
            self,
            state: State,
            select: SelectData,
            option: Option,
            agent_seat: int,
    ) -> tuple[int, int, int]:
        """
        Resolve an option's zone references to concrete card and attack IDs.

        :param state: Current engine state.
        :param select: Current selection (for deck-search lookups).
        :param option: Option to resolve.
        :param agent_seat: Player index of the agent, used as the owner for
            option types whose references omit ``playerIndex`` (own zones).
        :return: Tuple ``(card_id, target_id, attack_id)``; 0 marks
            none/face-down/unknown. ``target_id`` is the in-play Pokemon a
            card-directed option acts on (attach/evolve target, or the
            carrier of a selected tool/energy).
        """
        option_type = option.type
        owner_index = option.playerIndex if option.playerIndex is not None else agent_seat
        if option_type == OptionType.CARD:
            return self._card_id_at(state, select, owner_index, option.area, option.index), 0, 0
        if option_type in (OptionType.TOOL_CARD, OptionType.ENERGY_CARD, OptionType.ENERGY):
            pokemon = self._pokemon_at(state, owner_index, option.area, option.index)
            if pokemon is None:
                return 0, 0, 0
            if option_type == OptionType.TOOL_CARD:
                attached = self._card_in_list(pokemon.tools, option.toolIndex)
            else:
                attached = self._card_in_list(pokemon.energyCards, option.energyIndex)
            return attached, pokemon.id, 0
        if option_type in (OptionType.PLAY, OptionType.ABILITY, OptionType.DISCARD):
            return self._card_id_at(state, select, owner_index, option.area or AreaType.HAND, option.index), 0, 0
        if option_type in (OptionType.ATTACH, OptionType.EVOLVE):
            played = self._card_id_at(state, select, owner_index, option.area, option.index)
            target = self._pokemon_at(state, owner_index, option.inPlayArea, option.inPlayIndex)
            return played, target.id if target is not None else 0, 0
        if option_type == OptionType.ATTACK:
            return 0, 0, option.attackId if option.attackId is not None else 0
        if option_type == OptionType.SKILL:
            return option.cardId if option.cardId is not None else 0, 0, 0
        return 0, 0, 0

    def _card_id_at(
            self,
            state: State,
            select: SelectData,
            player_index: int,
            area: AreaType | None,
            index: int | None,
    ) -> int:
        """
        Look up the card ID at a (player, area, index) reference.

        :param state: Current engine state.
        :param select: Current selection (source of the deck-search list).
        :param player_index: Absolute owner index of the referenced zone.
        :param area: Referenced area, or None.
        :param index: Index within the area, or None.
        :return: Card ID, or 0 when the reference is absent, out of range,
            face-down, or in a zone the agent cannot see.
        """
        if area is None or index is None:
            return 0
        player = state.players[player_index]
        if area == AreaType.HAND:
            if player.hand is None:
                return 0
            return self._card_in_list(player.hand, index)
        if area == AreaType.DISCARD:
            return self._card_in_list(player.discard, index)
        if area == AreaType.ACTIVE:
            pokemon = self._pokemon_at(state, player_index, area, index)
            return pokemon.id if pokemon is not None else 0
        if area == AreaType.BENCH:
            pokemon = self._pokemon_at(state, player_index, area, index)
            return pokemon.id if pokemon is not None else 0
        if area == AreaType.PRIZE:
            if 0 <= index < len(player.prize) and player.prize[index] is not None:
                return player.prize[index].id
            return 0
        if area == AreaType.STADIUM:
            return self._card_in_list(state.stadium, index)
        if area == AreaType.DECK:
            if select is not None and select.deck is not None:
                return self._card_in_list(select.deck, index)
            return 0
        if area == AreaType.LOOKING:
            if state.looking is not None and 0 <= index < len(state.looking) and state.looking[index] is not None:
                return state.looking[index].id
            return 0
        return 0

    def _pokemon_at(
            self,
            state: State,
            player_index: int,
            area: AreaType | None,
            index: int | None,
    ) -> Pokemon | None:
        """
        Look up an in-play Pokemon at a (player, area, index) reference.

        :param state: Current engine state.
        :param player_index: Absolute owner index of the Pokemon.
        :param area: ``ACTIVE`` or ``BENCH``; anything else resolves to None.
        :param index: Index within the area, or None.
        :return: The Pokemon, or None when absent or face-down.
        """
        if area is None or index is None:
            return None
        player = state.players[player_index]
        if area == AreaType.ACTIVE:
            slots = player.active
        elif area == AreaType.BENCH:
            slots = player.bench
        else:
            return None
        if 0 <= index < len(slots):
            return slots[index]
        return None

    @staticmethod
    def _card_in_list(cards: list, index: int | None) -> int:
        """
        Read a card ID from a card list, tolerating bad indices.

        :param cards: List of ``Card`` objects (or None entries).
        :param index: Index to read, or None.
        :return: The card's ID, or 0 when out of range or face-down.
        """
        if index is None or not 0 <= index < len(cards):
            return 0
        card = cards[index]
        return card.id if card is not None else 0

    def _encode_pokemon(self, state: State, agent_seat: int) -> TensorDict:
        """
        Encode both boards into a fixed-size per-Pokemon table.

        Row layout: agent active, agent bench (``bench_cap`` rows), opponent
        active, opponent bench. The mask marks occupied slots; a face-down
        active is masked True with ``card_id`` 0.

        Per-element writes are staged into preallocated NumPy buffers
        (reset to zero/False at the start of this call) and converted to
        torch tensors with one copy each at the end.

        :param state: Current engine state.
        :param agent_seat: Player index of the agent.
        :return: TensorDict with the Pokemon tables and mask.
        """
        card_id = self._np_pokemon_card_id
        tool_id = self._np_pokemon_tool_id
        energy_card_ids = self._np_pokemon_energy_ids
        pre_evolution_ids = self._np_pokemon_pre_evolution_ids
        features = self._np_pokemon_features
        mask = self._np_pokemon_mask
        card_id.fill(0)
        tool_id.fill(0)
        energy_card_ids.fill(0)
        pre_evolution_ids.fill(0)
        features.fill(0.0)
        mask.fill(False)
        for side, player_index in enumerate((agent_seat, 1 - agent_seat)):
            player = state.players[player_index]
            base_row = side * (1 + self._bench_cap)
            if len(player.active) > 0:
                mask[base_row] = True
                self._fill_pokemon_row(player.active[0], base_row, True)
            bench = player.bench
            if len(bench) > self._bench_cap:
                self._warn_truncation("bench", len(bench), self._bench_cap)
                bench = bench[: self._bench_cap]
            for offset, pokemon in enumerate(bench):
                row = base_row + 1 + offset
                mask[row] = True
                self._fill_pokemon_row(pokemon, row, False)
        return TensorDict(
            {
                "card_id": torch.from_numpy(card_id).clone(),
                "tool_id": torch.from_numpy(tool_id).clone(),
                "energy_card_ids": torch.from_numpy(energy_card_ids).clone(),
                "pre_evolution_ids": torch.from_numpy(pre_evolution_ids).clone(),
                "features": torch.from_numpy(features).clone(),
                "mask": torch.from_numpy(mask).clone(),
            },
            batch_size=torch.Size(()),
        )

    def _fill_pokemon_row(self, pokemon: Pokemon | None, row: int, is_active: bool) -> None:
        """
        Write one Pokemon into the given row of the staged board buffers.

        :param pokemon: Pokemon to encode; None (face-down active) leaves the
            row zeroed apart from the is_active flag.
        :param row: Destination row index.
        :param is_active: Whether this row is an active slot.
        """
        features = self._np_pokemon_features
        features[row, 4] = 1.0 if is_active else 0.0
        if pokemon is None:
            return
        card_id = self._np_pokemon_card_id
        tool_id = self._np_pokemon_tool_id
        energy_card_ids = self._np_pokemon_energy_ids
        pre_evolution_ids = self._np_pokemon_pre_evolution_ids
        card_id[row] = pokemon.id
        if len(pokemon.tools) > 0:
            tool_id[row] = pokemon.tools[0].id
        energy_cards = pokemon.energyCards
        if len(energy_cards) > self._energy_cap:
            self._warn_truncation("energy_cards", len(energy_cards), self._energy_cap)
            energy_cards = energy_cards[: self._energy_cap]
        for column, card in enumerate(energy_cards):
            energy_card_ids[row, column] = card.id
        pre_evolutions = pokemon.preEvolution
        if len(pre_evolutions) > self._evolution_cap:
            self._warn_truncation("pre_evolution", len(pre_evolutions), self._evolution_cap)
            pre_evolutions = pre_evolutions[: self._evolution_cap]
        for column, card in enumerate(pre_evolutions):
            pre_evolution_ids[row, column] = card.id
        features[row, 0] = float(pokemon.hp)
        features[row, 1] = float(pokemon.maxHp)
        features[row, 2] = pokemon.hp / pokemon.maxHp if pokemon.maxHp > 0 else 0.0
        features[row, 3] = float(pokemon.appearThisTurn)
        features[row, 5] = float(len(pokemon.tools))
        features[row, 6] = float(len(pokemon.energyCards))
        features[row, 7] = float(len(pokemon.preEvolution))
        for energy in pokemon.energies:
            energy_index = int(energy)
            if 0 <= energy_index < self.ENERGY_TYPES:
                features[row, 8 + energy_index] += 1.0
        return

    def _encode_my_zones(self, state: State, agent_seat: int) -> TensorDict:
        """
        Encode the agent's hand, discard pile and prizes.

        :param state: Current engine state.
        :param agent_seat: Player index of the agent.
        :return: TensorDict with padded ID tables and masks.
        """
        player = state.players[agent_seat]
        hand_ids = [card.id for card in player.hand] if player.hand is not None else []
        hand_ids_tensor, hand_mask_tensor = self._stage_id_list(
            hand_ids, self._np_hand_ids, self._np_hand_mask, "hand"
        )
        discard_ids_tensor, discard_mask_tensor = self._stage_id_list(
            [card.id for card in player.discard], self._np_my_discard_ids, self._np_my_discard_mask, "discard"
        )
        prize_ids_tensor, prize_mask_tensor = self._stage_id_list(
            [card.id if card is not None else 0 for card in player.prize],
            self._np_my_prize_ids,
            self._np_my_prize_mask,
            "prize",
        )
        return TensorDict(
            {
                "hand_ids": hand_ids_tensor,
                "hand_mask": hand_mask_tensor,
                "discard_ids": discard_ids_tensor,
                "discard_mask": discard_mask_tensor,
                "prize_ids": prize_ids_tensor,
                "prize_mask": prize_mask_tensor,
            },
            batch_size=torch.Size(()),
        )

    def _encode_opp_zones(self, state: State, agent_seat: int) -> TensorDict:
        """
        Encode the opponent's public zones (discard pile and prizes).

        :param state: Current engine state.
        :param agent_seat: Player index of the agent.
        :return: TensorDict with padded ID tables and masks.
        """
        player = state.players[1 - agent_seat]
        discard_ids_tensor, discard_mask_tensor = self._stage_id_list(
            [card.id for card in player.discard], self._np_opp_discard_ids, self._np_opp_discard_mask, "discard"
        )
        prize_ids_tensor, prize_mask_tensor = self._stage_id_list(
            [card.id if card is not None else 0 for card in player.prize],
            self._np_opp_prize_ids,
            self._np_opp_prize_mask,
            "prize",
        )
        return TensorDict(
            {
                "discard_ids": discard_ids_tensor,
                "discard_mask": discard_mask_tensor,
                "prize_ids": prize_ids_tensor,
                "prize_mask": prize_mask_tensor,
            },
            batch_size=torch.Size(()),
        )

    def _stage_id_list(
            self,
            card_ids: list[int],
            ids_buffer: np.ndarray,
            mask_buffer: np.ndarray,
            zone_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pad a list of card IDs into a preallocated buffer pair and copy out.

        :param card_ids: Card IDs to encode; 0 entries mark face-down cards
            that occupy a slot (mask True) without a known identity.
        :param ids_buffer: Preallocated int64 buffer of shape ``(cap,)``,
            reused across calls and reset here.
        :param mask_buffer: Preallocated bool buffer of shape ``(cap,)``,
            reused across calls and reset here.
        :param zone_name: Zone label used in the one-time truncation warning.
        :return: Tuple of owned ``(ids, mask)`` torch tensors.
        """
        cap = ids_buffer.shape[0]
        if len(card_ids) > cap:
            self._warn_truncation(zone_name, len(card_ids), cap)
            card_ids = card_ids[:cap]
        ids_buffer.fill(0)
        mask_buffer.fill(False)
        if len(card_ids) > 0:
            ids_buffer[: len(card_ids)] = card_ids
            mask_buffer[: len(card_ids)] = True
        return torch.from_numpy(ids_buffer).clone(), torch.from_numpy(mask_buffer).clone()

    def _encode_id_list(
            self,
            card_ids: list[int],
            ids_buffer: np.ndarray,
            mask_buffer: np.ndarray,
            zone_name: str,
    ) -> TensorDict:
        """
        Pad a list of card IDs into a fixed-size table with a validity mask.

        :param card_ids: Card IDs to encode; 0 entries mark face-down cards
            that occupy a slot (mask True) without a known identity.
        :param ids_buffer: Preallocated int64 buffer of shape ``(cap,)``,
            reused across calls and reset here.
        :param mask_buffer: Preallocated bool buffer of shape ``(cap,)``,
            reused across calls and reset here.
        :param zone_name: Zone label used in the one-time truncation warning.
        :return: TensorDict with ``ids`` (int64) and ``mask`` (bool) entries.
        """
        ids_tensor, mask_tensor = self._stage_id_list(card_ids, ids_buffer, mask_buffer, zone_name)
        return TensorDict({"ids": ids_tensor, "mask": mask_tensor}, batch_size=torch.Size(()))

    def _warn_truncation(self, zone_name: str, length: int, cap: int) -> None:
        """
        Warn once per zone when its content exceeds the padded capacity.

        :param zone_name: Zone label for the log message.
        :param length: Actual number of entries.
        :param cap: Padded table size.
        """
        if zone_name not in self._warned_zones:
            logger.warning("Zone %s has %d entries, truncating to cap %d.", zone_name, length, cap)
            self._warned_zones.add(zone_name)
