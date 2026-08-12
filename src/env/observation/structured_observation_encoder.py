import logging
from enum import IntEnum

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Binary, Composite, Unbounded

from cg.api import (
    Observation,
    PlayerState,
    Pokemon,
    SelectData,
    State,
)

from .observation_encoder import ObservationEncoder
from .option_reference_resolver import OptionReferenceResolver

logger = logging.getLogger(__name__)


class StructuredObservationEncoder(ObservationEncoder):
    """
    Encoder mapping an engine observation to a structured, padded TensorDict.

    Card identities stay raw integer IDs for model-side embedding lookup,
    per-option features are aligned index-for-index with the action mask, and
    every visible zone becomes a padded ID table with a companion ``*_mask``.
    No modelling decision (normalization, embeddings, aggregation) is made
    here. The full schema and its conventions are in
    ``docs/environment/torchrl_environment.md``.

    Each zone is padded to a fixed ``*_cap``, so every observation has the same
    shape. Most caps trace to engine constants rather than to taste; see the
    ``__init__`` docstring. Zones over their cap truncate with a one-time
    warning, except the option list, which raises: it must stay in sync with
    the action mask. An out-of-range option reference raises too, rather than
    mapping to "unknown" and contaminating training data.

    Writes are staged into preallocated NumPy buffers and copied out per field
    with ``torch.from_numpy(...).clone()``. The clone is required because
    TorchRL keeps references to past tensordicts, e.g. in a rollout. An encoder
    instance is therefore not thread-safe: one per environment, called
    serially.
    """

    GLOBAL_FEATURE_COUNT = 41
    OPTION_CATEGORICAL_COUNT = 4
    OPTION_SCALAR_COUNT = 6
    #: Live state of the in-play Pokemon an option acts on, written onto the
    #: option's own row: ``[resolved, hp, maxHp, hp fraction, energy count,
    #: tool count, is-active]``. All zero when the option targets no Pokemon,
    #: with column 0 separating "no target" from "a target whose values are
    #: zero". ``target_id`` names the card but is shared by every copy of it,
    #: so without this block two "attach energy" options over two copies of one
    #: Pokemon differ only in a raw index scalar: rankable by no feedforward
    #: network, which cannot index the ``pokemon`` table with it.
    OPTION_TARGET_FEATURE_COUNT = 7
    POKEMON_FEATURE_COUNT = 20
    ENERGY_TYPE_COUNT = 12
    # Eight game fields followed by selection-present/min/max/option-count.
    ALREADY_CHOSEN_OPTION_COUNT_INDEX = 12

    def __init__(
        self,
        max_options: int = 96,
        bench_cap: int = 8,
        hand_cap: int = 60,
        discard_cap: int = 60,
        prize_cap: int = 6,
        deck_cap: int = 60,
        looking_cap: int = 60,
        energy_cap: int = 60,
        evolution_cap: int = 2,
        tool_cap: int = 2,
    ) -> None:
        """
        Most defaults trace to hard constants in the C++ engine
        (``ptcg_engine/.../Core.h``: ``DECK_SIZE=60``, ``PRIZE_SIZE=6``,
        ``BENCH_SIZE_MAX=8``) rather than being arbitrary; the rest are
        generous empirical headroom above what's observed in practice.

        :param max_options: Padded option-space size of the paired
            environment (stop action excluded); the option table has
            ``max_options + 1`` rows so row ``i`` matches action ``i``.
            No engine constant; set above the empirically observed max
            option count (a full deck search can offer ~60 options; 42
            was the largest seen under random play).
        :param bench_cap: Padded bench size per player. Matches the
            engine's hard ``BENCH_SIZE_MAX``; the default in-game bench
            is 5, but some card effects raise capacity up to this ceiling.
        :param hand_cap: Padded size of the agent's hand table. No engine
            limit on hand size, but a player owns exactly ``DECK_SIZE``
            cards for the whole game, so 60 is the true ceiling and this
            can no longer truncate. Was 30 as empirical headroom until
            self-play produced 31-card hands.
        :param discard_cap: Padded size of each discard-pile table.
            Matches ``DECK_SIZE``: a discard pile can never exceed a full
            deck's worth of cards.
        :param prize_cap: Padded size of each prize table. Always exactly
            6 by rule (``PRIZE_SIZE``); this is a fixed constant, not
            really a truncation cap.
        :param deck_cap: Padded size of the deck-search table. Matches
            ``DECK_SIZE``: the largest a full-deck search can reveal.
        :param looking_cap: Padded size of the "looking" card table.
            Matches ``DECK_SIZE``, for the same reason as ``deck_cap``.
        :param energy_cap: Padded number of attached energy cards per
            Pokemon. No engine limit, and this has been raised three times
            (16 -> 24 -> 40 -> 60) after self-play runs exceeded it. 60 is
            the last raise worth making: a player owns ``DECK_SIZE`` cards
            in total, so no single Pokemon can carry more. Note that
            overflow only drops the surplus cards' identities: the
            attachment count and the per-type histogram in ``features``
            are both computed from the untruncated list, and the model
            pools the identities into a masked mean, so widening this
            costs observation bytes but changes no model dimension.
        :param evolution_cap: Padded number of pre-evolution cards per
            Pokemon. Matches the fixed evolution chain depth: Basic ->
            Stage 1 -> Stage 2 is at most 2 pre-evolutions.
        :param tool_cap: Padded number of attached tool cards per Pokemon.
            One tool is the normal rule, but card effects can stack a
            second, which self-play reached at 14M frames. Overflow costs
            only the surplus identities, as with ``energy_cap``: the
            attachment count in ``features`` comes from the untruncated
            list and the model pools the identities, so this changes no
            model dimension.
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
        self._tool_cap = tool_cap
        self._pokemon_rows = 2 * (1 + bench_cap)
        self._warned_zones: set[str] = set()

        # This is allocated storage which can then be copied over in bulk to a tensor (e.g. GPU)
        n_slots = max_options + 1
        self._np_option_card_id = np.zeros(n_slots, dtype=np.int64)
        self._np_option_target_id = np.zeros(n_slots, dtype=np.int64)
        self._np_option_attack_id = np.zeros(n_slots, dtype=np.int64)
        self._np_option_owner = np.zeros(n_slots, dtype=np.int64)
        self._np_option_cats = np.zeros(
            (n_slots, self.OPTION_CATEGORICAL_COUNT), dtype=np.int64
        )
        self._np_option_scalars = np.full(
            (n_slots, self.OPTION_SCALAR_COUNT), -1.0, dtype=np.float32
        )
        self._np_option_target = np.zeros(
            (n_slots, self.OPTION_TARGET_FEATURE_COUNT), dtype=np.float32
        )

        self._np_pokemon_card_id = np.zeros(self._pokemon_rows, dtype=np.int64)
        self._np_pokemon_tool_id = np.zeros(
            (self._pokemon_rows, tool_cap), dtype=np.int64
        )
        self._np_pokemon_energy_ids = np.zeros(
            (self._pokemon_rows, energy_cap), dtype=np.int64
        )
        self._np_pokemon_pre_evolution_ids = np.zeros(
            (self._pokemon_rows, evolution_cap), dtype=np.int64
        )
        self._np_pokemon_features = np.zeros(
            (self._pokemon_rows, self.POKEMON_FEATURE_COUNT), dtype=np.float32
        )
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
            globals=Unbounded(shape=(self.GLOBAL_FEATURE_COUNT,), dtype=torch.float32),
            select_cats=Unbounded(shape=(2,), dtype=torch.int64),
            context_card_ids=Unbounded(shape=(2,), dtype=torch.int64),
            stadium_id=Unbounded(shape=(1,), dtype=torch.int64),
            options=Composite(
                card_id=Unbounded(shape=(n_action_slots,), dtype=torch.int64),
                target_id=Unbounded(shape=(n_action_slots,), dtype=torch.int64),
                attack_id=Unbounded(shape=(n_action_slots,), dtype=torch.int64),
                owner=Unbounded(shape=(n_action_slots,), dtype=torch.int64),
                cats=Unbounded(
                    shape=(n_action_slots, self.OPTION_CATEGORICAL_COUNT),
                    dtype=torch.int64,
                ),
                scalars=Unbounded(
                    shape=(n_action_slots, self.OPTION_SCALAR_COUNT),
                    dtype=torch.float32,
                ),
                target_state=Unbounded(
                    shape=(n_action_slots, self.OPTION_TARGET_FEATURE_COUNT),
                    dtype=torch.float32,
                ),
            ),
            pokemon=Composite(
                card_id=Unbounded(shape=(self._pokemon_rows,), dtype=torch.int64),
                tool_id=Unbounded(
                    shape=(self._pokemon_rows, self._tool_cap), dtype=torch.int64
                ),
                energy_card_ids=Unbounded(
                    shape=(self._pokemon_rows, self._energy_cap), dtype=torch.int64
                ),
                pre_evolution_ids=Unbounded(
                    shape=(self._pokemon_rows, self._evolution_cap), dtype=torch.int64
                ),
                features=Unbounded(
                    shape=(self._pokemon_rows, self.POKEMON_FEATURE_COUNT),
                    dtype=torch.float32,
                ),
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

    def encode(
        self,
        observation: Observation,
        agent_seat: int,
        already_chosen_option_count: int,
    ) -> TensorDict:
        """
        Encode an observation from the agent's perspective.

        :param observation: Current engine observation (``current`` must be set).
        :param agent_seat: Player index (0 or 1) of the agent.
        :param already_chosen_option_count: Number of options already picked in
            an ongoing multi-select accumulation.
        :return: TensorDict matching :meth:`spec`, batch size ``()``.
        """
        state = observation.current
        assert state is not None, "observation.current must be set"
        select = observation.select
        return TensorDict(
            {
                "globals": self._encode_globals(
                    state, select, agent_seat, already_chosen_option_count
                ),
                "select_cats": self._encode_select_cats(select),
                "context_card_ids": self._encode_context_cards(select),
                "stadium_id": torch.tensor(
                    [state.stadium[0].id if len(state.stadium) > 0 else 0],
                    dtype=torch.int64,
                ),
                "options": self._encode_options(state, select, agent_seat),
                "pokemon": self._encode_pokemon(state, agent_seat),
                "my": self._encode_my_zones(state, agent_seat),
                "opp": self._encode_opp_zones(state, agent_seat),
                "select_deck": self._encode_id_list(
                    [card.id for card in select.deck]
                    if select is not None and select.deck is not None
                    else [],
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

    def update_already_chosen_option_count(
        self,
        encoded: TensorDict,
        already_chosen_option_count: int,
    ) -> bool:
        """Update the sole count-dependent field in a cached encoding."""
        encoded["globals"][self.ALREADY_CHOSEN_OPTION_COUNT_INDEX] = float(
            already_chosen_option_count
        )
        return True

    @staticmethod
    def _encode_globals(
        state: State,
        select: SelectData | None,
        agent_seat: int,
        already_chosen_option_count: int,
    ) -> torch.Tensor:
        """
        Encode scalar game and selection context as raw float values.

        Layout: 8 turn/flag entries, 9 selection entries, then 12 entries per
        player (agent first); see ``docs/torchrl_environment.md`` for the exact
        index table. Values are unnormalized; scaling is a model-side choice.

        :param state: Current engine state.
        :param select: Current selection, or None on a terminal observation.
        :param agent_seat: Player index of the agent.
        :param already_chosen_option_count: Picks already accumulated in a
            multi-select.
        :return: Float32 tensor of shape ``(GLOBAL_FEATURE_COUNT,)``.
        """
        if state.firstPlayer == -1:
            first_player_flag = -1.0
        else:
            first_player_flag = 1.0 if state.firstPlayer == agent_seat else 0.0
        # Game block (8 entries): turn progress and the once-per-turn flags.
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
        # Selection block (9 entries): all zero on a terminal observation,
        # where entry 0 doubles as a "selection present" flag.
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
                1.0 if select.deck is not None else 0.0,
                1.0 if state.looking is not None else 0.0,
            ]
        # Player blocks (12 entries each, agent first): zone counts, the
        # active slot occupancy/face-down flags and special conditions.
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
        if len(features) != StructuredObservationEncoder.GLOBAL_FEATURE_COUNT:
            raise RuntimeError(
                f"Expected {StructuredObservationEncoder.GLOBAL_FEATURE_COUNT} global features, "
                f"got {len(features)}."
            )
        return torch.tensor(features, dtype=torch.float32)

    @staticmethod
    def _encode_select_cats(select: SelectData | None) -> torch.Tensor:
        """
        Encode the selection type and context as shifted categorical IDs.

        :param select: Current selection, or None on a terminal observation.
        :return: Int64 tensor ``[type + 1, context + 1]`` (0 when absent).
        """
        if select is None:
            return torch.zeros(2, dtype=torch.int64)
        return torch.tensor(
            [int(select.type) + 1, int(select.context) + 1], dtype=torch.int64
        )

    @staticmethod
    def _encode_context_cards(select: SelectData | None) -> torch.Tensor:
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

    @staticmethod
    def _shifted_category(value: IntEnum | None) -> int:
        """
        Shift an optional categorical enum so that 0 can mean "absent".

        :param value: Enum value, or None when the field is absent.
        :return: ``int(value) + 1``, or 0 when absent.
        """
        return int(value) + 1 if value is not None else 0

    @staticmethod
    def _float_or_absent(value: int | None) -> float:
        """
        Convert an optional integer field to its float feature value.

        :param value: Field value, or None when absent.
        :return: ``float(value)``, or -1.0 when absent (0 is a valid value).
        """
        return float(value) if value is not None else -1.0

    def _encode_options(
        self, state: State, select: SelectData | None, agent_seat: int
    ) -> TensorDict:
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
        target_state = self._np_option_target
        card_id.fill(0)
        target_id.fill(0)
        attack_id.fill(0)
        owner.fill(0)
        cats.fill(0)
        scalars.fill(-1.0)
        target_state.fill(0.0)
        options = select.option if select is not None else []
        if len(options) > self._max_options:
            raise ValueError(
                f"Selection offers {len(options)} options but max_options is {self._max_options}; "
                f"a truncated option would desynchronize the observation from the action space."
            )
        for slot, option in enumerate(options):
            cats[slot, 0] = int(option.type) + 1
            cats[slot, 1] = self._shifted_category(option.area)
            cats[slot, 2] = self._shifted_category(option.inPlayArea)
            cats[slot, 3] = self._shifted_category(option.specialConditionType)
            if option.playerIndex is not None:
                owner[slot] = 1 if option.playerIndex == agent_seat else 2
            scalars[slot, 0] = self._float_or_absent(option.number)
            scalars[slot, 1] = self._float_or_absent(option.count)
            scalars[slot, 2] = self._float_or_absent(option.index)
            scalars[slot, 3] = self._float_or_absent(option.toolIndex)
            scalars[slot, 4] = self._float_or_absent(option.energyIndex)
            scalars[slot, 5] = self._float_or_absent(option.inPlayIndex)
            card_id[slot], target_id[slot], attack_id[slot] = (
                OptionReferenceResolver.resolve(state, select, option, agent_seat)
            )
            # The targeted Pokemon's *live* state, which target_id cannot carry:
            # it is a card ID, identical across every copy of that card.
            target = OptionReferenceResolver.resolve_target_pokemon(
                state, option, agent_seat
            )
            if target is not None:
                is_active = any(
                    entry is target
                    for entry in state.players[
                        option.playerIndex
                        if option.playerIndex is not None
                        else agent_seat
                    ].active
                )
                target_state[slot, 0] = 1.0
                target_state[slot, 1] = float(target.hp)
                target_state[slot, 2] = float(target.maxHp)
                target_state[slot, 3] = (
                    target.hp / target.maxHp if target.maxHp > 0 else 0.0
                )
                target_state[slot, 4] = float(len(target.energyCards))
                target_state[slot, 5] = float(len(target.tools))
                target_state[slot, 6] = 1.0 if is_active else 0.0
        return TensorDict(
            {
                "card_id": torch.from_numpy(card_id).clone(),
                "target_id": torch.from_numpy(target_id).clone(),
                "attack_id": torch.from_numpy(attack_id).clone(),
                "owner": torch.from_numpy(owner).clone(),
                "cats": torch.from_numpy(cats).clone(),
                "scalars": torch.from_numpy(scalars).clone(),
                "target_state": torch.from_numpy(target_state).clone(),
            },
            batch_size=torch.Size(()),
        )

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

    def _fill_pokemon_row(
        self, pokemon: Pokemon | None, row: int, is_active: bool
    ) -> None:
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
        tools = pokemon.tools
        if len(tools) > self._tool_cap:
            self._warn_truncation("tool", len(tools), self._tool_cap)
            tools = tools[: self._tool_cap]
        for column, card in enumerate(tools):
            tool_id[row, column] = card.id
        energy_cards = pokemon.energyCards
        if len(energy_cards) > self._energy_cap:
            self._warn_truncation("energy_cards", len(energy_cards), self._energy_cap)
            energy_cards = energy_cards[: self._energy_cap]
        for column, card in enumerate(energy_cards):
            energy_card_ids[row, column] = card.id
        pre_evolutions = pokemon.preEvolution
        if len(pre_evolutions) > self._evolution_cap:
            self._warn_truncation(
                "pre_evolution", len(pre_evolutions), self._evolution_cap
            )
            pre_evolutions = pre_evolutions[: self._evolution_cap]
        for column, card in enumerate(pre_evolutions):
            pre_evolution_ids[row, column] = card.id
        # Feature columns: 0-3 HP block and freshness, 4 is-active (set
        # above), 5-7 attachment counts, 8+ histogram of provided energy
        # units by type.
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
        return

    def _encode_my_zones(self, state: State, agent_seat: int) -> TensorDict:
        """
        Encode the agent's hand, discard pile and prizes.

        :param state: Current engine state.
        :param agent_seat: Player index of the agent.
        :return: TensorDict with padded ID tables and masks.
        """
        player = state.players[agent_seat]
        entries = self._stage_public_zones(
            player,
            self._np_my_discard_ids,
            self._np_my_discard_mask,
            self._np_my_prize_ids,
            self._np_my_prize_mask,
        )
        hand_ids = [card.id for card in player.hand] if player.hand is not None else []
        entries["hand_ids"], entries["hand_mask"] = self._stage_id_list(
            hand_ids, self._np_hand_ids, self._np_hand_mask, "hand"
        )
        return TensorDict(entries, batch_size=torch.Size(()))

    def _encode_opp_zones(self, state: State, agent_seat: int) -> TensorDict:
        """
        Encode the opponent's public zones (discard pile and prizes).

        :param state: Current engine state.
        :param agent_seat: Player index of the agent.
        :return: TensorDict with padded ID tables and masks.
        """
        entries = self._stage_public_zones(
            state.players[1 - agent_seat],
            self._np_opp_discard_ids,
            self._np_opp_discard_mask,
            self._np_opp_prize_ids,
            self._np_opp_prize_mask,
        )
        return TensorDict(entries, batch_size=torch.Size(()))

    def _stage_public_zones(
        self,
        player: PlayerState,
        discard_ids_buffer: np.ndarray,
        discard_mask_buffer: np.ndarray,
        prize_ids_buffer: np.ndarray,
        prize_mask_buffer: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        """
        Stage the zones that are visible for both players: the discard pile
        and the prizes (face-down prizes keep a mask slot with ID 0).

        :param player: Player whose zones to encode.
        :param discard_ids_buffer: Staging buffer for the discard card IDs.
        :param discard_mask_buffer: Staging buffer for the discard mask.
        :param prize_ids_buffer: Staging buffer for the prize card IDs.
        :param prize_mask_buffer: Staging buffer for the prize mask.
        :return: Dict with ``discard_ids``/``discard_mask`` and
            ``prize_ids``/``prize_mask`` tensors.
        """
        discard_ids, discard_mask = self._stage_id_list(
            [card.id for card in player.discard],
            discard_ids_buffer,
            discard_mask_buffer,
            "discard",
        )
        prize_ids, prize_mask = self._stage_id_list(
            [card.id if card is not None else 0 for card in player.prize],
            prize_ids_buffer,
            prize_mask_buffer,
            "prize",
        )
        return {
            "discard_ids": discard_ids,
            "discard_mask": discard_mask,
            "prize_ids": prize_ids,
            "prize_mask": prize_mask,
        }

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
        return torch.from_numpy(ids_buffer).clone(), torch.from_numpy(
            mask_buffer
        ).clone()

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
        ids_tensor, mask_tensor = self._stage_id_list(
            card_ids, ids_buffer, mask_buffer, zone_name
        )
        return TensorDict(
            {"ids": ids_tensor, "mask": mask_tensor}, batch_size=torch.Size(())
        )

    def _warn_truncation(self, zone_name: str, length: int, cap: int) -> None:
        """
        Warn once per zone when its content exceeds the padded capacity.

        :param zone_name: Zone label for the log message.
        :param length: Actual number of entries.
        :param cap: Padded table size.
        """
        if zone_name not in self._warned_zones:
            logger.warning(
                "Zone %s has %d entries, truncating to cap %d.", zone_name, length, cap
            )
            self._warned_zones.add(zone_name)
