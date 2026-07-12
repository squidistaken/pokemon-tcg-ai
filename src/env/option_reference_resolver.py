from cg.api import AreaType, Option, OptionType, Pokemon, SelectData, State


class OptionReferenceResolver:
    """
    Resolves an option's zone references to concrete card and attack IDs.

    Every lookup is a pure function of the engine's ``state``/``select`` and
    the option being resolved: the resolver holds no buffers or caps of its
    own, so it has no dependency on (and no coupling to) the observation
    encoder that calls it.
    """

    @staticmethod
    def resolve(
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
            return OptionReferenceResolver._card_id_at(
                state, select, owner_index, option.area, option.index
            ), 0, 0
        # Attached-card options reference the carrier Pokemon plus an index
        # into its attachments; both the attachment and carrier are exposed.
        if option_type in (OptionType.TOOL_CARD, OptionType.ENERGY_CARD, OptionType.ENERGY):
            pokemon = OptionReferenceResolver._pokemon_at(state, owner_index, option.area, option.index)
            if pokemon is None:
                return 0, 0, 0
            if option_type == OptionType.TOOL_CARD:
                attached = OptionReferenceResolver._card_in_list(pokemon.tools, option.toolIndex)
            else:
                attached = OptionReferenceResolver._card_in_list(pokemon.energyCards, option.energyIndex)
            return attached, pokemon.id, 0
        if option_type in (OptionType.PLAY, OptionType.ABILITY, OptionType.DISCARD):
            return OptionReferenceResolver._card_id_at(
                state, select, owner_index, option.area or AreaType.HAND, option.index
            ), 0, 0
        if option_type in (OptionType.ATTACH, OptionType.EVOLVE):
            played = OptionReferenceResolver._card_id_at(state, select, owner_index, option.area, option.index)
            target = OptionReferenceResolver._pokemon_at(state, owner_index, option.inPlayArea, option.inPlayIndex)
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
        """
        Look up the card ID at a (player, area, index) reference.

        :param state: Current engine state.
        :param select: Current selection (source of the deck-search list).
        :param player_index: Absolute owner index of the referenced zone.
        :param area: Referenced area, or None.
        :param index: Index within the area, or None.
        :return: Card ID, or 0 when the reference is absent, face-down, or
            in a zone the agent cannot see.
        :raises IndexError: If the reference points outside a visible zone.
        """
        if area is None or index is None:
            return 0
        player = state.players[player_index]
        if area == AreaType.HAND:
            if player.hand is None:
                return 0
            return OptionReferenceResolver._card_in_list(player.hand, index)
        if area == AreaType.DISCARD:
            return OptionReferenceResolver._card_in_list(player.discard, index)
        if area == AreaType.ACTIVE:
            pokemon = OptionReferenceResolver._pokemon_at(state, player_index, area, index)
            return pokemon.id if pokemon is not None else 0
        if area == AreaType.BENCH:
            pokemon = OptionReferenceResolver._pokemon_at(state, player_index, area, index)
            return pokemon.id if pokemon is not None else 0
        if area == AreaType.PRIZE:
            return OptionReferenceResolver._card_in_list(player.prize, index)
        if area == AreaType.STADIUM:
            return OptionReferenceResolver._card_in_list(state.stadium, index)
        if area == AreaType.DECK:
            if select is not None and select.deck is not None:
                return OptionReferenceResolver._card_in_list(select.deck, index)
            return 0
        if area == AreaType.LOOKING:
            if state.looking is None:
                return 0
            return OptionReferenceResolver._card_in_list(state.looking, index)
        return 0

    @staticmethod
    def _pokemon_at(
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
        :return: The Pokemon, or None when the reference is absent or the
            slot holds a face-down card.
        :raises IndexError: If ``index`` falls outside the referenced board
            area.
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
        if not 0 <= index < len(slots):
            raise IndexError(
                f"Pokemon reference index {index} out of range for {area.name} of size {len(slots)}."
            )
        return slots[index]

    @staticmethod
    def _card_in_list(cards: list, index: int | None) -> int:
        """
        Read a card ID from a card list.

        :param cards: List of ``Card`` objects (or None entries for
            face-down cards).
        :param index: Index to read, or None for an absent reference.
        :return: The card's ID, or 0 when the reference is absent or the
            card is face-down.
        :raises IndexError: If ``index`` falls outside the list; an engine
            reference into a visible zone must always resolve.
        """
        if index is None:
            return 0
        if not 0 <= index < len(cards):
            raise IndexError(
                f"Card reference index {index} out of range for zone of size {len(cards)}."
            )
        card = cards[index]
        return card.id if card is not None else 0