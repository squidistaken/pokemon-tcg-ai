"""Parse a downloaded episode replay into our own perspective's event timeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from cg.api import AreaType, LogType


class GameEndReason(IntEnum):
    """Why the match actually ended."""

    PRIZES_TAKEN = 1
    DECKED_OUT = 2
    NO_POKEMON_LEFT = 3
    CARD_EFFECT = 4


@dataclass
class AttackUsage:
    """One use of one attack, with the damage it's credited for."""

    attack_id: int
    card_id: int
    damage: int = 0
    knocked_out: bool = False


@dataclass
class ParsedEpisode:
    """One episode's game log.

    :ivar cards_seen: Card IDs that were ever in our hand.
    :ivar cards_played: Card IDs we played (PLAY or EVOLVE) at least once.
    :ivar evolutions_made: Card IDs of Pokemon we evolved *into*.
    :ivar pre_evolutions_available: Card IDs of pre-evolution Pokemon that
        were in play when we evolved from them.
    :ivar our_kos: Our own Pokemon knocked out (ACTIVE/BENCH -> DISCARD).
    :ivar opponent_kos: Opponent Pokemon knocked out, by the same signal.
    :ivar our_pokemon_lost: Card IDs of our own Pokemon knocked out at least
        once this episode - the fragility counterpart to ``cards_played``.
        Per-game like the other card-id sets (not a raw KO count), so it
        stays directly comparable to ``cards_played`` even when a card has
        multiple copies knocked out in the same game.
    :ivar our_kos_cards: Card IDs behind each ``our_kos`` increment, in
        order - unlike ``our_pokemon_lost``, preserves multiplicity (two
        copies of the same card knocked out in one game is two entries),
        needed to look up prize value per KO (an ex/megaEx Pokemon is worth
        more than one prize when knocked out).
    :ivar opponent_kos_cards: Card IDs behind each ``opponent_kos``
        increment, in order - the other side of ``our_kos_cards``.
    :ivar prized_cards: Our own cards revealed as having sat in our prizes.
    :ivar first_attack_turn: The competition's ``current.turn`` count (1 =
        starting player's first turn) when we landed our first attack, or
        None if we never did.
    :ivar opponent_attacks: The opponent's attacks against us, tracked the
        same way as ``attacks`` (damage/KO attribution) but from the other
        side - which enemy cards/attacks are actually hurting us, not just
        that we got hurt.
    :ivar final_turn: The last ``current.turn`` value observed - a proxy for
        how long the game ran, independent of whether we ever attacked.
    :ivar game_end_reason: How the match actually ended (see
        :class:`GameEndReason`), from the replay's own ``RESULT`` log entry -
        None if that entry was never observed (e.g. a truncated replay).
    """

    episode_id: int
    result: str
    opponent_team: str
    had_basic_pokemon: bool | None = None
    cards_seen: set[int] = field(default_factory=set)
    cards_played: set[int] = field(default_factory=set)
    evolutions_made: set[int] = field(default_factory=set)
    pre_evolutions_available: set[int] = field(default_factory=set)
    attacks: list[AttackUsage] = field(default_factory=list)
    our_kos: int = 0
    opponent_kos: int = 0
    our_pokemon_lost: set[int] = field(default_factory=set)
    our_kos_cards: list[int] = field(default_factory=list)
    opponent_kos_cards: list[int] = field(default_factory=list)
    prized_cards: set[int] = field(default_factory=set)
    first_attack_turn: int | None = None
    opponent_attacks: list[AttackUsage] = field(default_factory=list)
    final_turn: int | None = None
    game_end_reason: int | None = None


def _record_hand_snapshot(
    episode: ParsedEpisode, hand: list[dict[str, Any]] | None
) -> None:
    """
    Add every card ID in ``hand`` to ``episode.cards_seen``.

    :param episode: Episode being built; mutated in place.
    :param hand: The hand snapshot from one player's ``current`` observation,
        or None if absent.
    :return: None.
    """
    if not hand:
        return
    for card in hand:
        card_id = card.get("id")
        if card_id is not None:
            episode.cards_seen.add(card_id)


def _step_logs(
    our_step: dict[str, Any], previous_logs: list[dict[str, Any]] | None
) -> list[dict[str, Any]] | None:
    """
    :param our_step: This step's entry from our side's index into ``steps``.
    :param previous_logs: The last genuinely new batch returned by this
        function, or None if there hasn't been one yet.
    :return: The new log batch, or None if this step doesn't add one.
    """
    if our_step.get("status") not in ("ACTIVE", "DONE"):
        return None
    logs = our_step.get("observation", {}).get("logs", [])
    return None if logs == previous_logs else logs


def parse_replay(
    raw: dict[str, Any],
    *,
    episode_id: int,
    our_index: int,
    result: str,
    opponent_team: str,
) -> ParsedEpisode:
    """
    Reduce one raw replay JSON to a :class:`ParsedEpisode` from our side.

    :param raw: The full replay JSON as downloaded from Kaggle.
    :param episode_id: Kaggle's episode identifier, carried through unchanged.
    :param our_index: Which side (0 or 1) of the replay's ``steps`` was us.
    :param result: Outcome label ("win"/"loss"/"draw") from the episode
        summary, carried through unchanged.
    :param opponent_team: Opponent team name, carried through unchanged.
    :return: The parsed episode.
    """
    episode = ParsedEpisode(
        episode_id=episode_id, result=result, opponent_team=opponent_team
    )
    steps = raw.get("steps", [])
    previous_logs: list[dict[str, Any]] | None = None

    for step in steps:
        if our_index >= len(step):
            continue
        our_step = step[our_index]
        observation = our_step.get("observation", {})
        current = observation.get("current")
        current_turn = None
        if current:
            current_turn = current.get("turn")
            if current_turn is not None:
                episode.final_turn = current_turn
            players = current.get("players") or []
            if our_index < len(players):
                _record_hand_snapshot(episode, players[our_index].get("hand"))

        logs = _step_logs(our_step, previous_logs)
        if logs is None:
            continue
        previous_logs = logs

        pending_attack: AttackUsage | None = None
        pending_opponent_attack: AttackUsage | None = None
        for log in logs:
            log_type = log.get("type")

            if (
                log_type == LogType.HAS_BASIC_POKEMON
                and log.get("playerIndex") == our_index
            ):
                episode.had_basic_pokemon = bool(log.get("hasBasicPokemon"))

            elif (
                log_type in (LogType.PLAY, LogType.ATTACH)
                and log.get("playerIndex") == our_index
            ):
                card_id = log.get("cardId")
                if card_id is not None:
                    episode.cards_played.add(card_id)
                    episode.cards_seen.add(card_id)

            elif log_type == LogType.EVOLVE and log.get("playerIndex") == our_index:
                card_id = log.get("cardId")
                target_id = log.get("cardIdTarget")
                if card_id is not None:
                    episode.cards_played.add(card_id)
                    episode.cards_seen.add(card_id)
                    episode.evolutions_made.add(card_id)
                if target_id is not None:
                    episode.pre_evolutions_available.add(target_id)

            elif log_type == LogType.ATTACK:
                attack_id = log.get("attackId")
                card_id = log.get("cardId")
                attacker_index = log.get("playerIndex")
                if attack_id is not None and card_id is not None:
                    if attacker_index == our_index:
                        pending_attack = AttackUsage(
                            attack_id=attack_id, card_id=card_id
                        )
                        episode.attacks.append(pending_attack)
                        if (
                            episode.first_attack_turn is None
                            and current_turn is not None
                        ):
                            episode.first_attack_turn = current_turn
                    else:
                        pending_opponent_attack = AttackUsage(
                            attack_id=attack_id, card_id=card_id
                        )
                        episode.opponent_attacks.append(pending_opponent_attack)

            elif log_type == LogType.HP_CHANGE:
                value = log.get("value")
                target_index = log.get("playerIndex")
                if value and pending_attack is not None and target_index != our_index:
                    pending_attack.damage += max(0, -value)
                elif (
                    value
                    and pending_opponent_attack is not None
                    and target_index == our_index
                ):
                    pending_opponent_attack.damage += max(0, -value)

            elif log_type == LogType.MOVE_CARD:
                from_area = log.get("fromArea")
                to_area = log.get("toArea")
                player_index = log.get("playerIndex")
                if (
                    from_area in (AreaType.ACTIVE, AreaType.BENCH)
                    and to_area == AreaType.DISCARD
                    and player_index is not None
                ):
                    card_id = log.get("cardId")
                    if player_index == our_index:
                        episode.our_kos += 1
                        if card_id is not None:
                            episode.our_pokemon_lost.add(card_id)
                            episode.our_kos_cards.append(card_id)
                        if pending_opponent_attack is not None:
                            pending_opponent_attack.knocked_out = True
                    else:
                        episode.opponent_kos += 1
                        if card_id is not None:
                            episode.opponent_kos_cards.append(card_id)
                        if pending_attack is not None:
                            pending_attack.knocked_out = True
                elif (
                    from_area == AreaType.PRIZE
                    and to_area == AreaType.HAND
                    and player_index == our_index
                    and log.get("cardId") is not None
                ):
                    episode.prized_cards.add(log["cardId"])

            elif log_type == LogType.RESULT:
                reason = log.get("reason")
                if reason is not None:
                    episode.game_end_reason = reason

    return episode
