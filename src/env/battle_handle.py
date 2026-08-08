import ctypes
import json

from cg.api import Observation, to_observation_class
from cg.sim import lib

_DECK_ERRORS = {
    1: "a card ID that is not in the engine's card database",
    2: "more than 4 copies of the same card name (Basic Energy is exempt)",
    3: "no Basic Pokemon",
    4: "more than 1 ACE SPEC card",
}


class BattleHandle:
    """
    Per-instance handle to a single battle inside the cabt C engine.

    The bundled ``cg.game`` module stores its battle pointer in a global class
    attribute, which limits a process to one battle at a time. This wrapper
    keeps the pointer per instance instead, which allows multiple interleaved
    battles in one process and is required to run several environments per
    worker (e.g. with torchrl's ``SerialEnv``).
    """

    def __init__(self) -> None:
        """
        Create an idle handle; call :meth:`start` to begin a battle.
        """
        self._battle_ptr: int | None = None
        self._select_player: int = -1

    @property
    def active(self) -> bool:
        """
        Whether a battle is currently in progress on this handle.

        :return: True if a battle has been started and not yet finished.
        """
        return self._battle_ptr is not None

    @property
    def select_player(self) -> int:
        """
        Index of the player the engine expects the next selection from.

        :return: Player index (0 or 1), or -1 if no battle is active.
        """
        return self._select_player

    def start(self, deck0: list[int], deck1: list[int]) -> Observation:
        """
        Start a new battle between two decks.

        :param deck0: 60 card IDs for player 0.
        :param deck1: 60 card IDs for player 1.
        :return: First observation of the battle.
        :raises ValueError: If either deck is not 60 cards, or the engine
            rejects one as illegal under the deck-construction rules.
        """
        if self._battle_ptr is not None:
            raise RuntimeError("A battle is already running on this handle.")
        if len(deck0) != 60 or len(deck1) != 60:
            raise ValueError("Each deck must contain exactly 60 cards.")
        cards = deck0 + deck1
        start_data = lib.BattleStart((ctypes.c_int * len(cards))(*cards))
        if not start_data.battlePtr:
            reason = _DECK_ERRORS.get(start_data.errorType)
            if reason is not None:
                raise ValueError(
                    f"Engine rejected player {start_data.errorPlayer}'s deck: "
                    f"it has {reason}."
                )
            raise RuntimeError(
                f"BattleStart failed (errorPlayer={start_data.errorPlayer}, "
                f"errorType={start_data.errorType})."
            )
        self._battle_ptr = start_data.battlePtr
        return self._read_observation()

    def select(self, select_list: list[int]) -> Observation:
        """
        Submit the chosen option indices for the current selection.

        :param select_list: Option indices, between ``select.minCount`` and
            ``select.maxCount`` entries with no duplicates.
        :return: Observation after the engine has processed the selection.
        """
        self._require_active()
        arg = (ctypes.c_int * len(select_list))(*select_list)
        error = lib.Select(self._battle_ptr, arg, len(select_list))
        if error != 0:
            raise RuntimeError(
                f"Select failed with error code {error} for selection {select_list}."
            )
        return self._read_observation()

    def finish(self) -> None:
        """
        End the battle and release the engine memory. Safe to call when idle.
        """
        if self._battle_ptr is not None:
            lib.BattleFinish(self._battle_ptr)
            self._battle_ptr = None
            self._select_player = -1

    def _read_observation(self) -> Observation:
        """
        Fetch and parse the current battle observation from the engine.

        :return: Parsed observation dataclass.
        """
        serial_data = lib.GetBattleData(self._battle_ptr)
        self._select_player = serial_data.selectPlayer
        obs_dict = json.loads(serial_data.json.decode())
        return to_observation_class(obs_dict)

    def _require_active(self) -> None:
        """
        Raise if no battle is currently running on this handle.
        """
        if self._battle_ptr is None:
            raise RuntimeError("No battle is running on this handle.")
