import logging
import random
from typing import Callable

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Binary, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from cg.api import Observation
from .battle_handle import BattleHandle
from .random_opponent import RandomOpponent
from .observation_encoder import ObservationEncoder
from .structured_observation_encoder import StructuredObservationEncoder

logger = logging.getLogger(__name__)


class TCGEnv(EnvBase):
    """
    Single-agent TorchRL environment over the cabt Pokémon TCG engine.

    Each engine selection is presented as a discrete action over a padded,
    masked option space: action ``i < max_options`` picks option ``i`` of the
    current ``select``, and the last action index is a synthetic "stop" used
    to terminate multi-select accumulation. Selections with
    ``maxCount > 1`` are decomposed into sequential single picks (already
    picked options are masked out; "stop" becomes legal once ``minCount``
    picks have been made) and submitted to the engine as one call.

    The opponent seat is played inside :meth:`_step` by an injected opponent
    policy, so the environment behaves as a standard single-agent env. The
    agent's seat is randomized on every reset. The reward is terminal only:
    +1 for a win, -1 for a loss and ``reward_draw`` for a draw.

    ``max_options`` must exceed the largest option list the engine can
    produce; longer lists are truncated with a warning.
    """

    def __init__(
            self,
            deck0: list[int],
            deck1: list[int],
            max_options: int = 96,
            opponent: Callable[[Observation], list[int]] | None = None,
            reward_draw: float = 0.0,
            max_engine_selections: int = 5000,
            seed: int | None = None,
            device: torch.device | str | None = None,
            encoder: ObservationEncoder | None = None,
    ) -> None:
        """
        :param deck0: 60 card IDs for player 0.
        :param deck1: 60 card IDs for player 1.
        :param max_options: Padded size of the option space (stop action excluded).
        :param opponent: Policy playing the non-agent seat; random if None.
            If it exposes ``on_reset()`` (e.g. :class:`OpponentPool`), that is
            called at every episode start; if it exposes ``seed(int)``, it is
            reseeded through :meth:`set_seed`.
        :param reward_draw: Terminal reward assigned on a draw.
        :param max_engine_selections: Safety cap on engine selections per
            episode; exceeding it truncates the episode.
        :param seed: Seed for seat randomization and the default opponent.
        :param device: Device of the produced tensordicts.
        :param encoder: Observation encoder; a default-capacity
            :class:`StructuredObservationEncoder` matching ``max_options``
            if None.
        """
        super().__init__(device=device, batch_size=torch.Size(()))
        self._deck0 = list(deck0)
        self._deck1 = list(deck1)
        self._max_options = max_options
        self._stop_index = max_options
        self._reward_draw = reward_draw
        self._max_engine_selections = max_engine_selections
        self._handle = BattleHandle()
        self._encoder = encoder if encoder is not None else StructuredObservationEncoder(max_options=max_options)
        self._opponent = opponent if opponent is not None else RandomOpponent(seed)
        self._rng = random.Random(seed)
        self._agent_seat = 0
        self._pending: Observation | None = None
        self._chosen: list[int] = []
        self._selection_count = 0
        self._truncate_flag = False
        self._warned_overflow = False

        n_actions = max_options + 1
        self.observation_spec = Composite(
            observation=self._encoder.spec(),
            action_mask=Binary(n=n_actions, dtype=torch.bool),
        )
        self.action_spec = Categorical(n_actions, dtype=torch.int64)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.done_spec = Composite(
            done=Binary(1, dtype=torch.bool),
            terminated=Binary(1, dtype=torch.bool),
            truncated=Binary(1, dtype=torch.bool),
        )

    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs) -> TensorDictBase:
        """
        Start a new battle and advance it to the agent's first selection.

        :param tensordict: Optional reset input (unused).
        :return: Tensordict with the initial observation and action mask.
        """
        while True:
            self._handle.finish()
            if hasattr(self._opponent, "on_reset"):
                self._opponent.on_reset()
            self._agent_seat = self._rng.randint(0, 1)   # flip a coint to decide who plays first
            self._selection_count = 0
            self._truncate_flag = False
            self._chosen = []
            observation = self._handle.start(self._deck0, self._deck1)
            observation = self._advance_to_agent(observation)
            if not self._game_over(observation) and not self._truncate_flag:
                break
        self._pending = observation
        return self._build_obs_tensordict()

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Apply one agent action: pick an option or stop a multi-select.

        A pick that completes the current selection (or a stop action)
        submits the accumulated indices to the engine, after which the
        opponent plays until it is the agent's seat again or the battle ends.

        :param tensordict: Input tensordict containing the "action" key.
        :return: Tensordict with next observation, mask, reward and done flags.
        """
        action = int(tensordict["action"].item())
        select = self._pending.select
        submit: list[int] | None = None
        if action == self._stop_index:
            submit = list(self._chosen)
        else:
            self._chosen.append(action)
            if len(self._chosen) >= select.maxCount:
                submit = list(self._chosen)

        if submit is None:
            out = self._build_obs_tensordict()
            self._set_step_keys(out, reward=0.0, terminated=False, truncated=False)
            return out

        observation = self._engine_select(submit)
        self._chosen = []
        observation = self._advance_to_agent(observation)
        self._pending = observation
        terminated = self._game_over(observation)
        truncated = self._truncate_flag and not terminated
        reward = self._terminal_reward(observation) if terminated else 0.0
        out = self._build_obs_tensordict()
        self._set_step_keys(out, reward=reward, terminated=terminated, truncated=truncated)
        return out

    def _set_seed(self, seed: int | None) -> None:
        """
        Seed seat randomization and the opponent policy.

        :param seed: Seed value; None leaves the generators untouched.
        """
        if seed is None:
            return
        self._rng.seed(seed)
        if hasattr(self._opponent, "seed"):
            self._opponent.seed(seed + 1)

    def close(self, *, raise_if_closed: bool = True) -> None:
        """
        Finish any running battle and close the environment.

        :param raise_if_closed: Forwarded to :meth:`EnvBase.close`.
        """
        self._handle.finish()
        super().close(raise_if_closed=raise_if_closed)

    def _advance_to_agent(self, observation: Observation) -> Observation:
        """
        Step the engine until the agent must select or the battle is over.

        Plays the opponent's selections and auto-submits empty selections
        (``maxCount == 0``) for either seat.

        :param observation: Observation to advance from.
        :return: Observation at the agent's next selection or at game end.
        """
        while not self._game_over(observation) and not self._truncate_flag:
            select = observation.select
            if select is None:
                raise RuntimeError("Engine returned no selection while the battle is running.")
            if select.maxCount == 0:
                observation = self._engine_select([])
                continue
            if self._handle.select_player == self._agent_seat:
                break
            observation = self._engine_select(self._opponent(observation))
        return observation

    def _engine_select(self, select_list: list[int]) -> Observation:
        """
        Submit a selection to the engine, tracking the per-episode cap.

        :param select_list: Option indices to submit.
        :return: Observation after the selection.
        """
        self._selection_count += 1
        if self._selection_count >= self._max_engine_selections:
            self._truncate_flag = True
        return self._handle.select(select_list)

    def _game_over(self, observation: Observation) -> bool:
        """
        Whether the battle has finished.

        :param observation: Observation to inspect.
        :return: True if the engine reported a result.
        """
        return observation.current.result != -1

    def _terminal_reward(self, observation: Observation) -> float:
        """
        Reward from the agent's perspective at the end of the battle.

        :param observation: Terminal observation.
        :return: +1 on win, -1 on loss, ``reward_draw`` otherwise.
        """
        result = observation.current.result
        if result == self._agent_seat:
            return 1.0
        if result == 1 - self._agent_seat:
            return -1.0
        return self._reward_draw

    def _build_obs_tensordict(self) -> TensorDict:
        """
        Encode the pending observation and action mask into a tensordict.

        :return: Tensordict with the structured "observation" entry and the
            "action_mask" key.
        """
        obs = self._encoder.encode(self._pending, self._agent_seat, len(self._chosen))
        return TensorDict(
            {"observation": obs.to(self.device), "action_mask": self._build_mask().to(self.device)},
            batch_size=torch.Size(()),
        )

    def _build_mask(self) -> torch.Tensor:
        """
        Build the action mask for the pending selection.

        Options already picked in the current accumulation are masked out;
        the stop action is legal once ``minCount`` picks have been made. On a
        terminal observation only the stop action is marked legal so that the
        spec never carries an all-False mask.

        :return: Bool tensor of shape ``(max_options + 1,)``.
        """
        mask = torch.zeros(self._max_options + 1, dtype=torch.bool)
        select = self._pending.select
        if select is None or self._game_over(self._pending) or self._truncate_flag:
            mask[self._stop_index] = True
            return mask
        n_options = len(select.option)
        if n_options > self._max_options:
            if not self._warned_overflow:
                logger.warning(
                    "Selection offered %d options, truncating to max_options=%d.",
                    n_options,
                    self._max_options,
                )
                self._warned_overflow = True
            n_options = self._max_options
        mask[:n_options] = True
        for index in self._chosen:
            mask[index] = False
        if len(self._chosen) >= select.minCount:
            mask[self._stop_index] = True
        return mask

    def _set_step_keys(
            self,
            tensordict: TensorDict,
            reward: float,
            terminated: bool,
            truncated: bool,
    ) -> None:
        """
        Write reward and done flags into a step output tensordict.

        :param tensordict: Tensordict to write into.
        :param reward: Reward for the transition.
        :param terminated: Whether the battle reached a terminal state.
        :param truncated: Whether the episode was cut by the selection cap.
        """
        tensordict.set("reward", torch.tensor([reward], dtype=torch.float32, device=self.device))
        tensordict.set("terminated", torch.tensor([terminated], device=self.device))
        tensordict.set("truncated", torch.tensor([truncated], device=self.device))
        tensordict.set("done", torch.tensor([terminated or truncated], device=self.device))
