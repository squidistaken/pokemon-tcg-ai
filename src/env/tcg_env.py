import random
from collections.abc import Callable

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Binary, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from cg.api import Observation, SelectData, State

from .battle_handle import BattleHandle
from .curriculum_deck_sampler import NO_LEVEL
from .deck_sampler import DeckSampler, FixedDeckSampler, sample_for_seat
from .observation_encoder import ObservationEncoder
from .random_opponent import RandomOpponent
from .structured_observation_encoder import StructuredObservationEncoder


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
    produce; a selection that overflows it raises, since a truncated option
    would silently desynchronize the observation from the action space.
    """

    #: Battles started per :meth:`_reset` before giving up. A battle that ends
    #: before the agent's first selection is retried rather than handed over as
    #: an episode with no decisions in it; the cap keeps an unsatisfiable
    #: deck/cap combination from spinning forever.
    MAX_RESET_ATTEMPTS = 100

    def __init__(
        self,
        deck0: list[int] | None = None,
        deck1: list[int] | None = None,
        max_options: int = 96,
        opponent: Callable[[Observation], list[int]] | None = None,
        reward_draw: float = 0.0,
        max_engine_selections: int = 5000,
        seed: int | None = None,
        device: torch.device | str | None = None,
        encoder: ObservationEncoder | None = None,
        deck_sampler: DeckSampler | None = None,
        deck_switch_steps: int = 0,
    ) -> None:
        """
        :param deck0: 60 card IDs for player 0. Ignored if ``deck_sampler`` is
            given; otherwise required and wrapped in a fixed sampler.
        :param deck1: 60 card IDs for player 1. Same handling as ``deck0``.
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
        :param deck_sampler: Produces the ``(deck0, deck1)`` matchup at each
            reset. When None, a :class:`FixedDeckSampler` is built from
            ``deck0``/``deck1`` (which are then required).
        :param deck_switch_steps: Timesteps between resampling the decks.
            When 0, resampling happens on every reset.
        :raises ValueError: If neither a sampler nor both decks are given.
        """
        super().__init__(device=device, batch_size=torch.Size(()))
        if deck_sampler is None:
            if deck0 is None or deck1 is None:
                raise ValueError(
                    "TCGEnv needs either deck_sampler or both deck0 and deck1"
                )
            deck_sampler = FixedDeckSampler(deck0, deck1)
        self._deck_sampler = deck_sampler
        self._deck_switch_steps = deck_switch_steps
        self._steps_since_switch = 0

        # The active episode's decks, (re)sampled on every reset.
        self._deck0, self._deck1 = deck_sampler.sample()
        self._max_options = max_options
        self._stop_index = max_options
        self._reward_draw = reward_draw
        self._max_engine_selections = max_engine_selections
        self._handle = BattleHandle()
        self._encoder = (
            encoder
            if encoder is not None
            else StructuredObservationEncoder(max_options=max_options)
        )
        self._opponent = opponent if opponent is not None else RandomOpponent(seed)
        self._rng = random.Random(seed)
        self._agent_seat = 0
        self._pending: Observation | None = None
        self._chosen: list[int] = []
        self._selection_count = 0
        self._truncate_flag = False

        self._level_id = NO_LEVEL
        self._opponent_is_anchor = True

        n_actions = max_options + 1
        # ``level_id`` and ``opponent_is_anchor`` sit beside the observation
        # rather than inside it: they exist so the learner can attribute each
        # step back to the curriculum level and opponent that produced it, and
        # the model never reads them (it consumes only the keys named in its
        # backbone's ``in_keys``). Both are constant across an episode.
        self.observation_spec = Composite(
            observation=self._encoder.spec(),
            action_mask=Binary(n=n_actions, dtype=torch.bool),
            level_id=Unbounded(shape=(1,), dtype=torch.int64),
            opponent_is_anchor=Binary(1, dtype=torch.bool),
        )
        self.action_spec = Categorical(n_actions, dtype=torch.int64)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.done_spec = Composite(
            done=Binary(1, dtype=torch.bool),
            terminated=Binary(1, dtype=torch.bool),
            truncated=Binary(1, dtype=torch.bool),
        )

    @property
    def agent_seat(self) -> int:
        """
        Seat index (0 or 1) occupied by the agent in the current episode.

        :return: The agent's seat index.
        """
        return self._agent_seat

    @property
    def deck_labels(self) -> tuple[str, str] | None:
        """
        Archetype labels of the current episode's ``(deck0, deck1)``.

        Populated only when the deck sampler carries labels, or None
        otherwise.

        :return: The ``(deck0, deck1)`` archetype labels, or None when the
            sampler is unlabelled.
        """
        return getattr(self._deck_sampler, "last_labels", None)

    @property
    def pending_select(self) -> SelectData:
        """
        Selection data of the pending agent decision.

        Only valid between a reset and episode termination, where the env
        guarantees a pending selection exists.

        :return: The engine's selection data for the agent's current decision.
        """
        assert self._pending is not None and self._pending.select is not None, (
            "no pending selection; call reset() first"
        )
        return self._pending.select

    @property
    def already_chosen_option_count(self) -> int:
        """
        Number of options accumulated so far in the current multi-select.

        :return: Count of already-picked option indices.
        """
        return len(self._chosen)

    @property
    def current_state(self) -> State:
        """
        Raw engine state at the pending agent decision.

        Only valid between a reset and episode termination, where the env
        guarantees a pending observation exists.

        :return: The engine state visible to the agent.
        """
        assert self._pending is not None and self._pending.current is not None, (
            "no pending observation; call reset() first"
        )
        return self._pending.current

    # `tensordict` and `kwargs` are deliberately unused: TorchRL's EnvBase dictates
    # this exact override signature, and this env ignores the reset input because it
    # always starts a fresh battle. Suppressed for PyCharm (noinspection) and Ruff (noqa).
    # noinspection PyUnusedLocal
    def _reset(
        self,
        tensordict: TensorDictBase | None = None,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> TensorDictBase:
        """
        Start a new battle and advance it to the agent's first selection.

        :param tensordict: Optional reset input (unused).
        :return: Tensordict with the initial observation and action mask.
        """
        on_reset = getattr(self._opponent, "on_reset", None)
        if on_reset is not None:
            on_reset()
        self._opponent_is_anchor = bool(
            getattr(self._opponent, "active_is_anchor", True)
        )

        for attempt in range(self.MAX_RESET_ATTEMPTS):
            self._handle.finish()
            self._agent_seat = self._rng.randint(0, 1)
            self._selection_count = 0
            self._truncate_flag = False
            self._chosen = []
            # `attempt > 0` forces a fresh matchup on every retry. The
            # switch-interval test alone would not: the first attempt zeroes
            # `_steps_since_switch`, so with `deck_switch_steps > 0` every
            # later attempt re-deals the *same* pair that just failed to
            # produce an agent decision, and the loop can only burn all
            # MAX_RESET_ATTEMPTS and kill the worker.
            if attempt > 0 or self._steps_since_switch >= self._deck_switch_steps:
                # Seat-aware: _agent_seat is already drawn above, and a sampler
                # that pins the agent's deck or scores an ordered matchup needs
                # to know which of the two decks the agent will receive.
                self._deck0, self._deck1 = sample_for_seat(
                    self._deck_sampler, self._agent_seat
                )
                self._steps_since_switch = 0
                # Only meaningful under a curriculum sampler; every other
                # sampler leaves the level at NO_LEVEL, which the learner skips.
                self._level_id = int(getattr(self._deck_sampler, "level_id", NO_LEVEL))
            observation = self._handle.start(self._deck0, self._deck1)
            observation = self._advance_to_agent(observation)
            if not self._game_over(observation) and not self._truncate_flag:
                self._pending = observation
                return self._build_obs_tensordict()
        raise RuntimeError(
            f"No battle survived setup in {self.MAX_RESET_ATTEMPTS} attempts: every "
            f"one ended or hit the {self._max_engine_selections}-selection cap before "
            f"the agent could act. Check the sampled decks and max_engine_selections."
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Apply one agent action: pick an option or stop a multi-select.

        A pick that completes the current selection (or a stop action)
        submits the accumulated indices to the engine, after which the
        opponent plays until it is the agent's seat again or the battle ends.

        :param tensordict: Input tensordict containing the "action" key.
        :return: Tensordict with next observation, mask, reward and done flags.
        """
        self._steps_since_switch += 1
        action = int(tensordict["action"].item())
        submit: list[int] | None = None
        if action == self._stop_index:
            submit = list(self._chosen)
        else:
            self._chosen.append(action)
            if len(self._chosen) >= self.pending_select.maxCount:
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
        if terminated:
            self._report_outcome(reward)
        out = self._build_obs_tensordict()
        self._set_step_keys(
            out, reward=reward, terminated=terminated, truncated=truncated
        )
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
            self._opponent.seed(seed)
        self._deck_sampler.seed(seed)

    def close(self, *, raise_if_closed: bool = True) -> None:
        """
        Finish any running battle and close the environment.

        :param raise_if_closed: Forwarded to :meth:`EnvBase.close`.
        """
        self._handle.finish()
        super().close(raise_if_closed=raise_if_closed)

    def _report_outcome(self, reward: float) -> None:
        """
        Tell the opponent policy how the finished battle went, if it cares.

        Leagues that weight their members by strength (e.g.
        :class:`~src.env.pfsp_opponent_pool.PFSPOpponentPool`) need the result
        of each episode, and the terminal reward is only available here. Other
        opponents do not expose the hook and are left untouched, exactly as
        with ``on_reset``.

        :param reward: Terminal reward from the agent's perspective.
        """
        record_outcome = getattr(self._opponent, "record_outcome", None)
        if record_outcome is not None:
            record_outcome(reward)

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
                raise RuntimeError(
                    "Engine returned no selection while the battle is running."
                )
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

    @staticmethod
    def _game_over(observation: Observation) -> bool:
        """
        Whether the battle has finished.

        :param observation: Observation to inspect.
        :return: True if the engine reported a result.
        """
        state = observation.current
        return state is not None and state.result != -1

    def _terminal_reward(self, observation: Observation) -> float:
        """
        Reward from the agent's perspective at the end of the battle.

        :param observation: Terminal observation.
        :return: +1 on win, -1 on loss, ``reward_draw`` otherwise.
        """
        state = observation.current
        assert state is not None, "terminal observation must carry a state"
        result = state.result
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
        pending = self._pending
        assert pending is not None, "no pending observation; call reset() first"
        obs = self._encoder.encode(pending, self._agent_seat, len(self._chosen))
        return TensorDict(
            {
                "observation": obs.to(self.device),
                "action_mask": self._build_mask().to(self.device),
                "level_id": torch.tensor(
                    [self._level_id], dtype=torch.int64, device=self.device
                ),
                "opponent_is_anchor": torch.tensor(
                    [self._opponent_is_anchor], dtype=torch.bool, device=self.device
                ),
            },
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
        pending = self._pending
        assert pending is not None, "no pending observation; call reset() first"
        select = pending.select
        if select is None or self._game_over(pending) or self._truncate_flag:
            mask[self._stop_index] = True
            return mask
        n_options = len(select.option)
        if n_options > self._max_options:
            raise ValueError(
                f"Selection offers {n_options} options but max_options is {self._max_options}; "
                f"increase max_options so every option stays addressable."
            )
        mask[:n_options] = True
        for index in self._chosen:
            mask[index] = False
        if len(self._chosen) >= select.minCount:
            mask[self._stop_index] = True
        return mask

    def _set_step_keys(
        self, tensordict: TensorDict, reward: float, terminated: bool, truncated: bool
    ) -> None:
        """
        Encode reward and done flags into a step output tensordict.

        :param tensordict: Tensordict to encode the step outcome into.
        :param reward: Reward for the transition.
        :param terminated: Whether the battle reached a terminal state.
        :param truncated: Whether the episode was cut by the selection cap.
        """
        tensordict.set(
            "reward", torch.tensor([reward], dtype=torch.float32, device=self.device)
        )
        tensordict.set("terminated", torch.tensor([terminated], device=self.device))
        tensordict.set("truncated", torch.tensor([truncated], device=self.device))
        tensordict.set(
            "done", torch.tensor([terminated or truncated], device=self.device)
        )
