import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

from tools.head_to_head.match_statistics import EpisodeRecord

logger = logging.getLogger(__name__)


class HeadToHeadMatch:
    """
    Plays one policy against one frozen opponent and records every game.

    This is the measurement half of a two-way comparison. It differs from
    :class:`~src.training.evaluator.Evaluator` in what it keeps: the evaluator
    reduces a round to aggregate rates, which is all a training curve needs,
    while a significance test needs the per-episode outcomes plus the covariates
    that explain them. Seat and deck are recorded for exactly that reason.

    The engine shuffles with ``std::random_device`` and :meth:`BattleHandle.start`
    takes no seed, so two runs of the same matchup are different games and a
    given deal cannot be replayed for the other policy. Pairing on the deal is
    therefore impossible; the design blocks on what is controllable, the deck
    matchup and the seat, and buys the rest with sample size.
    """

    def __init__(
        self,
        env: EnvBase,
        policy: nn.Module,
        max_steps: int = 2000,
        deterministic: bool = True,
    ) -> None:
        """
        :param env: Evaluation environment with the opponent already installed.
        :param policy: Collection policy writing ``action``, played in the agent
            seat.
        :param max_steps: Safety cap on agent decisions per episode.
        :param deterministic: Take the distribution's mode instead of sampling.
            Greedy is the mode the submission runs in, so it is the default.
        """
        self._env = env
        self._policy = policy
        self._max_steps = max_steps
        self._deterministic = deterministic

    @torch.no_grad()
    def play(self, n_episodes: int) -> list[EpisodeRecord]:
        """
        Play ``n_episodes`` games and return one record each.

        :param n_episodes: Games to play.
        :return: Per-episode outcomes in play order.
        """
        was_training = self._policy.training
        self._policy.eval()
        exploration = (
            ExplorationType.DETERMINISTIC
            if self._deterministic
            else ExplorationType.RANDOM
        )
        records: list[EpisodeRecord] = []
        try:
            with set_exploration_type(exploration):
                for _ in range(n_episodes):
                    records.append(self._play_episode())
        finally:
            self._policy.train(was_training)
        return records

    def _play_episode(self) -> EpisodeRecord:
        """
        Play a single game to termination or the step cap.

        :return: The episode's outcome and covariates.
        """
        tensordict = self._env.reset()
        base = self._base_env()
        agent_seat = int(getattr(base, "agent_seat", 0))
        labels = getattr(base, "deck_labels", None) or ("unknown", "unknown")
        reward = 0.0
        terminated = False
        steps = 0
        for _ in range(self._max_steps):
            tensordict = self._policy(tensordict)
            tensordict = self._env.step(tensordict)
            steps += 1
            reward = float(tensordict["next", "reward"].reshape(-1)[-1])
            if bool(tensordict["next", "done"].reshape(-1)[-1]):
                terminated = bool(tensordict["next", "terminated"].reshape(-1)[-1])
                break
            tensordict = step_mdp(tensordict)
        else:
            logger.warning("Episode hit the %d-step cap.", self._max_steps)

        return EpisodeRecord(
            score=1.0 if reward > 0 else 0.0 if reward < 0 else 0.5,
            terminated=terminated,
            agent_seat=agent_seat,
            deck=labels[agent_seat],
            opponent_deck=labels[1 - agent_seat],
            steps=steps,
        )

    def _base_env(self) -> EnvBase:
        """
        Unwrap the transform stack down to the :class:`~src.env.tcg_env.TCGEnv`.

        :return: The innermost environment, which owns the seat and deck labels.
        """
        base: EnvBase = self._env
        while (inner := getattr(base, "base_env", None)) is not None:
            base = inner
        return base


def build_match_policy(
    checkpoint_path: str | Path,
    cfg: DictConfig,
    obs_spec: object,
    action_spec: object,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """
    Load a snapshot as a policy operator for the agent seat.

    The architecture comes from the config the checkpoint embeds, so a BC
    checkpoint and a self-play checkpoint with different backbone depths both
    load without the caller knowing which is which.

    :param checkpoint_path: Path to a :func:`save_actor_critic` snapshot.
    :param cfg: Fallback config for a legacy checkpoint that embeds none.
    :param obs_spec: Environment observation composite spec.
    :param action_spec: Environment action spec.
    :param device: Device for inference.
    :return: The policy operator, in eval mode on ``device``.
    """
    from src.policies.greedy_policy_opponent import load_actor_critic
    from src.policies.ppo_actor import build_ppo_operator

    actor_critic = load_actor_critic(
        checkpoint_path, cfg, obs_spec, action_spec, device
    )
    operator = build_ppo_operator(actor_critic, action_spec)
    return operator.get_policy_operator().to(device).eval()
