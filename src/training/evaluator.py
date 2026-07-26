import logging
from collections.abc import Callable

import torch
from tensordict import TensorDict
from torch import nn
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Measures the policy against a fixed reference opponent.

    Under self-play the training win-rate is uninformative by construction: the
    league tracks the learner, so it hovers near 0.5 no matter how strong the
    policy becomes. This evaluator restores a readable learning curve by
    periodically playing the current policy against a *frozen* reference (in
    practice :class:`~src.env.random_opponent.RandomOpponent`), on its own
    environment that is never used for collection.

    Episodes are played one at a time on a single-process environment that is
    never used for collection. Action selection is deterministic by default
    (the distribution's mode rather than a sample), so the reported win-rate
    reflects the policy's committed choices and carries no exploration noise;
    set ``deterministic=False`` to score the sampling behaviour actually used
    during collection, which is noisier but matches training conditions.

    Evaluation is serial and therefore pure overhead on collection throughput.
    Its cost is bounded entirely by how often it runs and how many episodes it
    plays (``eval_interval`` / ``eval_episodes``), not by which action-selection
    mode is chosen — the two modes cost the same.
    """

    def __init__(
            self,
            env_factory: Callable[[], EnvBase],
            n_episodes: int = 100,
            max_steps: int = 2000,
            device: torch.device | str = "cpu",
            deterministic: bool = True,
    ) -> None:
        """
        :param env_factory: Builds the evaluation environment; called once and
            the instance reused across evaluations.
        :param n_episodes: Episodes played per evaluation. Higher is less noisy
            but costs collection throughput.
        :param max_steps: Safety cap on steps per episode, guarding against an
            episode that never terminates.
        :param device: Device the policy lives on. The environment always steps
            on CPU, so observations are moved across per step.
        :param deterministic: Take the distribution's mode instead of sampling
            from it. Defaults to True for a low-variance progress signal.
        """
        self._env_factory = env_factory
        self._n_episodes = n_episodes
        self._max_steps = max_steps
        self._device = torch.device(device)
        self._deterministic = deterministic
        self._env: EnvBase | None = None

    def _get_env(self) -> EnvBase:
        """
        Return the evaluation environment, building it on first use.

        :return: The evaluation environment instance.
        """
        if self._env is None:
            self._env = self._env_factory()
        return self._env

    @torch.no_grad()
    def evaluate(self, policy: nn.Module) -> dict[str, float]:
        """
        Play ``n_episodes`` against the reference opponent and score them.

        Episodes that never terminated — cut off by this evaluator's step cap
        or by the environment's own truncation — produced no outcome, so they
        are excluded from the rates (their zero reward would otherwise read as
        a draw) and reported separately as ``unfinished_episodes``.

        :param policy: Collection policy (any tensordict module writing
            ``action``); used in eval mode and restored to its previous mode
            afterwards.
        :return: Win/draw/loss rates over the episodes that terminated, plus
            the terminated and unfinished episode counts and the mean length
            of a terminated episode.
        """
        env = self._get_env()
        was_training = policy.training
        policy.eval()
        wins = 0
        draws = 0
        unfinished = 0
        total_steps = 0
        exploration = (
            ExplorationType.DETERMINISTIC if self._deterministic else ExplorationType.RANDOM
        )
        try:
            with set_exploration_type(exploration):
                for _ in range(self._n_episodes):
                    reward, steps, terminated = self._play_episode(env, policy)
                    if not terminated:
                        unfinished += 1
                        continue
                    wins += int(reward > 0)
                    draws += int(reward == 0)
                    total_steps += steps
        finally:
            policy.train(was_training)

        episodes = max(self._n_episodes - unfinished, 0)
        scored = max(episodes, 1)
        metrics = {
            "win_rate": wins / scored,
            "draw_rate": draws / scored,
            "loss_rate": (episodes - wins - draws) / scored,
            "episodes": float(episodes),
            "unfinished_episodes": float(unfinished),
            "mean_episode_length": total_steps / scored,
        }
        if unfinished:
            logger.warning(
                "%d of %d evaluation episodes did not terminate and are excluded from the "
                "rates; raise max_steps if this persists.",
                unfinished,
                self._n_episodes,
            )
        logger.info(
            "Evaluation over %d terminated episodes: win_rate=%.3f draw_rate=%.3f",
            episodes,
            metrics["win_rate"],
            metrics["draw_rate"],
        )
        return metrics

    def _play_episode(self, env: EnvBase, policy: nn.Module) -> tuple[float, int, bool]:
        """
        Play one episode to termination and report its outcome.

        Stepped manually rather than via ``env.rollout`` so the CPU environment
        and a possibly GPU-resident policy can be bridged explicitly on each
        step, and so the terminal reward is read directly.

        :param env: Environment to play in.
        :param policy: Policy producing the action.
        :return: Terminal reward (the win/draw/loss signal), step count, and
            whether the episode terminated. The environment can also end an
            episode by truncation (its engine-selection cap), which is done
            but not terminated and carries a zero reward, so termination
            rather than doneness is what makes the reward meaningful.
        """
        tensordict = env.reset()
        assert isinstance(tensordict, TensorDict)
        reward = 0.0
        steps = 0
        for _ in range(self._max_steps):
            action_td = policy(tensordict.to(self._device)).to("cpu")
            tensordict = env.step(action_td)
            steps += 1
            reward = float(tensordict["next", "reward"].reshape(-1)[-1])
            if bool(tensordict["next", "done"].reshape(-1)[-1]):
                return reward, steps, bool(tensordict["next", "terminated"].reshape(-1)[-1])
            tensordict = step_mdp(tensordict)
        logger.warning("Evaluation episode hit the %d-step cap without terminating.", self._max_steps)
        return reward, steps, False

    def close(self) -> None:
        """
        Release the evaluation environment.
        """
        if self._env is not None:
            self._env.close()
            self._env = None
