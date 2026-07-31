import logging
import statistics
from collections.abc import Callable

import torch
from tensordict import TensorDict
from torch import nn
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

logger = logging.getLogger(__name__)


def _episode_archetype(env: EnvBase) -> str | None:
    """
    Archetype the agent piloted in the episode the env just finished.

    Reads the deck labels and agent seat off the underlying
    :class:`~src.env.tcg_env.TCGEnv`, unwrapping any env transforms. Returns
    None when the env carries no labels (an unlabelled deck pool or the
    fixed-deck baseline), which leaves the aggregate-only path untouched.

    :param env: The evaluation environment, possibly transform-wrapped.
    :return: The agent's deck archetype, or None when unlabelled.
    """
    base: object = env
    while (inner := getattr(base, "base_env", None)) is not None:
        base = inner
    labels = getattr(base, "deck_labels", None)
    if labels is None:
        return None

    return labels[getattr(base, "agent_seat", 0)]


def _archetype_metrics(per_archetype: dict[str, list[int]]) -> dict[str, float]:
    """
    Summarize per-archetype tallies into generalization metrics.

    :param per_archetype: Mapping archetype -> ``[scored episodes, wins]``.
    :return: Macro mean/std/min/max/worst-quartile, the archetype count, and one
        ``archetype_win_rate/<name>`` series per archetype. Empty when nothing
        was recorded (unlabelled pool).
    """
    if not per_archetype:
        return {}
    rates = {
        archetype: wins / scored
        for archetype, (scored, wins) in per_archetype.items()
    }
    values = list(rates.values())
    ordered = sorted(values)
    worst_quartile = ordered[: max(1, len(ordered) // 4)]
    metrics: dict[str, float] = {
        "archetype_count": float(len(rates)),
        "archetype_win_rate_mean": statistics.fmean(values),
        "archetype_win_rate_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "archetype_win_rate_min": min(values),
        "archetype_win_rate_worst_quartile": statistics.fmean(worst_quartile),
        "archetype_win_rate_max": max(values),
    }
    for archetype, rate in rates.items():
        metrics[f"archetype_win_rate/{archetype}"] = rate
    return metrics


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

    When the evaluation environment draws from a labelled deck pool and
    ``per_archetype`` is set, each episode is additionally attributed to the
    archetype the agent piloted, and the report carries a per-archetype
    win-rate plus its spread across archetypes. A wide spread or a low
    worst-case archetype is the generalization failure the aggregate win-rate
    hides.
    """

    def __init__(
            self,
            env_factory: Callable[[], EnvBase],
            n_episodes: int = 100,
            max_steps: int = 2000,
            device: torch.device | str = "cpu",
            deterministic: bool = True,
            per_archetype: bool = True,
    ) -> None:
        """
        :param env_factory: Builds the evaluation environment; called once and
            the instance reused across evaluations.
        :param n_episodes: Episodes played per evaluation. Higher is less noisy
            but costs collection throughput. Under a per-archetype breakdown
            this budget is spread across the held-out archetypes, so raise it
            when the pool is wide or the per-archetype rates read as noise.
        :param max_steps: Safety cap on steps per episode, guarding against an
            episode that never terminates.
        :param device: Device the policy lives on. The environment always steps
            on CPU, so observations are moved across per step.
        :param deterministic: Take the distribution's mode instead of sampling
            from it. Defaults to True for a low-variance progress signal.
        :param per_archetype: Break the win-rate down by the agent's deck
            archetype and report its spread across archetypes.
        """
        self._env_factory = env_factory
        self._n_episodes = n_episodes
        self._max_steps = max_steps
        self._device = torch.device(device)
        self._deterministic = deterministic
        self._per_archetype = per_archetype
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
        # archetype -> [scored episodes, wins]; populated only when the eval env
        # exposes a labelled deck pool and per_archetype is on.
        per_archetype: dict[str, list[int]] = {}
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
                    win = int(reward > 0)
                    wins += win
                    draws += int(reward == 0)
                    total_steps += steps
                    if self._per_archetype:
                        archetype = _episode_archetype(env)
                        if archetype is not None:
                            tally = per_archetype.setdefault(archetype, [0, 0])
                            tally[0] += 1
                            tally[1] += win
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
        metrics.update(_archetype_metrics(per_archetype))
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
        if "archetype_win_rate_mean" in metrics:
            logger.info(
                "Per-archetype win-rate over %d archetype(s): mean=%.3f std=%.3f "
                "min=%.3f worst_quartile=%.3f max=%.3f",
                int(metrics["archetype_count"]),
                metrics["archetype_win_rate_mean"],
                metrics["archetype_win_rate_std"],
                metrics["archetype_win_rate_min"],
                metrics["archetype_win_rate_worst_quartile"],
                metrics["archetype_win_rate_max"],
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
