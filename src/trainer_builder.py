"""
Assembly of a runnable trainer from a composed Hydra config.

Everything :func:`src.train.main` needs between reading the config and calling
``train()`` lives here: the metric backends, the actor-critic and its warm
start, the self-play and evaluation opponents, and the run-directory paths that
keep concurrent runs from writing over each other.
"""

import logging
import random
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torchrl.data import Categorical, Composite
from torchrl.envs import EnvBase

from src.models.actor_critic import ActorCritic
from src.policies.greedy_policy_opponent import checkpoint_state_dict
from src.policies.ppo_actor import build_actor_critic
from src.policies.random_masked_policy import RandomMaskedPolicy
from src.training import (
    CrossPlayCallback,
    CurriculumStateCallback,
    Evaluator,
    MultiEvaluator,
    PPOTrainer,
    SnapshotCallback,
    Trainer,
    TrainingCallback,
    WeightsAndBiases,
    build_best_response_opponent_factory,
    build_curriculum,
    build_evaluator,
    build_opponent_factory,
    build_probe_specs,
    make_env_factories,
)
from src.training.collectors import AsyncCollectorOptions
from src.training.env_factory import OpponentFactory, _build_sampler_spec

logger = logging.getLogger(__name__)


def seed_everything(cfg: DictConfig) -> None:
    """
    Seed Python and Torch from the config, when the run asks for it.

    :param cfg: Hydra configuration with ``set_seed`` and ``seed``.
    """
    if not cfg.set_seed:
        return
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)


def build_callbacks(cfg: DictConfig) -> list[TrainingCallback]:
    """
    Instantiate the metric backends selected by the ``callbacks`` config group.

    :param cfg: Hydra configuration with a top-level ``callbacks`` list.
    :return: Instantiated callbacks; empty for ``callbacks=none``.
    """
    return [hydra.utils.instantiate(callback) for callback in cfg.callbacks]


def resolve_run_config(cfg: DictConfig) -> dict[str, Any]:
    """
    Resolve the Hydra config into a plain dict for the callbacks.

    :param cfg: Hydra configuration object.
    :return: The fully resolved config as nested plain Python types.
    """
    return cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))


def build_trainer(
    cfg: DictConfig,
    callbacks: list[TrainingCallback],
    run_config: dict[str, Any],
) -> Trainer:
    """
    Build the trainer named by ``cfg.agent.name``.

    ``ppo`` runs the shared-trunk actor-critic with clipped PPO and
    invalid-action masking; anything else runs the pure random-collection
    baseline.

    :param cfg: Hydra configuration object.
    :param callbacks: Metric backends to attach to the trainer.
    :param run_config: Resolved run config, forwarded to the callbacks.
    :return: A trainer ready for :meth:`~src.training.trainer.Trainer.train`.
    """
    if cfg.agent.name == "ppo":
        return _build_ppo_trainer(cfg, callbacks, run_config)
    return _build_baseline_trainer(cfg, callbacks, run_config)


def _build_baseline_trainer(
    cfg: DictConfig,
    callbacks: list[TrainingCallback],
    run_config: dict[str, Any],
) -> Trainer:
    """
    Build the random-collection baseline: no network, no optimization.

    :param cfg: Hydra configuration object.
    :param callbacks: Metric backends to attach to the trainer.
    :param run_config: Resolved run config, forwarded to the callbacks.
    :return: A configured :class:`~src.training.trainer.Trainer`.
    """
    return Trainer(
        env_factories=make_env_factories(cfg),
        policy=RandomMaskedPolicy(),
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        use_parallel_env=cfg.env.parallel,
        mp_start_method=cfg.env.mp_start_method,
        serial_for_single=cfg.env.serial_for_single,
        callbacks=callbacks,
        run_config=run_config,
        max_collector_restarts=_max_collector_restarts(cfg),
        rebuild_env_factories=_rebuild_env_factories(cfg),
        pipe_timeout=_pipe_timeout(cfg),
        collector_type=cfg.collector.get("type", "sync"),
        async_options=_async_collector_options(cfg),
    )


def _build_ppo_trainer(
    cfg: DictConfig,
    callbacks: list[TrainingCallback],
    run_config: dict[str, Any],
) -> PPOTrainer:
    """
    Assemble the PPO trainer, actor-critic and environment from the config.

    Derives the observation/action specs from a throwaway environment instance
    (built from the same factory as training), so the network is sized against
    the exact env contract.

    :param cfg: Hydra configuration with ``agent`` (PPO hyperparameters),
        ``model`` and ``env`` sections.
    :param callbacks: Metric backends to attach to the trainer.
    :param run_config: Resolved run config, forwarded to the callbacks.
    :return: A configured :class:`~src.training.ppo_trainer.PPOTrainer`.
    """
    obs_spec, action_spec = build_probe_specs(cfg)

    actor_critic = build_actor_critic(cfg, obs_spec, action_spec)
    start_frames, resume_state = _warm_start(cfg, actor_critic)

    checkpoint_dir = _resolve_checkpoint_dir(cfg)
    is_best_response = bool(cfg.train.get("best_response", False))
    if is_best_response:
        # Exploitability run: the learner both trains against and is scored
        # against one frozen agent, and its eval win-rate is that agent's
        # exploitability. No self-play league, so cross-play against the
        # learner's own history is meaningless.
        opponent_factory = build_best_response_opponent_factory(
            cfg, obs_spec, action_spec
        )
        eval_opponent_factory = opponent_factory
    else:
        opponent_factory = build_opponent_factory(
            cfg, obs_spec, action_spec, checkpoint_dir
        )
        eval_opponent_factory = None

    snapshot_interval = int(cfg.train.get("snapshot_interval", 0))
    cross_play_enabled = (
        not is_best_response
        and opponent_factory is not None
        and bool(cfg.train.get("cross_play", False))
    )
    eval_interval = int(cfg.train.get("eval_interval", 0))
    # Built once and shared with both consumers below when both are active, so
    # the held-out deck corpus is not parsed twice for identical data.
    eval_sampler_spec = (
        _build_sampler_spec(cfg, deck_split="eval")
        if cross_play_enabled and eval_interval > 0
        else None
    )

    checkpoint_loggers = [
        callback.log_checkpoint
        for callback in callbacks
        if isinstance(callback, WeightsAndBiases)
    ]
    # Snapshotting starts first and finishes before cross-play and W&B. This
    # makes the final checkpoint visible to cross-play and keeps W&B alive while
    # both final checkpoint and evaluation artifacts are logged.
    new_callbacks: list[TrainingCallback] = [
        SnapshotCallback(
            actor_critic=actor_critic,
            checkpoint_dir=checkpoint_dir,
            interval=snapshot_interval,
            checkpoint_loggers=checkpoint_loggers,
            registry_path=_resolve_checkpoint_registry(cfg),
            repo_root=Path(__file__).parents[1],
        )
    ]
    if snapshot_interval > 0 and cross_play_enabled:
        new_callbacks.append(
            CrossPlayCallback(
                actor_critic=actor_critic,
                cfg=cfg,
                obs_spec=obs_spec,
                action_spec=action_spec,
                checkpoint_dir=checkpoint_dir,
                output_dir=HydraConfig.get().runtime.output_dir,
                n_games=int(cfg.train.get("cross_play_games", 20)),
                max_checkpoints=int(cfg.train.get("cross_play_max_checkpoints", 8)),
                seed=int(cfg.seed),
                sampler_spec=eval_sampler_spec,
            )
        )
    callbacks = [*new_callbacks, *callbacks]

    curriculum = build_curriculum(cfg)
    if curriculum is not None:
        callbacks = [
            *callbacks,
            CurriculumStateCallback(
                curriculum=curriculum,
                state_dir=checkpoint_dir.parent / "curriculum",
                interval=int(
                    cfg.env.curriculum.get("state_interval")
                    or cfg.collector.frames_per_batch * 50
                ),
            ),
        ]

    frames_per_batch = int(
        cfg.agent.get("frames_per_batch", cfg.collector.frames_per_batch)
    )
    return PPOTrainer(
        env_factories=make_env_factories(
            cfg, opponent_factory=opponent_factory, curriculum=curriculum
        ),
        actor_critic=actor_critic,
        action_spec=action_spec,
        frames_per_batch=frames_per_batch,
        total_frames=cfg.collector.total_frames,
        clip_epsilon=cfg.agent.clip_epsilon,
        entropy_bonus=cfg.agent.get("entropy_bonus", True),
        entropy_coeff=cfg.agent.entropy_coeff,
        gamma=cfg.agent.gamma,
        lmbda=cfg.agent.lmbda,
        average_gae=cfg.agent.get("average_gae", True),
        gae_num_chunks=cfg.agent.get("gae_num_chunks", None),
        value_estimator=cfg.agent.get("value_estimator", "gae"),
        vtrace_rho_thresh=float(cfg.agent.get("vtrace_rho_thresh", 1.0)),
        vtrace_c_thresh=float(cfg.agent.get("vtrace_c_thresh", 1.0)),
        lr=cfg.agent.lr,
        num_epochs=cfg.agent.num_epochs,
        sub_batch_size=cfg.agent.sub_batch_size,
        max_grad_norm=cfg.agent.max_grad_norm,
        device=cfg.agent.get("device", "cpu"),
        collector_device=cfg.agent.get("collector_device"),
        use_parallel_env=cfg.env.parallel,
        mp_start_method=cfg.env.mp_start_method,
        serial_for_single=cfg.env.serial_for_single,
        target_kl=cfg.agent.get("target_kl"),
        target_kl_multiplier=cfg.agent.get("target_kl_multiplier", 1.5),
        compile_loss=cfg.agent.get("compile_loss", False),
        compile_policy=cfg.agent.get("compile_policy", False),
        lr_anneal=cfg.agent.get("lr_anneal", False),
        ent_anneal=cfg.agent.get("ent_anneal", False),
        ent_warm_frac=cfg.agent.get("ent_warm_frac", 0.5),
        callbacks=callbacks,
        run_config=run_config,
        evaluator=_build_evaluator(
            cfg,
            obs_spec,
            action_spec,
            checkpoint_dir,
            eval_opponent_factory,
            eval_sampler_spec,
        ),
        eval_interval=eval_interval,
        curriculum=curriculum,
        max_collector_restarts=_max_collector_restarts(cfg),
        rebuild_env_factories=_rebuild_env_factories(
            cfg, opponent_factory=opponent_factory, curriculum=curriculum
        ),
        pipe_timeout=_pipe_timeout(cfg),
        start_frames=start_frames,
        collector_type=cfg.collector.get("type", "sync"),
        async_options=_async_collector_options(cfg),
        train_state_path=_resolve_train_state_path(cfg),
        train_state_interval=int(cfg.train.get("train_state_interval", 0) or 0),
        resume_state=resume_state,
    )


def _warm_start(
    cfg: DictConfig, actor_critic: ActorCritic
) -> tuple[int, Mapping[str, Any] | None]:
    """
    Continue a previous run's policy, from either kind of saved state.

    ``train.resume_state`` names a training state
    (:class:`~src.training.callbacks.train_state_callback.TrainStateCallback`),
    which carries the optimizer alongside the weights and so continues training
    properly. ``train.init_checkpoint`` names a league snapshot, which carries
    weights only: Adam's moment estimates restart from zero and the first
    rollouts after such a warm start are noisier than the ones that preceded it.
    The self-play league starts empty either way, refilling from the loaded
    weights rather than from scratch.

    The inherited frame count comes from the loaded file rather than the config,
    so a continued run cannot silently disagree with what it continues about
    where it started. ``train.start_frames`` overrides it for legacy checkpoints
    that record no frame count.

    :param cfg: Hydra configuration with a ``train`` section.
    :param actor_critic: Freshly built network to load the weights into.
    :return: Frames the loaded state had already collected (``0`` without a warm
        start), and the optimizer state to resume from (``None`` to start cold).
    :raises ValueError: If the named file is missing, or records no frame count
        and ``train.start_frames`` does not supply one.
    """
    resume_configured = cfg.train.get("resume_state")
    configured = resume_configured or cfg.train.get("init_checkpoint")
    if not configured:
        return int(cfg.train.get("start_frames", 0) or 0), None

    source = Path(to_absolute_path(str(configured)))
    key = "train.resume_state" if resume_configured else "train.init_checkpoint"
    if not source.is_file():
        raise ValueError(f"{key} {source} does not exist.")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    actor_critic.load_state_dict(checkpoint_state_dict(payload), strict=True)

    override = cfg.train.get("start_frames")
    if override is not None:
        start_frames = int(override)
    elif isinstance(payload, Mapping) and payload.get("frames") is not None:
        start_frames = int(payload["frames"])
    else:
        raise ValueError(
            f"{key} {source} records no frame count, so the continued run cannot "
            f"number its snapshots after it. Set train.start_frames explicitly."
        )

    resume_state = payload.get("optimizer") if isinstance(payload, Mapping) else None
    if resume_configured and resume_state is None:
        raise ValueError(
            f"train.resume_state {source} carries no optimizer state. Point it at "
            f"a file written by TrainStateCallback, or use train.init_checkpoint "
            f"to warm-start from weights alone."
        )
    logger.info(
        "Warm start: loaded %s at %d frames (%s).",
        source,
        start_frames,
        "optimizer restored" if resume_state is not None else "weights only",
    )
    return start_frames, resume_state


def _resolve_train_state_path(cfg: DictConfig) -> Path | None:
    """
    Resolve where this run writes its resumable training state.

    Relative paths anchor to the Hydra run directory for the same reason
    :func:`_resolve_checkpoint_dir` does it: two runs sharing one path would
    overwrite each other's optimizer state.

    :param cfg: Hydra configuration with a ``train`` section.
    :return: Absolute destination, or None when the run writes no state.
    """
    configured = cfg.train.get("train_state_path")
    if not configured:
        return None
    path = Path(str(configured))
    if path.is_absolute():
        return path
    return Path(HydraConfig.get().runtime.output_dir) / path


def _max_collector_restarts(cfg: DictConfig) -> int:
    """
    Read the dead-worker restart budget from the collector config.

    :param cfg: Hydra configuration with a ``collector`` section.
    :return: Restarts allowed; ``0`` fails the run on the first worker death.
    """
    return int(cfg.collector.get("max_restarts", 0))


def _async_collector_options(cfg: DictConfig) -> AsyncCollectorOptions:
    """
    Read the asynchronous collectors' settings from the collector config.

    :param cfg: Hydra configuration with a ``collector`` section.
    :return: Settings for whichever asynchronous collector ``collector.type``
        selected; ignored entirely under ``sync``.
    """
    workers_per_batch = cfg.collector.get("workers_per_batch")
    return AsyncCollectorOptions(
        workers_per_batch=(
            None if workers_per_batch is None else int(workers_per_batch)
        ),
    )


def _pipe_timeout(cfg: DictConfig) -> float | None:
    """
    Read how long the worker pool may wait on itself before raising.

    :param cfg: Hydra configuration with a ``collector`` section.
    :return: Seconds, or None to keep torchrl's 10000-second default.
    """
    seconds = cfg.collector.get("pipe_timeout", None)
    return None if seconds is None else float(seconds)


def _rebuild_env_factories(
    cfg: DictConfig,
    opponent_factory: OpponentFactory | None = None,
    curriculum: Any = None,
) -> Callable[[int], list[Callable[[], EnvBase]]]:
    """
    Build the per-restart environment-factory builder handed to the trainer.

    Offsets every worker's seed by a full pool width per restart, so the
    replacement workers draw disjoint matchup/seat/opponent streams rather than
    replaying from the top the ones the dead pool already played.

    The live ``curriculum`` is passed through rather than rebuilt, so the
    replacement workers attach to the same shared-memory distribution channel
    and the same archetype index the buffer's level ids are addressed against.

    :param cfg: Hydra configuration, re-read on every restart.
    :param opponent_factory: Opponent factory to give the new workers.
    :param curriculum: Live curriculum driving the train split, if any.
    :return: Callable mapping a 1-based restart index to fresh factories.
    """
    num_workers = int(cfg.env.num_workers)

    def rebuild(restart_index: int) -> list[Callable[[], EnvBase]]:
        return make_env_factories(
            cfg,
            opponent_factory=opponent_factory,
            curriculum=curriculum,
            seed_offset=restart_index * num_workers,
        )

    return rebuild


def _resolve_checkpoint_dir(cfg: DictConfig) -> Path:
    """
    Resolve ``cfg.train.checkpoint_dir`` to the current run's snapshot directory.

    A relative path is anchored to the Hydra run directory rather than the
    working directory, so every run writes its league somewhere unique. This
    matters for correctness, not tidiness: a shared directory would let a new
    run load a *previous* run's snapshots as opponents. Hydra's ``job.chdir``
    defaults to False, so the working directory alone cannot be relied on to
    provide that isolation.

    The directory is created exclusively, so any collision that survives the
    unique run directory fails at startup instead of silently corrupting the
    run: two runs sharing one directory write identically-named snapshots over
    each other and draw each other's policies into their self-play leagues.

    :param cfg: Hydra configuration with a ``train`` section.
    :return: Absolute path to this run's snapshot directory.
    :raises RuntimeError: If the run-directory-relative path already exists.
    """
    configured = Path(cfg.train.get("checkpoint_dir", "checkpoints"))
    if configured.is_absolute():
        # An absolute path names a directory the operator chose deliberately,
        # so its contents and lifecycle are theirs to manage.
        return configured
    resolved = Path(HydraConfig.get().runtime.output_dir) / configured
    try:
        resolved.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise RuntimeError(
            f"Snapshot directory {resolved} already exists, so another run owns it. "
            f"Sharing it would overwrite that run's snapshots and mix its policies "
            f"into this league. Give this run its own hydra.run.dir, or point "
            f"train.checkpoint_dir at a fresh path."
        ) from None
    return resolved


def _resolve_checkpoint_registry(cfg: DictConfig) -> Path:
    """Resolve the append-only completed-checkpoint CSV from the repository root."""
    configured = Path(cfg.paths.checkpoint_registry)
    if configured.is_absolute():
        return configured.resolve()
    return (Path(__file__).parents[1] / configured).resolve()


def _build_evaluator(
    cfg: DictConfig,
    obs_spec: Composite,
    action_spec: Categorical,
    checkpoint_dir: Path | None = None,
    opponent_factory: OpponentFactory | None = None,
    sampler_spec: dict[str, Any] | None = None,
) -> Evaluator | MultiEvaluator | None:
    """
    Build the fixed-opponent evaluator(s) selected by ``cfg.train``, gated on
    ``eval_interval``.

    The evaluation environment is given its opponent explicitly, from
    ``cfg.train.eval_opponents`` (or the singular ``eval_opponent``), rather
    than inheriting whichever opponent :class:`~src.env.tcg_env.TCGEnv` happens
    to default to. Under self-play the collected ``win_rate`` is pinned near
    0.5 by construction, so this fixed reference is what makes the run's
    progress readable.

    With several opponents configured the result is a
    :class:`~src.training.multi_evaluator.MultiEvaluator`, whose metrics are
    namespaced per opponent.

    :param cfg: Hydra configuration with a ``train`` section.
    :param obs_spec: Environment observation spec, forwarded to the opponent
        factory so a snapshot-backed reference can rebuild its network.
    :param action_spec: Environment action spec, same purpose.
    :param checkpoint_dir: Snapshot directory, forwarded for
        ``first_snapshot``.
    :param opponent_factory: Explicit eval opponent, overriding
        ``cfg.train.eval_opponent(s)`` and collapsing the result to a single
        evaluator. Used by a best-response run to score the learner against the
        same frozen agent it trains against.
    :param sampler_spec: Precomputed eval-split sampler spec, shared with
        ``CrossPlayCallback`` when both are active; built fresh when ``None``.
    :return: An evaluator, or None when ``eval_interval`` disables evaluation.
    """
    if int(cfg.train.get("eval_interval", 0)) <= 0:
        return None
    if opponent_factory is not None:
        return build_evaluator(
            cfg, obs_spec, action_spec, opponent_factory, sampler_spec
        )
    evaluators = [
        build_evaluator(
            cfg,
            obs_spec,
            action_spec,
            sampler_spec=sampler_spec,
            opponent=name,
            checkpoint_dir=checkpoint_dir,
        )
        for name in _eval_opponent_names(cfg)
    ]
    return evaluators[0] if len(evaluators) == 1 else MultiEvaluator(evaluators)


def _eval_opponent_names(cfg: DictConfig) -> list[str]:
    """
    Resolve which reference opponents the evaluator scores against.

    ``train.eval_opponents`` (plural, a list) takes precedence when set, so a
    run can be scored against several references at once, typically a frozen
    snapshot, which keeps discriminating late in training, alongside random,
    which stays comparable across runs. Falls back to the singular
    ``train.eval_opponent``.

    :param cfg: Hydra configuration with a ``train`` section.
    :return: Opponent names, in the order they should be evaluated.
    """
    configured = cfg.train.get("eval_opponents")
    if configured:
        return [str(name) for name in configured]
    return [str(cfg.train.get("eval_opponent", "random"))]
