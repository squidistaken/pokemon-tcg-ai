import random
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torchrl.data import Categorical, Composite

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
from src.training.env_factory import OpponentFactory, _build_sampler_spec

load_dotenv(Path(__file__).parents[1] / ".env", override=False)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Local entry point.

    Selects the trainer from ``cfg.agent.name``: ``ppo`` runs the Phase-1
    shared-trunk actor-critic with clipped PPO and invalid-action masking;
    anything else runs the pure random-collection baseline.

    :param cfg: Hydra configuration object, composed from conf/config.yaml.
    """
    # Resolved, so the callbacks block shows the W&B values it interpolates
    # from cfg.wandb rather than the raw ${wandb.*} references.
    print(OmegaConf.to_yaml(cfg, resolve=True))
    if cfg.set_seed:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    callbacks = _build_callbacks(cfg)
    run_config = _run_config(cfg)

    if cfg.agent.name == "ppo":
        trainer = _build_ppo_trainer(cfg, callbacks, run_config)
    else:
        trainer = Trainer(
            env_factories=make_env_factories(cfg),
            policy=RandomMaskedPolicy(),
            frames_per_batch=cfg.collector.frames_per_batch,
            total_frames=cfg.collector.total_frames,
            use_parallel_env=cfg.env.parallel,
            mp_start_method=cfg.env.mp_start_method,
            serial_for_single=cfg.env.serial_for_single,
            callbacks=callbacks,
            run_config=run_config,
        )

    stats = trainer.train()
    print(
        f"done: {stats['frames']} frames, {stats['episodes']} episodes, "
        f"win-rate {stats['win_rate']:.3f}, draw-rate {stats['draw_rate']:.3f}, "
        f"{stats['fps']:.0f} fps"
    )


def _build_callbacks(cfg: DictConfig) -> list[TrainingCallback]:
    """
    Instantiate the metric backends selected by the ``callbacks`` config group.

    :param cfg: Hydra configuration with a top-level ``callbacks`` list.
    :return: Instantiated callbacks; empty for ``callbacks=none``.
    """
    return [hydra.utils.instantiate(callback) for callback in cfg.callbacks]


def _run_config(cfg: DictConfig) -> dict[str, Any]:
    """
    Resolve the Hydra config into a plain dict for the callbacks.

    :param cfg: Hydra configuration object.
    :return: The fully resolved config as nested plain Python types.
    """
    return cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))


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

    checkpoint_dir = _resolve_checkpoint_dir(cfg)
    is_best_response = bool(cfg.train.get("best_response", False))
    if is_best_response:
        # Exploitability run: the learner both trains against and is scored
        # against one frozen agent, and its eval win-rate is that agent's
        # exploitability. No self-play league, so cross-play against the
        # learner's own history is meaningless.
        opponent_factory = build_best_response_opponent_factory(cfg, obs_spec, action_spec)
        eval_opponent_factory = opponent_factory
    else:
        opponent_factory = build_opponent_factory(
            cfg, obs_spec, action_spec, checkpoint_dir
        )
        eval_opponent_factory = None

    snapshot_interval = int(cfg.train.get("snapshot_interval", 0))
    cross_play_enabled = (
        not is_best_response and opponent_factory is not None
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
        lr=cfg.agent.lr,
        num_epochs=cfg.agent.num_epochs,
        sub_batch_size=cfg.agent.sub_batch_size,
        max_grad_norm=cfg.agent.max_grad_norm,
        device=cfg.agent.get("device", "cpu"),
        use_parallel_env=cfg.env.parallel,
        mp_start_method=cfg.env.mp_start_method,
        serial_for_single=cfg.env.serial_for_single,
        target_kl=cfg.agent.get("target_kl"),
        target_kl_multiplier=cfg.agent.get("target_kl_multiplier", 1.5),
        use_amp=cfg.agent.get("use_amp", False),
        compile_loss=cfg.agent.get("compile_loss", False),
        compile_policy=cfg.agent.get("compile_policy", False),
        lr_anneal=cfg.agent.get("lr_anneal", False),
        ent_anneal=cfg.agent.get("ent_anneal", False),
        ent_warm_frac=cfg.agent.get("ent_warm_frac", 0.5),
        reward_scaling=cfg.agent.get("reward_scaling", 1.0),
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
    )


def _resolve_checkpoint_dir(cfg: DictConfig) -> Path:
    """
    Resolve ``cfg.train.checkpoint_dir`` to the current run's snapshot directory.

    A relative path is anchored to the Hydra run directory rather than the
    working directory, so every run writes its league somewhere unique. This
    matters for correctness, not tidiness: a shared directory would let a new
    run load a *previous* run's snapshots as opponents. Hydra's ``job.chdir``
    defaults to False, so the working directory alone cannot be relied on to
    provide that isolation.

    :param cfg: Hydra configuration with a ``train`` section.
    :return: Absolute path to this run's snapshot directory.
    """
    configured = Path(cfg.train.get("checkpoint_dir", "checkpoints"))
    if configured.is_absolute():
        return configured
    return Path(HydraConfig.get().runtime.output_dir) / configured


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
    run can be scored against several references at once — typically a frozen
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


if __name__ == "__main__":
    main()
