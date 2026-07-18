import random
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torchrl.data import Categorical

from src.policies.ppo_actor import build_actor_critic
from src.policies.random_masked_policy import RandomMaskedPolicy
from src.training import PPOTrainer, Trainer, TrainingCallback, make_env_factories

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
    env_factories = make_env_factories(cfg)
    probe_env = env_factories[0]()
    try:
        obs_spec = probe_env.observation_spec
        action_spec = probe_env.action_spec
    finally:
        probe_env.close()
    assert isinstance(action_spec, Categorical), "env action spec must be Categorical"

    actor_critic = build_actor_critic(cfg, obs_spec, action_spec)
    frames_per_batch = int(cfg.agent.get("frames_per_batch", cfg.collector.frames_per_batch))
    return PPOTrainer(
        env_factories=env_factories,
        actor_critic=actor_critic,
        action_spec=action_spec,
        frames_per_batch=frames_per_batch,
        total_frames=cfg.collector.total_frames,
        clip_epsilon=cfg.agent.clip_epsilon,
        entropy_coeff=cfg.agent.entropy_coeff,
        gamma=cfg.agent.gamma,
        lmbda=cfg.agent.lmbda,
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
    )


if __name__ == "__main__":
    main()
