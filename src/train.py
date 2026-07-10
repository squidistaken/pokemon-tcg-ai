import random

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.policies.ppo_actor import build_actor_critic
from src.policies.random_masked_policy import RandomMaskedPolicy
from src.training import PPOTrainer, Trainer, make_env_factories


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Local entry point.

    Selects the trainer from ``cfg.agent.name``: ``ppo`` runs the Phase-1
    shared-trunk actor-critic with clipped PPO and invalid-action masking;
    anything else runs the pure random-collection baseline.

    :param cfg: Hydra configuration object, composed from conf/config.yaml.
    """
    print(OmegaConf.to_yaml(cfg))
    if cfg.set_seed:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    if cfg.agent.name == "ppo":
        trainer = _build_ppo_trainer(cfg)
    else:
        trainer = Trainer(
            env_factories=make_env_factories(cfg),
            policy=RandomMaskedPolicy(),
            frames_per_batch=cfg.collector.frames_per_batch,
            total_frames=cfg.collector.total_frames,
            use_parallel_env=cfg.env.parallel,
            mp_start_method=cfg.env.mp_start_method,
            serial_for_single=cfg.env.serial_for_single,
        )

    stats = trainer.train()
    print(
        f"done: {stats['frames']} frames, {stats['episodes']} episodes, "
        f"win-rate {stats['win_rate']:.3f}, draw-rate {stats['draw_rate']:.3f}, "
        f"{stats['fps']:.0f} fps"
    )


def _build_ppo_trainer(cfg: DictConfig) -> PPOTrainer:
    """
    Assemble the PPO trainer, actor-critic and environment from the config.

    Derives the observation/action specs from a throwaway environment instance
    (built from the same factory as training), so the network is sized against
    the exact env contract.

    :param cfg: Hydra configuration with ``agent`` (PPO hyperparameters),
        ``model`` and ``env`` sections.
    :return: A configured :class:`~src.training.ppo_trainer.PPOTrainer`.
    """
    env_factories = make_env_factories(cfg)
    probe_env = env_factories[0]()
    try:
        obs_spec = probe_env.observation_spec
        action_spec = probe_env.action_spec
    finally:
        probe_env.close()

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
        use_parallel_env=cfg.env.parallel,
        mp_start_method=cfg.env.mp_start_method,
        serial_for_single=cfg.env.serial_for_single,
    )


if __name__ == "__main__":
    main()
