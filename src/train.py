import random

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.policies.random_masked_policy import RandomMaskedPolicy
from src.training import Trainer, make_env_factories


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Local entry point.

    :param cfg: Hydra configuration object, composed from conf/config.yaml.
    """
    print(OmegaConf.to_yaml(cfg))
    if cfg.set_seed:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
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


if __name__ == "__main__":
    main()
