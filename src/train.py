from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from src.hydra_resolvers import register_resolvers
from src.trainer_builder import (
    build_callbacks,
    build_trainer,
    resolve_run_config,
    seed_everything,
)

load_dotenv(Path(__file__).parents[1] / ".env", override=False)
# Before Hydra composes anything: the run directory interpolates ${run_uid:}.
register_resolvers()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Local entry point.

    :param cfg: Hydra configuration object, composed from conf/config.yaml.
    """
    # Resolved, so the callbacks block shows the W&B values it interpolates
    # from cfg.wandb rather than the raw ${wandb.*} references.
    print(OmegaConf.to_yaml(cfg, resolve=True))
    seed_everything(cfg)

    callbacks = build_callbacks(cfg)
    trainer = build_trainer(cfg, callbacks, resolve_run_config(cfg))

    stats = trainer.train()
    print(
        f"done: {stats['frames']} frames, {stats['episodes']} episodes, "
        f"win-rate {stats['win_rate']:.3f}, draw-rate {stats['draw_rate']:.3f}, "
        f"{stats['fps']:.0f} fps"
    )


if __name__ == "__main__":
    main()
