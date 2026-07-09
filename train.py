
import hydra
from omegaconf import DictConfig, OmegaConf
import random


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Local entry point.

    :param cfg: Hydra configuration object, composed from conf/config.yaml.
    """
    print(OmegaConf.to_yaml(cfg))
    random.seed(cfg.seed)


if __name__ == "__main__":
    main()
