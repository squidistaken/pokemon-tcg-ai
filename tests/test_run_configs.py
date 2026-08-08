"""
Every shipped top-level run config must compose into a buildable model.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.policies.inference import build_inference_specs
from src.policies.ppo_actor import build_actor_critic

CONF_DIR = Path(__file__).parents[1] / "conf"
ROOT_CONFIGS = sorted(path.stem for path in CONF_DIR.glob("*.yaml"))


def test_root_configs_were_discovered() -> None:
    """Guard against the glob silently matching nothing."""
    assert "ppo_transformer" in ROOT_CONFIGS
    assert len(ROOT_CONFIGS) >= 4


@pytest.mark.parametrize("config_name", ROOT_CONFIGS)
def test_root_config_builds_its_model(config_name: str) -> None:
    """
    Compose one shipped root config and build the actor-critic it describes.

    :param config_name: Stem of a ``conf/*.yaml`` root config.
    """
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(config_name=config_name)

    if cfg.get("model") is None or cfg.model.get("backbone") is None:
        pytest.skip(f"{config_name} configures no model")

    obs_spec, _encoder, action_spec = build_inference_specs(
        int(cfg.env.max_options), str(cfg.env.get("encoder", "structured"))
    )
    build_actor_critic(cfg, obs_spec, action_spec)
