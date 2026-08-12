"""
Every shipped top-level run config must compose into a buildable model.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

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


def test_fixed_deck_finetune_config_resolves_requested_experiment() -> None:
    """The production preset pins only Player 1 and preserves baseline shape."""
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(config_name="ppo_fixed_deck_finetune")

    assert cfg.seed == 42
    assert cfg.collector.total_frames == 30_015_488
    assert cfg.collector.max_restarts == 100
    assert cfg.agent.frames_per_batch == 16_384
    assert 85_688_320 + cfg.collector.total_frames == 115_703_808
    assert cfg.train.opponent_pool_mode == "frozen"
    assert cfg.train.opponent_sampling == "uniform"
    assert cfg.train.pool_size == 10
    assert cfg.train.snapshot_interval == 500_000
    assert cfg.train.eval_interval == 500_000
    assert cfg.train.eval_episodes == 50
    assert list(cfg.train.eval_opponents) == ["checkpoint_pool", "random"]
    assert cfg.env.agent_deck == cfg.env.eval_agent_deck
    assert cfg.env.agent_deck_field_prob == 0.0
    assert cfg.env.agent_deck_single_field_draw is True
    assert cfg.env.deck_matchup == "independent"
    assert cfg.env.eval_deck_matchup == "independent"
    assert cfg.env.deck_weighting == "observation"
    assert cfg.env.deck_holdout_frac == 0.0
    assert cfg.env.curriculum.enabled is False
    assert cfg.model.embed_dim == 256
    assert cfg.model.backbone.token_groups == ["pokemon"]
    assert cfg.model.backbone.option_tokens is True
    assert cfg.model.adapter.card_embed_dim == 16
    assert cfg.model.adapter.entity_dim == 128
    assert cfg.model.adapter.option_target_state is True
    assert cfg.model.adapter.pokemon_seat_split is True
    assert cfg.train.init_checkpoint.endswith("snapshot_000085688320.pt")
    assert OmegaConf.is_config(cfg)


@pytest.mark.parametrize("config_name", ["config", "ppo_selfplay_multideck"])
def test_existing_configs_resolve_inert_selfplay_default(config_name: str) -> None:
    """Existing run presets opt into no new external-checkpoint behavior."""
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(config_name=config_name)

    assert cfg.train.opponent_pool_mode == "selfplay"
    assert cfg.train.opponent_checkpoint_dir is None
