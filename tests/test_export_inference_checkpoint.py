from pathlib import Path
from typing import cast

import pytest
import torch
from omegaconf import OmegaConf

from scripts.export_inference_checkpoint import (
    export_inference_checkpoint,
    resolve_source_config,
)
from src.models.structured_obs_adapter import StructuredObsAdapter
from src.policies.greedy_policy_opponent import (
    checkpoint_state_dict,
    load_actor_critic,
    save_actor_critic,
)
from src.policies.inference import build_inference_specs
from src.policies.ppo_actor import build_actor_critic
from tests.conftest import MAX_OPTIONS


def _training_config(structured_model_cfg):
    """Build a resolved run config with a non-default semantic setting."""
    cfg = OmegaConf.create(
        {
            "env": {"max_options": MAX_OPTIONS, "encoder": "structured"},
            "model": OmegaConf.to_container(structured_model_cfg.model, resolve=True),
        }
    )
    cfg.model.backbone.activation = "relu"
    return cfg


def _adapter_width(actor_critic) -> int:
    """
    Flattened observation width the trunk consumes.

    :param actor_critic: Built actor-critic whose backbone carries an adapter.
    :return: The adapter's ``out_features``.
    """
    return int(cast(StructuredObsAdapter, actor_critic.backbone.adapter).out_features)


def test_export_uses_the_training_runs_config(tmp_path: Path, structured_model_cfg) -> None:
    """Export preserves semantic config that state-dict shape checks cannot detect."""
    cfg = _training_config(structured_model_cfg)
    run_dir = tmp_path / "run"
    config_path = run_dir / ".hydra" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    OmegaConf.save(cfg, config_path)

    obs_spec, _, action_spec = build_inference_specs(MAX_OPTIONS)
    actor_critic = build_actor_critic(cfg, obs_spec, action_spec)
    source_path = save_actor_critic(
        actor_critic,
        run_dir / "checkpoints" / "snapshot.pt",
        config=OmegaConf.to_container(cfg, resolve=True),
        frames=42,
    )

    assert resolve_source_config(source_path, None) == config_path
    checkpoint_path, model_config_path = export_inference_checkpoint(
        source_path, config_path, tmp_path / "export"
    )

    exported_config = OmegaConf.load(model_config_path)
    assert exported_config.model.backbone.activation == "relu"
    source_payload = torch.load(source_path, map_location="cpu", weights_only=True)
    source_state = checkpoint_state_dict(source_payload)
    exported_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert source_state.keys() == exported_state.keys()
    assert all(
        torch.equal(source_state[key], exported_state[key]) for key in source_state
    )


def test_resolve_source_config_requires_training_provenance(tmp_path: Path) -> None:
    """A bare checkpoint cannot be paired silently with current defaults."""
    checkpoint_path = tmp_path / "snapshot.pt"
    checkpoint_path.touch()

    with pytest.raises(FileNotFoundError, match="pass --source-config"):
        resolve_source_config(checkpoint_path, None)


def test_load_actor_critic_rebuilds_from_the_checkpoints_own_config(
    tmp_path: Path, structured_model_cfg
) -> None:
    """
    A frozen reference must keep loading after the current run's architecture
    has moved on, which is the only way "am I better than the agent we already
    submitted" stays answerable across an architecture change.

    ``pokemon_seat_split`` is the concrete case: it widens the ``pokemon``
    group, so a snapshot saved without it has a narrower first projection than
    the current config builds. Rebuilding from the passed config instead of the
    checkpoint's own would fail here on a size mismatch.
    """
    old_cfg = _training_config(structured_model_cfg)
    old_cfg.model.adapter = {"pokemon_seat_split": False}
    obs_spec, _, action_spec = build_inference_specs(MAX_OPTIONS)
    old_agent = build_actor_critic(old_cfg, obs_spec, action_spec)
    checkpoint = save_actor_critic(
        old_agent,
        tmp_path / "snapshot.pt",
        config=OmegaConf.to_container(old_cfg, resolve=True),
    )

    current_cfg = _training_config(structured_model_cfg)
    current_cfg.model.adapter = {"pokemon_seat_split": True}
    current_agent = build_actor_critic(current_cfg, obs_spec, action_spec)
    assert _adapter_width(current_agent) != _adapter_width(old_agent)

    loaded = load_actor_critic(checkpoint, current_cfg, obs_spec, action_spec)
    for restored, original in zip(
        loaded.state_dict().values(), old_agent.state_dict().values(), strict=True
    ):
        assert torch.equal(restored, original)


def test_load_actor_critic_falls_back_to_config_for_a_legacy_checkpoint(
    tmp_path: Path, structured_model_cfg
) -> None:
    """
    A bare-state-dict checkpoint embeds no config, so the passed one is still
    the only thing available to rebuild it from.
    """
    cfg = _training_config(structured_model_cfg)
    obs_spec, _, action_spec = build_inference_specs(MAX_OPTIONS)
    agent = build_actor_critic(cfg, obs_spec, action_spec)
    checkpoint = save_actor_critic(agent, tmp_path / "legacy.pt")

    loaded = load_actor_critic(checkpoint, cfg, obs_spec, action_spec)
    for restored, original in zip(
        loaded.state_dict().values(), agent.state_dict().values(), strict=True
    ):
        assert torch.equal(restored, original)


def test_load_actor_critic_survives_an_observation_spec_widening(
    tmp_path: Path, structured_model_cfg
) -> None:
    """
    ``option_target_state`` widens an option row, and that width comes from the
    *observation spec* rather than from anything a checkpoint records. So
    unlike a pure config change it cannot be recovered by rebuilding from the
    checkpoint's own config -- the flag has to stay opt-in, or every snapshot
    predating the block becomes unloadable the moment the encoder gains it.

    That is not hypothetical: it would fail at the first evaluation round of
    any run scoring itself against an earlier agent, and it would break the
    deployed Kaggle model. This pins the compatibility.
    """
    obs_spec, _, action_spec = build_inference_specs(MAX_OPTIONS)

    old_cfg = _training_config(structured_model_cfg)
    old_cfg.model.adapter = {"option_target_state": False}
    old_agent = build_actor_critic(old_cfg, obs_spec, action_spec)
    checkpoint = save_actor_critic(
        old_agent,
        tmp_path / "pre_target_state.pt",
        config=OmegaConf.to_container(old_cfg, resolve=True),
    )

    current_cfg = _training_config(structured_model_cfg)
    current_cfg.model.adapter = {"option_target_state": True}
    assert _adapter_width(
        build_actor_critic(current_cfg, obs_spec, action_spec)
    ) == _adapter_width(old_agent), "target_state must not change the pooled width"

    loaded = load_actor_critic(checkpoint, current_cfg, obs_spec, action_spec)
    for restored, original in zip(
        loaded.state_dict().values(), old_agent.state_dict().values(), strict=True
    ):
        assert torch.equal(restored, original)
