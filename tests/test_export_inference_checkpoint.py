from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from scripts.export_inference_checkpoint import (
    export_inference_checkpoint,
    resolve_source_config,
)
from src.policies.greedy_policy_opponent import save_actor_critic
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
        actor_critic, run_dir / "checkpoints" / "snapshot.pt"
    )

    assert resolve_source_config(source_path, None) == config_path
    checkpoint_path, model_config_path = export_inference_checkpoint(
        source_path, config_path, tmp_path / "export"
    )

    exported_config = OmegaConf.load(model_config_path)
    assert exported_config.model.backbone.activation == "relu"
    source_state = torch.load(source_path, map_location="cpu", weights_only=True)
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
