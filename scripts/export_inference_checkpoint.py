"""
Export the checkpoint + config pair main.py loads for inference.

Writes ``checkpoint/model.pt`` (an actor-critic state_dict) and
``checkpoint/model_config.yaml`` (the resolved ``model`` config plus the
``max_options``/``encoder`` needed to rebuild the observation/action specs) --
the two files ``main.py`` loads at inference time via
:func:`src.policies.inference.load_inference_agent`. This is not the Kaggle
submission packaging step (no bundling/zipping/CLI upload here); it just
produces the artifact that checkpoint-loading code needs to have something to
load.

The weights must come from ``--source-checkpoint`` (e.g. a
``SnapshotCallback`` snapshot or any other :func:`~src.policies.
greedy_policy_opponent.save_actor_critic` file). The matching resolved Hydra
run config is loaded from ``--source-config`` or discovered at
``.hydra/config.yaml`` above the checkpoint. Reading the training run's config
is essential: composing today's defaults can produce a shape-compatible but
semantically different network (for example, a different activation).

Run from the repository root::

    uv run python scripts/export_inference_checkpoint.py --source-checkpoint outputs/.../snapshot_000000004096.pt
    uv run python scripts/export_inference_checkpoint.py \
        --source-checkpoint /scratch/run/checkpoints/snapshot_000000004096.pt \
        --source-config /scratch/run/.hydra/config.yaml
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from omegaconf import OmegaConf

from src.policies.greedy_policy_opponent import (
    checkpoint_state_dict,
    save_actor_critic,
)
from src.policies.inference import build_inference_specs
from src.policies.ppo_actor import build_actor_critic

DEFAULT_OUTPUT_DIR = REPO_ROOT / "checkpoint"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        required=True,
        help="Existing trained actor-critic state_dict to export.",
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        default=None,
        help="Resolved Hydra config from the training run. If omitted, search parent directories "
             "of --source-checkpoint for .hydra/config.yaml.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write model.pt / model_config.yaml to.")
    return parser.parse_args()


def resolve_source_config(
    source_checkpoint: Path,
    source_config: Path | None,
) -> Path:
    """
    Resolve the exact Hydra config that produced a training checkpoint.

    :param source_checkpoint: Training checkpoint whose parent run is searched.
    :param source_config: Explicit config path, or None to auto-discover it.
    :return: Existing resolved config path.
    :raises FileNotFoundError: If the explicit config or an auto-discovered
        ``.hydra/config.yaml`` cannot be found.
    """
    if source_config is not None:
        if not source_config.is_file():
            raise FileNotFoundError(f"Training config does not exist: {source_config}")
        return source_config

    checkpoint = source_checkpoint.resolve()
    for directory in checkpoint.parents:
        candidate = directory / ".hydra" / "config.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find .hydra/config.yaml above "
        f"{source_checkpoint}; pass --source-config explicitly."
    )


def export_inference_checkpoint(
    source_checkpoint: Path,
    source_config: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    Validate trained weights against their run config and export the pair.

    :param source_checkpoint: Trained actor-critic state_dict.
    :param source_config: Resolved Hydra config used for that training run.
    :param output_dir: Destination directory.
    :return: Paths to ``model.pt`` and ``model_config.yaml``.
    """
    cfg = OmegaConf.load(source_config)
    max_options = int(cfg.env.max_options)
    encoder_name = str(cfg.env.get("encoder", "structured"))
    obs_spec, _, action_spec = build_inference_specs(max_options, encoder_name)
    actor_critic = build_actor_critic(cfg, obs_spec, action_spec)

    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    actor_critic.load_state_dict(checkpoint_state_dict(payload))

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_actor_critic(actor_critic, output_dir / "model.pt")
    model_config = OmegaConf.create({
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "max_options": max_options,
        "encoder": encoder_name,
    })
    model_config_path = output_dir / "model_config.yaml"
    OmegaConf.save(model_config, model_config_path)
    return checkpoint_path, model_config_path


def main() -> None:
    """
    Load the training run config, validate its actor-critic checkpoint, and
    write the inference checkpoint + config pair.
    """
    args = parse_args()
    source_config = resolve_source_config(args.source_checkpoint, args.source_config)
    checkpoint_path, model_config_path = export_inference_checkpoint(
        args.source_checkpoint,
        source_config,
        args.output_dir,
    )

    print(f"Wrote {checkpoint_path} and {model_config_path}")


if __name__ == "__main__":
    main()
