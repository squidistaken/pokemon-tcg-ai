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

The config is composed the normal Hydra way (``conf/config.yaml`` + any
overrides), so it always matches whatever architecture is current. The
weights come from ``--source-checkpoint`` if given (e.g. a
``SnapshotCallback`` snapshot or any other :func:`~src.policies.
greedy_policy_opponent.save_actor_critic` file); without it, a freshly
initialized (untrained) actor-critic is exported instead, as a placeholder
that exercises the same checkpoint/config format ahead of a real run.

Run from the repository root::

    uv run python scripts/export_inference_checkpoint.py
    uv run python scripts/export_inference_checkpoint.py --source-checkpoint outputs/.../snapshot_000000004096.pt
    uv run python scripts/export_inference_checkpoint.py --overrides model/backbone=mlp model/head=linear
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.policies.greedy_policy_opponent import save_actor_critic
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
        default=None,
        help="Existing actor-critic state_dict to export (must match the composed architecture). "
             "If omitted, a freshly initialized (untrained) actor-critic is exported as a placeholder.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Hydra overrides used to compose the architecture config, e.g. model/backbone=mlp.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write model.pt / model_config.yaml to.")
    return parser.parse_args()


def main() -> None:
    """
    Compose the model config, build (and optionally load) the actor-critic,
    and write the checkpoint + config pair.
    """
    args = parse_args()
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "conf")):
        cfg = compose(config_name="config", overrides=args.overrides)

    max_options = int(cfg.env.max_options)
    encoder_name = str(cfg.env.get("encoder", "structured"))
    obs_spec, _, action_spec = build_inference_specs(max_options, encoder_name)
    actor_critic = build_actor_critic(cfg, obs_spec, action_spec)

    if args.source_checkpoint is not None:
        state_dict = torch.load(args.source_checkpoint, map_location="cpu", weights_only=True)
        actor_critic.load_state_dict(state_dict)
    else:
        print("No --source-checkpoint given; exporting freshly initialized (untrained) weights as a placeholder.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_actor_critic(actor_critic, args.output_dir / "model.pt")

    model_config = OmegaConf.create({
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "max_options": max_options,
        "encoder": encoder_name,
    })
    model_config_path = args.output_dir / "model_config.yaml"
    OmegaConf.save(model_config, model_config_path)

    print(f"Wrote {checkpoint_path} and {model_config_path}")


if __name__ == "__main__":
    main()
