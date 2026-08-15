"""
Clone expert decisions into the training architecture.

The policy head is fit with masked cross-entropy against the option the expert
submitted, and the value head with MSE against the episode's real result from
the acting seat. That value target is the thing self-play never supplied: a
critic trained on actual games rather than on a league's opinion of itself.

Architecture and checkpoint layout are copied from a reference snapshot, so the
result loads in the existing eval and submission paths without changes. The
card-effect ablation is a pure model-config switch, because the dataset stores
card ids and the effect columns are looked up inside the adapter.
"""
import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as functional
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.data import Binary, Categorical, Composite, Unbounded

from src.env.observation.structured_observation_encoder import (
    StructuredObservationEncoder,
)
from src.policies.ppo_actor import build_actor_critic

MAX_OPTIONS = 128
REFERENCE = (
    "outputs/deck-pinned-150m-local/tf-ptr-pinned-selfplay-10m-s42/"
    "checkpoints/snapshot_000175702016.pt"
)


def build_config(card_effect_features: bool):
    """
    Load the reference architecture and set the ablation switch.

    :param card_effect_features: Whether the adapter appends effect columns.
    :return: Hydra-style config carrying a ``model`` section.
    """
    reference = torch.load(REFERENCE, map_location="cpu", weights_only=False)
    config = OmegaConf.create(reference["config"])
    OmegaConf.set_struct(config, False)
    config.model.adapter.card_effect_features = bool(card_effect_features)
    return config


def build_network(config):
    """
    Instantiate the actor-critic against the encoder's observation spec.

    :param config: Config carrying the ``model`` section.
    :return: Untrained actor-critic.
    """
    encoder = StructuredObservationEncoder(max_options=MAX_OPTIONS)
    n_actions = MAX_OPTIONS + 1
    obs_spec = Composite(
        observation=encoder.spec(),
        action_mask=Binary(n=n_actions, dtype=torch.bool),
        level_id=Unbounded(shape=(1,), dtype=torch.int64),
        opponent_is_anchor=Binary(1, dtype=torch.bool),
    )
    return build_actor_critic(
        config, obs_spec, Categorical(n_actions, dtype=torch.int64)
    )


def masked_logits(logits: torch.Tensor, n_options: torch.Tensor) -> torch.Tensor:
    """
    Blank out option slots the state does not offer.

    :param logits: Raw head output, ``(batch, MAX_OPTIONS + 1)``.
    :param n_options: Legal option count per row.
    :return: Logits with illegal slots driven to ``-inf``.
    """
    positions = torch.arange(logits.shape[-1], device=logits.device)
    legal = positions.unsqueeze(0) < n_options.unsqueeze(-1)
    return logits.masked_fill(~legal, float("-inf"))


def evaluate(network, dataset, index, device, batch_size: int, value_coef: float):
    """
    Score the network on a held-out split.

    :param network: Actor-critic under test.
    :param dataset: Full sample store.
    :param index: Row indices forming the held-out split.
    :param device: Torch device.
    :param batch_size: Rows per forward pass.
    :param value_coef: Weight applied to the value term in the total.
    :return: Dict with accuracy, top-5, policy loss, value loss.
    """
    network.eval()
    totals = {"n": 0, "correct": 0, "top5": 0, "policy": 0.0, "value": 0.0}
    with torch.inference_mode():
        for start in range(0, index.numel(), batch_size):
            batch = dataset[index[start : start + batch_size]].to(device)
            rows = batch.shape[0]
            forward = TensorDict(
                {"observation": batch["observation"]}, batch_size=batch.batch_size
            )
            network(forward)
            logits = masked_logits(
                forward.get("logits").reshape(rows, -1), batch["n_options"]
            )
            value = forward.get("state_value").reshape(rows)
            action = batch["action"]
            totals["policy"] += float(
                functional.cross_entropy(logits, action, reduction="sum")
            )
            totals["value"] += float(
                functional.mse_loss(value, batch["outcome"], reduction="sum")
            )
            totals["correct"] += int((logits.argmax(dim=-1) == action).sum())
            top5 = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
            totals["top5"] += int((top5 == action.unsqueeze(-1)).any(dim=-1).sum())
            totals["n"] += rows
    count = max(totals["n"], 1)
    policy = totals["policy"] / count
    value_loss = totals["value"] / count
    return {
        "val/accuracy": totals["correct"] / count,
        "val/top5": totals["top5"] / count,
        "val/policy_loss": policy,
        "val/value_loss": value_loss,
        "val/total": policy + value_coef * value_loss,
    }


def train(args) -> None:
    """
    Fit one behaviour-cloning arm and write a loadable checkpoint.

    :param args: Parsed command-line arguments.
    """
    import wandb

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=False)
    total = dataset.shape[0]
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(total, generator=generator)
    n_val = max(1, int(total * args.val_fraction))
    # The dataset is ~9 GB, so the split is kept as index tensors and the rows
    # are gathered per batch. Materializing two sub-TensorDicts would double
    # the resident set and exhaust memory on this box.
    val_index = order[:n_val]
    train_index = order[n_val:]
    print(f"train {train_index.numel()}  val {val_index.numel()}", flush=True)

    config = build_config(args.card_effect_features)
    network = build_network(config).to(device)
    parameters = sum(p.numel() for p in network.parameters())
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    wandb.init(
        project="pokemon-tcg-ai",
        group=args.wandb_group,
        name=args.wandb_name,
        config={
            "arm": args.wandb_name,
            "card_effect_features": bool(args.card_effect_features),
            "dataset": str(args.dataset),
            "samples": total,
            "train_samples": int(train_index.numel()),
            "val_samples": int(val_index.numel()),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "value_coef": args.value_coef,
            "parameters": parameters,
            "seed": args.seed,
        },
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best = math.inf
    step = 0
    for epoch in range(args.epochs):
        network.train()
        shuffle = train_index[torch.randperm(train_index.numel(), generator=generator)]
        for start in range(0, shuffle.numel(), args.batch_size):
            batch = dataset[shuffle[start : start + args.batch_size]].to(device)
            rows = batch.shape[0]
            forward = TensorDict(
                {"observation": batch["observation"]}, batch_size=batch.batch_size
            )
            network(forward)
            logits = masked_logits(
                forward.get("logits").reshape(rows, -1), batch["n_options"]
            )
            value = forward.get("state_value").reshape(rows)
            policy_loss = functional.cross_entropy(logits, batch["action"])
            value_loss = functional.mse_loss(value, batch["outcome"])
            loss = policy_loss + args.value_coef * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                network.parameters(), args.max_grad_norm
            )
            optimizer.step()
            step += 1
            if step % args.log_every == 0:
                wandb.log(
                    {
                        "train/policy_loss": float(policy_loss),
                        "train/value_loss": float(value_loss),
                        "train/loss": float(loss),
                        "train/grad_norm": float(grad_norm),
                        "epoch": epoch,
                    },
                    step=step,
                )

        metrics = evaluate(
            network, dataset, val_index, device, args.batch_size, args.value_coef
        )
        metrics["epoch"] = epoch
        wandb.log(metrics, step=step)
        print(
            f"epoch {epoch}: acc {metrics['val/accuracy']:.4f} "
            f"top5 {metrics['val/top5']:.4f} "
            f"policy {metrics['val/policy_loss']:.4f} "
            f"value {metrics['val/value_loss']:.4f}",
            flush=True,
        )
        if metrics["val/total"] < best:
            best = metrics["val/total"]
            torch.save(
                {
                    "format_version": 1,
                    "state_dict": {
                        k: v.cpu() for k, v in network.state_dict().items()
                    },
                    "config": OmegaConf.to_container(config, resolve=True),
                    "frames": 0,
                },
                output,
            )
            wandb.summary["best_val_accuracy"] = metrics["val/accuracy"]
            wandb.summary["best_val_top5"] = metrics["val/top5"]
            wandb.summary["best_epoch"] = epoch

    print(f"best val total {best:.4f}, checkpoint at {output}")
    wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("logs/bc_dataset.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--card-effect-features", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-group", default="bc-20260815")
    parser.add_argument("--wandb-name", required=True)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
