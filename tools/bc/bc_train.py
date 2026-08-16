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


def build_config(
    card_effect_features: bool,
    num_layers: int | None = None,
    ff_dim: int | None = None,
    dropout: float | None = None,
):
    """
    Load the reference architecture, with optional capacity overrides.

    The reference was sized for self-play, where the data is generated on
    demand. Cloning a fixed corpus of over a million decisions is a different
    regime, so the backbone is allowed to grow here.

    :param card_effect_features: Whether the adapter appends effect columns.
    :param num_layers: Transformer layers, or None to keep the reference value.
    :param ff_dim: Feed-forward width, or None to keep the reference value.
    :param dropout: Dropout probability, or None to keep the reference value.
    :return: Hydra-style config carrying a ``model`` section.
    """
    reference = torch.load(REFERENCE, map_location="cpu", weights_only=False)
    config = OmegaConf.create(reference["config"])
    OmegaConf.set_struct(config, False)
    config.model.adapter.card_effect_features = bool(card_effect_features)
    if num_layers is not None:
        config.model.backbone.num_layers = int(num_layers)
    if ff_dim is not None:
        config.model.backbone.ff_dim = int(ff_dim)
    if dropout is not None:
        config.model.backbone.dropout = float(dropout)
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


class ShardedDataset:
    """
    A corpus split across memory-mapped shards, addressed as one row space.

    The full corpus is far larger than RAM, so rows stay on disk and are paged
    in per batch. Indexing mirrors a plain TensorDict: pass row indices, get a
    stacked TensorDict back.
    """

    def __init__(self, root: Path):
        """
        :param root: Directory holding ``shard_*`` memory-maps and ``meta.json``.
        :raises FileNotFoundError: If no shards are present.
        """
        self._shards = [
            TensorDict.load_memmap(str(path))
            for path in sorted(root.glob("shard_*"))
            if path.is_dir()
        ]
        if not self._shards:
            raise FileNotFoundError(f"no shard_* directories under {root}")
        sizes = [int(shard.batch_size[0]) for shard in self._shards]
        self._offsets = torch.tensor([0] + sizes).cumsum(0)
        self.n_rows = int(self._offsets[-1])
        self.episode = torch.cat([shard["episode"] for shard in self._shards])

    def __len__(self) -> int:
        """
        :return: Total rows across all shards.
        """
        return self.n_rows

    def gather(self, rows: torch.Tensor) -> TensorDict:
        """
        Fetch a set of rows, regardless of which shards hold them.

        :param rows: Global row indices.
        :return: Stacked TensorDict of those rows.
        """
        shard_of = torch.bucketize(rows, self._offsets[1:], right=True)
        pieces: list[TensorDict] = []
        for shard_index in shard_of.unique().tolist():
            selected = rows[shard_of == shard_index]
            local = selected - int(self._offsets[shard_index])
            pieces.append(self._shards[shard_index][local])
        # TensorDict.cat rather than torch.cat: the latter only returns a
        # TensorDict through __torch_function__ at runtime and is stubbed as
        # returning a plain Tensor.
        return TensorDict.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Blank out every action the environment would reject.

    The mask comes from the dataset and reproduces ``TCGEnv._build_mask``, so
    it excludes options already picked and includes the stop slot exactly when
    ``minCount`` has been met.

    :param logits: Raw head output, ``(batch, MAX_OPTIONS + 1)``.
    :param mask: Bool legality mask of the same shape.
    :return: Logits with illegal slots driven to ``-inf``.
    """
    return logits.masked_fill(~mask, float("-inf"))


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
    # Roughly 45% of expert decisions are option 0, so raw accuracy is mostly a
    # measure of how often the clone says "0". These track the part that is not
    # free: how often it is right when the answer is not 0, and whether it
    # predicts 0 more often than the experts do.
    trivial = {"base": 0, "nontrivial_n": 0, "nontrivial_correct": 0, "pred_zero": 0}
    with torch.inference_mode():
        for start in range(0, index.numel(), batch_size):
            batch = dataset.gather(index[start : start + batch_size]).to(device)
            rows = batch.shape[0]
            forward = TensorDict(
                {"observation": batch["observation"]}, batch_size=batch.batch_size
            )
            network(forward)
            logits = masked_logits(
                forward.get("logits").reshape(rows, -1), batch["action_mask"]
            )
            value = forward.get("state_value").reshape(rows)
            action = batch["action"]
            totals["policy"] += float(
                functional.cross_entropy(logits, action, reduction="sum")
            )
            totals["value"] += float(
                functional.mse_loss(value, batch["outcome"], reduction="sum")
            )
            prediction = logits.argmax(dim=-1)
            totals["correct"] += int((prediction == action).sum())
            top5 = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
            totals["top5"] += int((top5 == action.unsqueeze(-1)).any(dim=-1).sum())
            totals["n"] += rows
            is_zero = action == 0
            trivial["base"] += int(is_zero.sum())
            trivial["pred_zero"] += int((prediction == 0).sum())
            trivial["nontrivial_n"] += int((~is_zero).sum())
            trivial["nontrivial_correct"] += int(
                (prediction[~is_zero] == action[~is_zero]).sum()
            )
    count = max(totals["n"], 1)
    policy = totals["policy"] / count
    value_loss = totals["value"] / count
    return {
        "val/accuracy": totals["correct"] / count,
        "val/top5": totals["top5"] / count,
        "val/base_rate": trivial["base"] / count,
        "val/nontrivial_accuracy": (
            trivial["nontrivial_correct"] / max(trivial["nontrivial_n"], 1)
        ),
        "val/predicts_zero": trivial["pred_zero"] / count,
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
    dataset = ShardedDataset(Path(args.dataset))
    total = len(dataset)
    generator = torch.Generator().manual_seed(args.seed)
    n_val = max(1, int(total * args.val_fraction))
    # Splitting at random by decision would put states from the same game on
    # both sides, and states within a game are far too correlated for that to
    # measure generalization. Holding out whole games keeps the two sides
    # independent; the dataset records the source episode for exactly this.
    if args.split == "game":
        episodes = dataset.episode
        unique = episodes.unique()
        shuffled = unique[torch.randperm(unique.numel(), generator=generator)]
        held_out = shuffled[: max(1, int(unique.numel() * args.val_fraction))]
        is_val = torch.isin(episodes, held_out)
        val_index = is_val.nonzero(as_tuple=True)[0]
        train_index = (~is_val).nonzero(as_tuple=True)[0]
        print(f"held out {held_out.numel()} of {unique.numel()} games", flush=True)
    else:
        order = torch.randperm(total, generator=generator)
        val_index = order[:n_val]
        train_index = order[n_val:]
    # The dataset is ~9 GB, so the split is kept as index tensors and the rows
    # are gathered per batch. Materializing two sub-TensorDicts would double
    # the resident set and exhaust memory on this box.
    print(
        f"split={args.split}  train {train_index.numel()}  val {val_index.numel()}",
        flush=True,
    )

    config = build_config(
        args.card_effect_features, args.num_layers, args.ff_dim, args.dropout
    )
    network = build_network(config).to(device)
    parameters = sum(p.numel() for p in network.parameters())
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    wandb.init(
        entity="pokemon-tcg-ai",
        project="pokemon-tcg-ai",
        group=args.wandb_group,
        name=args.wandb_name,
        config={
            "arm": args.wandb_name,
            "card_effect_features": bool(args.card_effect_features),
            "dataset": str(args.dataset),
            "samples": total,
            "split": args.split,
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
            batch = dataset.gather(shuffle[start : start + args.batch_size]).to(device)
            rows = batch.shape[0]
            forward = TensorDict(
                {"observation": batch["observation"]}, batch_size=batch.batch_size
            )
            network(forward)
            logits = masked_logits(
                forward.get("logits").reshape(rows, -1), batch["action_mask"]
            )
            value = forward.get("state_value").reshape(rows)
            policy_loss = functional.cross_entropy(logits, batch["action"])
            value_loss = functional.mse_loss(value, batch["outcome"])
            with torch.no_grad():
                # Train accuracy on the same rows the gradient just used, so a
                # train/val gap is visible without a second pass over the data.
                train_correct = (logits.argmax(dim=-1) == batch["action"]).float()
                train_accuracy = float(train_correct.mean())
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
                        "train/accuracy": train_accuracy,
                        "train/policy_loss": float(policy_loss.detach()),
                        "train/value_loss": float(value_loss.detach()),
                        "train/loss": float(loss.detach()),
                        "train/grad_norm": float(grad_norm.detach()),
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
            f"(base {metrics['val/base_rate']:.4f}) "
            f"nontrivial {metrics['val/nontrivial_accuracy']:.4f} "
            f"pred0 {metrics['val/predicts_zero']:.4f} "
            f"top5 {metrics['val/top5']:.4f} "
            f"value {metrics['val/value_loss']:.4f}",
            flush=True,
        )
        score = (
            metrics["val/policy_loss"]
            if args.select_on == "policy"
            else metrics["val/total"]
        )
        if score < best:
            best = score
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

    print(f"best val {args.select_on} {best:.4f}, checkpoint at {output}")
    wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("logs/bc_dataset.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--card-effect-features", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--ff-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--select-on",
        choices=("policy", "total"),
        default="policy",
        help="Metric picking the saved checkpoint. The value head degrades "
        "while the policy improves, so 'total' saves a worse policy.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument(
        "--split",
        choices=("game", "decision"),
        default="game",
        help="game: contiguous tail, a whole-game holdout. decision: random rows.",
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-group", default="bc-20260815")
    parser.add_argument("--wandb-name", required=True)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
