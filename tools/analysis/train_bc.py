"""
Behaviour-clone the top-8 leaderboard teams from their replays.

Supervised, so it accumulates: the policy head is fit with masked cross-entropy
to the expert's pick and the value head with MSE to the episode result. That is
the opposite of what the RL loop does, which overwrites the previous
distribution at about 0.20 win rate per million frames.

Observations are cached as dicts on the first pass and encoded on the fly after
that, because encoding all rows up front is roughly 4 GB of option tables.
"""
import argparse
import json
import pickle
import random
import time
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch import nn
from torchrl.data import Binary, Categorical, Composite, Unbounded

from cg.api import to_observation_class
from src.env.observation.structured_observation_encoder import (
    StructuredObservationEncoder,
)
from src.policies.ppo_actor import build_actor_critic

load_dotenv(Path(__file__).parent / ".env", override=False)

MAX_OPTIONS = 128
STOP_INDEX = MAX_OPTIONS
CACHE = Path("logs/bc_observations.pkl")


def load_rows(index_path: str, cache_path: Path) -> list[dict]:
    """
    Load the index and attach each row's raw observation dict.

    :param index_path: Pickle written by build_bc_index.py.
    :param cache_path: Where to cache the extracted observations.
    :return: Rows carrying an ``observation`` dict alongside the label.
    """
    if cache_path.is_file():
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)

    with open(index_path, "rb") as handle:
        index = pickle.load(handle)
    index.sort(key=lambda row: (row["replay"], row["step"]))

    rows: list[dict] = []
    current_path, replay = None, None
    for row in index:
        if row["replay"] != current_path:
            current_path = row["replay"]
            replay = json.loads(Path(current_path).read_text())
        observation = replay["steps"][row["step"]][row["seat"]]["observation"]
        rows.append({
            "episode": row["replay"],
            "team": row.get("team"),
            "observation": observation,
            "seat": row["seat"],
            "chosen": row["chosen"],
            "label": row["label"],
            "n_options": row["n_options"],
            "min_count": row["min_count"],
            "outcome": row["outcome"],
        })
    with open(cache_path, "wb") as handle:
        pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return rows


def build_mask(row: dict) -> torch.Tensor:
    """
    Rebuild the env's action mask for one sub-decision.

    Mirrors ``TCGEnv._build_mask``: real options are legal, already-taken ones
    are not, and stop becomes legal once ``minCount`` picks are in.

    :param row: One training row.
    :return: Bool tensor of shape ``(MAX_OPTIONS + 1,)``.
    """
    mask = torch.zeros(MAX_OPTIONS + 1, dtype=torch.bool)
    mask[: row["n_options"]] = True
    for index in row["chosen"]:
        mask[index] = False
    if len(row["chosen"]) >= row["min_count"]:
        mask[STOP_INDEX] = True
    return mask


def build_network(init_checkpoint: str | None, encoder, overrides: dict | None = None):
    """
    Build the actor-critic, optionally warm-starting from a checkpoint.

    Cloning trains from scratch, so the architecture is free: ``overrides``
    reshapes the trunk without touching the observation spec, which stays
    fixed by the encoder.

    :param init_checkpoint: Snapshot to start from, or None for a fresh net.
    :param encoder: Observation encoder supplying the spec.
    :param overrides: Backbone keys to change (layers, width, dropout).
    :return: The network and the config it was built from.
    """
    n_actions = MAX_OPTIONS + 1
    obs_spec = Composite(
        observation=encoder.spec(),
        action_mask=Binary(n=n_actions, dtype=torch.bool),
        level_id=Unbounded(shape=(1,), dtype=torch.int64),
        opponent_is_anchor=Binary(1, dtype=torch.bool),
    )
    action_spec = Categorical(n_actions, dtype=torch.int64)
    if init_checkpoint:
        checkpoint = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        config = OmegaConf.create(checkpoint["config"])
        network = build_actor_critic(config, obs_spec, action_spec)
        network.load_state_dict(checkpoint["state_dict"], strict=True)
    else:
        reference = torch.load(
            "outputs/deck-pinned-150m-local/tf-ptr-pinned-selfplay-10m-s42/"
            "checkpoints/snapshot_000175702016.pt",
            map_location="cpu", weights_only=False,
        )
        config = OmegaConf.create(reference["config"])
        if overrides:
            for key, value in overrides.items():
                if key == "embed_dim":
                    config.model.embed_dim = value
                else:
                    config.model.backbone[key] = value
        network = build_actor_critic(config, obs_spec, action_spec)
    return network, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="logs/bc_index.pkl")
    parser.add_argument("--init", default=None,
                        help="Checkpoint to warm-start from; omit to train fresh.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--out", default="outputs/bc/bc_policy.pt")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--team", default=None,
                        help="Clone one team instead of the pooled corpus.")
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--ff-dim", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--wandb-group", default="behaviour-clone-20260814")
    parser.add_argument("--wandb-name", default="bc-top8-scratch")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = StructuredObservationEncoder(max_options=MAX_OPTIONS)

    print("loading rows...", flush=True)
    started = time.time()
    rows = load_rows(args.index, CACHE)
    print(f"  {len(rows):,} rows in {time.time() - started:.0f}s", flush=True)

    if args.team:
        # Cross-entropy over a pool of teams that disagree fits their average,
        # which is a policy none of them plays. Restricting to one team tests
        # whether that mixture is what caps accuracy.
        before = len(rows)
        rows = [row for row in rows if row.get("team") == args.team]
        print(f"  team filter {args.team!r}: {len(rows):,} of {before:,} rows", flush=True)

    # A few expert selections repeat an option index. TCGEnv masks options
    # already taken, so those picks have no representation in this action
    # space and would otherwise contribute a ~1e9 cross-entropy each.
    legal = [row for row in rows if bool(build_mask(row)[row["label"]])]
    dropped = len(rows) - len(legal)
    print(f"  dropped {dropped:,} rows with an unrepresentable duplicate pick "
          f"({dropped / max(len(rows), 1):.2%})", flush=True)
    rows = legal

    # Split by episode, not by row. Decisions inside one game share a board
    # state and an opponent, so a row-level split puts near-duplicates on both
    # sides and inflates held-out accuracy.
    episodes = sorted({row["episode"] for row in rows})
    random.Random(0).shuffle(episodes)
    holdout = set(episodes[: max(1, len(episodes) // 20)])
    train_rows = [row for row in rows if row["episode"] not in holdout]
    eval_rows = [row for row in rows if row["episode"] in holdout]
    print(f"  split by episode: {len(episodes) - len(holdout)} train games, "
          f"{len(holdout)} held-out games "
          f"({len(train_rows):,} / {len(eval_rows):,} rows)", flush=True)

    # The baseline any clone has to beat: option 0 is the expert's pick far
    # more often than chance, so accuracy against uniform random flatters it.
    always_zero = sum(1 for row in eval_rows if row["label"] == 0) / max(len(eval_rows), 1)
    print(f"  held-out baseline, always option 0: {always_zero:.3f}", flush=True)

    run = wandb.init(
        project="pokemon-tcg-ai",
        entity="pokemon-tcg-ai",
        group=args.wandb_group,
        name=args.wandb_name,
        job_type="behaviour-clone",
        config={
            "rows_total": len(rows),
            "rows_dropped_duplicate_pick": dropped,
            "rows_train": len(train_rows),
            "rows_eval": len(eval_rows),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "value_weight": args.value_weight,
            "init_checkpoint": args.init,
            "index": args.index,
        },
        tags=["behaviour-cloning", "expert-replays", "top8-leaderboard"],
    )

    overrides = {}
    if args.layers is not None:
        overrides["num_layers"] = args.layers
    if args.ff_dim is not None:
        overrides["ff_dim"] = args.ff_dim
    if args.dropout is not None:
        overrides["dropout"] = args.dropout
    if args.embed_dim is not None:
        overrides["embed_dim"] = args.embed_dim
    network, config = build_network(args.init, encoder, overrides)
    parameters = sum(p.numel() for p in network.parameters())
    print(f"  network: {parameters:,} parameters, overrides {overrides}", flush=True)
    wandb.config.update({"parameters": parameters, **{f"arch_{k}": v for k, v in overrides.items()}})
    network = network.to(device).train()
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_hard = -1.0

    def masked_cross_entropy(
        logits: torch.Tensor, masks: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-entropy with label smoothing spread over the legal actions only.

        torch's ``label_smoothing`` puts target mass on every class, including
        the ones masked to a large negative logit, which makes each row cost
        ``eps * 1e9`` and buries the real signal under a constant whose
        gradient only pushes illegal logits up.

        :param logits: Raw action logits.
        :param masks: Legal-action mask.
        :param labels: Index the expert chose.
        :return: Scalar loss.
        """
        log_probs = torch.log_softmax(logits.masked_fill(~masks, -torch.inf), dim=-1)
        log_probs = torch.nan_to_num(log_probs, neginf=0.0) * masks
        legal = masks.sum(dim=-1, keepdim=True).clamp(min=1)
        smoothing = args.label_smoothing
        # eps spread over every legal action, then the remaining 1-eps added
        # on the expert's pick, so the row sums to exactly 1.
        target = masks.float() * (smoothing / legal)
        target.scatter_add_(
            1,
            labels.unsqueeze(1),
            torch.full_like(labels, 1.0 - smoothing, dtype=target.dtype).unsqueeze(1),
        )
        return -(target * log_probs).sum(dim=-1).mean()

    def encode_batch(batch: list[dict]):
        observations = [
            encoder.encode(
                to_observation_class(row["observation"]), row["seat"], len(row["chosen"])
            )
            for row in batch
        ]
        stacked = torch.stack(observations).to(device)
        masks = torch.stack([build_mask(row) for row in batch]).to(device)
        labels = torch.tensor([row["label"] for row in batch], device=device)
        outcomes = torch.tensor(
            [row["outcome"] for row in batch], dtype=torch.float32, device=device
        )
        return TensorDict({"observation": stacked}, batch_size=[len(batch)]), masks, labels, outcomes

    @torch.no_grad()
    def evaluate() -> tuple[float, float, float]:
        """
        Score the held-out games.

        :return: Overall top-1, top-1 on rows where the expert did not take
            option 0, and the share of predictions that are option 0. The
            second number is the one that matters: the first can be bought
            with the positional prior alone.
        """
        network.eval()
        correct = total = 0
        hard_correct = hard_total = 0
        predicted_zero = 0
        for start in range(0, len(eval_rows), args.batch_size):
            batch = eval_rows[start : start + args.batch_size]
            td, masks, labels, _ = encode_batch(batch)
            network(td)
            logits = td.get("logits").masked_fill(~masks, -float("inf"))
            predictions = logits.argmax(dim=-1)
            hit = predictions == labels
            correct += int(hit.sum())
            total += len(batch)
            predicted_zero += int((predictions == 0).sum())
            hard = labels != 0
            hard_correct += int((hit & hard).sum())
            hard_total += int(hard.sum())
        network.train()
        return (
            correct / max(total, 1),
            hard_correct / max(hard_total, 1),
            predicted_zero / max(total, 1),
        )

    for epoch in range(args.epochs):
        random.Random(epoch).shuffle(train_rows)
        running, seen, started = 0.0, 0, time.time()
        for start in range(0, len(train_rows), args.batch_size):
            batch = train_rows[start : start + args.batch_size]
            td, masks, labels, outcomes = encode_batch(batch)
            network(td)
            policy_loss = masked_cross_entropy(td.get("logits"), masks, labels)
            values = td.get("state_value").reshape(-1)
            value_loss = nn.functional.mse_loss(values, outcomes)
            loss = policy_loss + args.value_weight * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            running += float(policy_loss.detach()) * len(batch)
            seen += len(batch)
        accuracy, hard_accuracy, predicted_zero = evaluate()
        scheduler.step()
        wandb.log({
            "bc/epoch": epoch + 1,
            "bc/train_cross_entropy": running / seen,
            "bc/heldout_top1": accuracy,
            "bc/heldout_top1_nontrivial": hard_accuracy,
            "bc/heldout_lift_over_always_zero": accuracy - always_zero,
            "bc/predicted_option_zero_share": predicted_zero,
            "bc/lr": scheduler.get_last_lr()[0],
            "bc/epoch_seconds": time.time() - started,
        })
        print(f"epoch {epoch + 1}/{args.epochs}  CE {running / seen:.4f}  "
              f"top1 {accuracy:.3f} (base {always_zero:.3f})  "
              f"nontrivial {hard_accuracy:.3f}  "
              f"pred-0 {predicted_zero:.3f}  {time.time() - started:.0f}s", flush=True)
        # Keep the best epoch by the metric the positional prior cannot buy.
        if hard_accuracy > best_hard:
            best_hard = hard_accuracy
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "format_version": 1,
                "state_dict": {k: v.cpu() for k, v in network.state_dict().items()},
                "config": OmegaConf.to_container(config, resolve=True),
                "frames": 0,
            }, args.out)

    wandb.summary["bc/final_heldout_top1"] = accuracy
    wandb.summary["bc/best_heldout_nontrivial"] = best_hard
    wandb.summary["bc/always_zero_baseline"] = always_zero
    run.finish()
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
