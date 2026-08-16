# Behaviour cloning

Turn Kaggle's ladder replays into a supervised policy (the "clone"), which then
serves as the self-play init weights and the frozen KL anchor. Three steps:
fetch episodes, extract decisions, train.

## 1. Fetch episodes

The raw data is Kaggle's daily episode export,
`kaggle/pokemon-tcg-ai-battle-episodes-<date>`, listed in
`kaggle/pokemon-tcg-ai-battle-episodes-index`. Download a day into a flat
`<date>/<episode>.json` tree:

```bash
kaggle datasets download -d kaggle/pokemon-tcg-ai-battle-episodes-2026-08-14 \
  -p logs/kaggle_episodes/2026-08-14 --unzip
```

`logs/bc/fetch_episodes.sh` wraps this loop for the days the corpus used
(9,062 games, 42 GB).

## 2. Extract decisions

```bash
uv run python tools/bc/bc_extract.py \
  --replay-glob 'logs/kaggle_episodes/*/*.json'
```

Writes memory-mapped shards and `meta.json` into the `--output` directory
(default `logs/bc_dataset.pt`). One row per accumulation position, encoded with
the same `StructuredObservationEncoder` RL uses; each row carries `observation`,
`action`, `action_mask`, `outcome` and `episode`. Useful flags:

| flag | default | meaning |
| --- | --- | --- |
| `--replay-glob` | required | glob for the flat daily-export tree |
| `--output` | `logs/bc_dataset.pt` | directory the shards are written into |
| `--max-replays` | 10000 | cap, for a smoke run |
| `--winners-only` | off | keep only decisions by the winning side |
| `--seed` | 0 | shuffle seed, so the split is reproducible |
| `--shard-size` | 1200 | replays per memory-mapped shard |

## 3. Train

```bash
uv run python tools/bc/bc_train.py \
  --output outputs/bc/my-clone.pt \
  --wandb-name my-clone
```

Loads the reference architecture from `--reference`, fits the policy head with
masked cross-entropy and the value head with MSE against the episode outcome,
and saves the best checkpoint by validation policy loss. Needs `WANDB_API_KEY`.

The model is the training stack unchanged: a `StructuredObsAdapter` turns the
encoded observation into tokens, a `TransformerBackbone` reads them, and a
`PointerHead` scores each option plus stop. The architecture is copied from
`--reference` (default pin-175.7M, 1 layer / 256 ff); `--num-layers` /
`--ff-dim` grow the backbone.

| flag | default | meaning |
| --- | --- | --- |
| `--dataset` | `logs/bc_dataset.pt` | the shard directory from step 2 |
| `--output` | required | where the trained checkpoint is written |
| `--reference` | pin-175.7M snapshot | architecture source, copied into the checkpoint config |
| `--num-layers` / `--ff-dim` / `--dropout` | None | grow the reference backbone |
| `--epochs` | 12 | |
| `--lr` / `--weight-decay` | 3e-4 / 1e-4 | AdamW |
| `--value-coef` | 0.5 | value-loss weight added to policy cross-entropy |
| `--split` | game | hold out whole games, not rows |
| `--device` | cuda | set `cpu` to train without a GPU |
| `--wandb-name` | required | W&B run label |

## Also here

- `head_to_head.py` — play two checkpoints against each other.
- `judge_arms.py` — score experiment arms against a baseline across decks.
- `lookahead.py`, `mc_rollout.py`, `mcts.py` — search over the engine's
  determinized simulator, wrapping a checkpoint.
- `critic_calibration.py` — correlate the critic's value against real outcomes.
