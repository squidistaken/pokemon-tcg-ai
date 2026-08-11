# Slurm

Slurm profiles live here; training configs remain under `conf/`.

## Setup

```bash
./slurm-conf/setup_uv.sh            # Creates .venv
./slurm-conf/setup_uv.sh --use-rtx  # Submits a job that creates .venv-rtx
```

RTX setup returns immediately; wait for its job to finish before training.
Setup uses `uv.lock`, so rerun it after merges or dependency changes.

## Smoke test

```bash
./slurm-conf/train.sh \
  --config ppo \
  --slurm-config train_gpu \
  +experiment=debug
```

Check the job and its outputs:

```bash
squeue --me
JOB_ID=30314565  # Replace with the submitted job ID
tail -n 50 -f "slurm-conf/logs/pokemon-tcg-train_${JOB_ID}.out"
tail -n 50 -f "slurm-conf/logs/pokemon-tcg-train_${JOB_ID}.err"
```

## Training

```bash
./slurm-conf/train.sh --config <CONFIG_NAME> --slurm-config <PROFILE_NAME>
```

Both options are required and may appear in either order. Additional arguments
are passed to Hydra:

```bash
./slurm-conf/train.sh --config ppo --slurm-config train_gpu \
  collector.total_frames=1000000 \
  paths.data_dir=/scratch/$USER/pokemon-tcg-ai/decks \
  paths.output_dir=/scratch/$USER/pokemon-tcg-ai/outputs
```

Use `--dry-run` to print the `sbatch` command without submitting. Online W&B
requires `WANDB_API_KEY` in `.env` or a verified `wandb login`.

`slurm-conf/logs/` contains only Slurm stdout/stderr. Completed training runs
append their final checkpoint to the repository-local
`logs/checkpoint_keys.csv`; `run_job.sh` exports and prints its absolute path,
and the normal Python training callback performs the locked CSV append.

## Fixed-deck fine-tuning on RTX

The frozen and refreshed Alakazam-Dudunsparce experiments share
`conf/ppo_fixed_deck_finetune.yaml`. Both warm-start from the 85,688,320-frame
baseline with a fresh optimizer and collect 30,015,488 additional frames. The
launcher selects the `train_gpu_rtx` profile, seed 42, separate output
directories, and one shared W&B group.

The deck fetcher and launcher both accept one scratch-storage root. On the HPC
login node, download the released corpus directly to scratch and create the
checkpoint destination:

```bash
SCRATCH_ROOT=/scratch/s5862159/slopemon
./scripts/fetch_decks.sh --root "$SCRATCH_ROOT"
mkdir -p "$SCRATCH_ROOT/checkpoints/baseline-training-checkpoints"
```

The baseline checkpoint directory is ignored by Git. Copy all ten `.pt` files
and their `.json` sidecars into that directory. If the files are already in the
repository checkout on the cluster:

```bash
rsync -av checkpoints/baseline-training-checkpoints/ \
  "$SCRATCH_ROOT/checkpoints/baseline-training-checkpoints/"
```

From another machine, use the same destination after the HPC login hostname,
for example `hpc-login:"$SCRATCH_ROOT/checkpoints/baseline-training-checkpoints/"`.
The resulting layout is:

```text
/scratch/s5862159/slopemon/
├── decks/heuristic-resolved/...
├── checkpoints/baseline-training-checkpoints/snapshot_*.pt
└── outputs/                         # created by the training launcher
```

On the login node, prepare the RTX-specific environment and wait for its setup
job to finish:

```bash
./slurm-conf/setup_uv.sh --use-rtx
squeue --me
```

First inspect both submissions without starting jobs:

```bash
./scripts/run_fixed_deck_finetune_slurm.sh --dry-run \
  --storage-root "$SCRATCH_ROOT"
```

Then run one short RTX smoke job. It writes to a separate location and exercises
the real checkpoint population with one environment:

```bash
./scripts/run_fixed_deck_finetune_slurm.sh --mode frozen \
  --storage-root "$SCRATCH_ROOT" \
  paths.output_dir="$SCRATCH_ROOT/outputs/fixed-deck-smoke" \
  env.num_workers=1 collector.type=sync collector.total_frames=16384 \
  train.eval_interval=0 callbacks=none
```

After the smoke job succeeds, submit both production arms:

```bash
./scripts/run_fixed_deck_finetune_slurm.sh \
  --storage-root "$SCRATCH_ROOT"
```

Use `--mode refresh` or `--mode frozen` to submit only one arm. Monitor with
`squeue --me`, the files under `slurm-conf/logs/`, and W&B. Resumption is manual:
submit the same mode and output path with
`train.resume_state=/absolute/path/to/train_state.pt`, and set
`collector.total_frames` to the still-uncollected additional-frame budget.
`resume_state` restores the optimizer and absolute frame counter; do not point
it at a league snapshot. `FINETUNE_STORAGE_ROOT` is the environment-variable
equivalent of `--storage-root`; individual Hydra path overrides still take
precedence because the launcher forwards them last.

## Deck scraping

First submit the 6-hour fetch-only discovery job. It scans at most 15,000
Limitless decks and 600 Bulbapedia category pages, writes no training decks, and
atomically checkpoints a resumable research inventory:

```bash
mkdir -p slurm-conf/logs
sbatch slurm-conf/discover_cards.sh
# outputs/card_discovery/seen_cards.jsonl.gz
```

Download that inventory and complete the proposer/reviewer mapping work before
submitting production.

`scrape_all.sh` submits a 48-hour job to the regular CPU partition and runs every
network-backed deck source concurrently via `--source all` (Limitless and
Bulbapedia). Each
source deck is resolved through both strategies in the same fetch pass, producing
`decks/mapping-resolved/manifest.json` and
`decks/heuristic-resolved/manifest.json`:

```bash
mkdir -p slurm-conf/logs
sbatch slurm-conf/scrape_all.sh
```

Production fetches Limitless events since 2026-01-01 without a default deck cap
and follows Bulbapedia's entire `Deck archetypes` category. Bounds can be changed
through `SCRAPER_LIMIT`, `SCRAPER_MAX_PAGES`, `SCRAPER_PER_TOURNAMENT`,
`SCRAPER_SINCE`, `BULBAPEDIA_CATEGORY`, `BULBAPEDIA_MAX_PAGES`, and `SCRAPER_OUT`;
an explicit `--max-decks` may still be appended for a smaller run. Discovery also
accepts `SCRAPER_MAX_DECKS` (default 15000) and `CARD_DISCOVERY_OUT`.

Expect the resulting corpus to be Limitless-only. Bulbapedia's `Deck archetypes`
pages are historical decks built from pre-Scarlet & Violet sets that are not in
`EN_Card_Data.csv` and never will be, so all of them drop as unresolved — see the card
pool section of `scraper/README.md`. The source is still worth running for the card
research inventory, but it adds no training decks.

Multi-deck training and evaluation use the heuristic corpus by default. Select the
mapping corpus with the top-level Hydra override:

```bash
./slurm-conf/train.sh --config ppo_selfplay_multideck \
  --slurm-config train_cpu deck_corpus=mapping-resolved
```
