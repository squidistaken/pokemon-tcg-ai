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

## Deck scraping

First submit the 6-hour fetch-only discovery job. It scans at most 15,000
Limitless decks and 600 Bulbapedia category pages, writes no training decks, and
atomically checkpoints a resumable research inventory:

```bash
mkdir -p slurm-conf/logs
sbatch slurm-conf/discover_cards.sh
# outputs/card_discovery/seen_cards.jsonl.gz
```

Download that inventory, complete proposer/reviewer mapping work, and obtain Stef
or Teun's approval before submitting production.

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

Multi-deck training and evaluation use the heuristic corpus by default. Select the
mapping corpus with the top-level Hydra override:

```bash
./slurm-conf/train.sh --config ppo_selfplay_multideck \
  --slurm-config train_cpu deck_corpus=mapping-resolved
```
