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

`scrape_all.sh` submits a 48-hour job to the regular CPU partition and runs every
network-backed deck source via `--source all` (Limitless and Bulbapedia). Each
source deck is resolved through both strategies in the same fetch pass, producing
`decks/mapping-resolved/manifest.json` and
`decks/heuristic-resolved/manifest.json`:

```bash
./slurm-conf/scrape_all.sh
```

The defaults fetch Limitless events since 2026-01-01 and Bulbapedia's
`Deck archetypes` category. Bounds can be changed through `SCRAPER_LIMIT`,
`SCRAPER_MAX_PAGES`, `SCRAPER_MAX_DECKS`, `SCRAPER_PER_TOURNAMENT`,
`SCRAPER_SINCE`, `BULBAPEDIA_CATEGORY`, and `SCRAPER_OUT`. Additional arguments
are appended to the scraper command.

Multi-deck training and evaluation use the heuristic corpus by default. Select the
mapping corpus with the top-level Hydra override:

```bash
./slurm-conf/train.sh --config ppo_selfplay_multideck \
  --slurm-config train_cpu deck_corpus=mapping-resolved
```
