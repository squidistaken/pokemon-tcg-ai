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

## Evaluation

`eval.sh` submits `src.eval_deck_field` (the per-deck field-performance probe,
see the top-level README) instead of `src.train`, on the same profiles and
with the same options as `train.sh`:

```bash
./slurm-conf/eval.sh --config eval_deck_field --slurm-config train_cpu \
  train.eval_opponent_checkpoint=/path/to/agent.pt
```

`train.sh` itself also accepts `--module <dotted.path>` (default `src.train`)
if you need to run some other entry point on a profile directly.
