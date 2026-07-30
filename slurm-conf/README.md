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
./slurm-conf/train.sh --config baseline --slurm-config train_gpu
./slurm-conf/train.sh --config ppo --slurm-config train_gpu
./slurm-conf/train.sh --config ppo_selfplay --slurm-config train_gpu_rtx
./slurm-conf/train.sh --config ppo_selfplay_multideck --slurm-config train_gpu
./slurm-conf/train.sh --config ppo_selfplay_multideck --slurm-config train_cpu
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
