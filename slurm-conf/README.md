# Slurm configuration

Scheduler profiles and scripts live here, separate from the Hydra configs in
`conf/`.

## Setup

Run the setup for the GPU type you will use:

```bash
./slurm-conf/setup_uv.sh
./slurm-conf/setup_uv.sh --use-rtx
```

The normal setup creates `.venv` directly. RTX setup submits a 15-minute RTX
job and returns immediately; that job creates `.venv-rtx`, checks the GPU, and
runs a CUDA calculation. Setup uses `uv.lock`; rerun it after merges or
dependency changes. Wait for the RTX setup job to finish before submitting
training.

## Submit

```bash
# Random baseline on A100
./slurm-conf/train.sh --config baseline --slurm-config train_gpu

# PPO against the fixed opponent on A100
./slurm-conf/train.sh --config ppo --slurm-config train_gpu

# PPO self-play on RTX
./slurm-conf/train.sh --config ppo_selfplay --slurm-config train_gpu_rtx
```

`--config` selects a complete YAML under `conf/`. `--slurm-config` selects a
YAML under `slurm-conf/`. Both are required and their order does not matter.
Extra arguments are passed to Hydra:

```bash
./slurm-conf/train.sh --config ppo --slurm-config train_gpu \
  collector.total_frames=1000000 \
  paths.data_dir=/scratch/$USER/pokemon-tcg-ai/decks \
  paths.output_dir=/scratch/$USER/pokemon-tcg-ai/outputs
```

Add `--dry-run` to print the `sbatch` command without submitting.

Online W&B runs require `WANDB_API_KEY` in the project `.env` or a verified
`wandb login`. W&B and Hydra files follow the output location configured in
`conf/paths/default.yaml`.
