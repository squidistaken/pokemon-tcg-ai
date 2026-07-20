# Slurm configuration

Scheduler profiles and scripts live here, separate from the Hydra configs in
`conf/`.

## Setup

Run the setup for the GPU type you will use:

```bash
./slurm-conf/setup_uv.sh
./slurm-conf/setup_uv.sh --use-rtx
```

These create `.venv` and `.venv-rtx`, respectively. Setup uses `uv.lock` and
checks that PyTorch and the included Linux engine can be loaded.

## Submit

```bash
./slurm-conf/train.sh --config config --slurm-config train_gpu
./slurm-conf/train.sh --config config --slurm-config train_gpu_rtx
```

`--config` selects a complete YAML under `conf/`. `--slurm-config` selects a
YAML under `slurm-conf/`. Both are required and their order does not matter.
Extra arguments are passed to Hydra:

```bash
./slurm-conf/train.sh --slurm-config train_gpu --config config \
  collector.total_frames=1000000 \
  paths.data_dir=/scratch/$USER/pokemon-tcg-ai/decks
```

Add `--dry-run` to print the `sbatch` command without submitting.

Data and output paths are configured in `conf/paths/default.yaml`.
