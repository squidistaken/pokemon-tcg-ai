#!/bin/bash
#SBATCH --job-name=ptcg-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

uv run python -m src.train \
  collector.total_frames=4096 \
  collector.frames_per_batch=1024 \
  env.num_workers=4 \
  set_seed=true \
  "$@"