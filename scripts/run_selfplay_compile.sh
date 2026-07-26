#!/usr/bin/env bash
#
# 1M-frame PPO/MLP self-play run with torch.compile on the loss and policy.
#
# Any extra arguments are appended as Hydra overrides, e.g.
#   scripts/run_selfplay_compile.sh collector.total_frames=200000 train.pool_size=2
set -euo pipefail

cd "$(dirname "$0")/.."

# Inductor defaults compile_threads to the CPU count (16 here), and each worker
# is a full Python process with torch imported. Stacked on top of the 16 forked
# env workers, that peak is what got this run SIGKILLed by the OOM killer at
# 15 GB. Single-threaded compile trades a slower warmup for surviving.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

uv run python -m src.train \
  agent=ppo \
  train=ppo_selfplay \
  agent.device=cuda \
  agent.compile_loss=true \
  agent.compile_policy=true \
  collector.total_frames=1000000 \
  wandb.group=mlp-selfplay-compile \
  "$@"
