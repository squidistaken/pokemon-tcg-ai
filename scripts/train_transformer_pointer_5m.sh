#!/bin/bash
# 5M-frame PPO self-play run on the transformer backbone with the pointer head.
#
# Launcher only: every hyperparameter, and the rationale for it, lives in
# conf/experiment/tf_pointer_5m.yaml. Extra arguments go straight to Hydra:
#
#   ./scripts/train_transformer_pointer_5m.sh
#   ./scripts/train_transformer_pointer_5m.sh seed=7 train.pool_size=5
#
# Runs ~3.5h in the foreground. Launch inside tmux/screen so it survives a
# disconnect -- the 15M run died at 3.08M frames when WSL restarted, and there
# is no resume path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# The repo's own modules (submission/, src/, main.py) collide with same-named
# top-level modules on an inherited PYTHONPATH, so the run gets a clean one.
export PYTHONPATH=

mkdir -p logs
LOG="logs/train-tf-pointer-5m-$(date +%Y%m%dT%H%M%S).log"

uv run --frozen --no-sync python -m src.train \
  --config-name ppo_selfplay_multideck \
  +experiment=tf_pointer_5m \
  "$@" \
  2>&1 | tee "$LOG"

echo "Log: $LOG"
