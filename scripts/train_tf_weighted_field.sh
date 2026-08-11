#!/bin/bash
# Launch conf/experiment/weighted_field.yaml under the crash supervisor.
#
# Every hyperparameter lives in that config. This script owns only what the
# supervisor needs as flags: which directory the run writes to, and how many
# frames it is worth in total. Anything passed here is forwarded to src.train
# as a Hydra override, so a variant needs no edit:
#
#   ./scripts/train_tf_weighted_field.sh
#   ./scripts/train_tf_weighted_field.sh env.num_workers=32 agent.lr=1e-4
#   RESUME=1 ./scripts/train_tf_weighted_field.sh
#
# Launch inside tmux/screen so the run survives a disconnect. A crash restarts
# from train_state.pt; see scripts/train_supervised.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG_NAME="ppo_selfplay_multideck"
EXPERIMENT="weighted_field"

GROUP="${GROUP:-weighted-field-20260808}"
RUN_NAME="${RUN_NAME:-tf-ptr-weighted-15m-s42}"
# Absolute target, not an increment: the supervisor subtracts what
# train_state.pt already records. A multiple of the 16384-frame batch keeps the
# last batch whole.
TOTAL_FRAMES="${TOTAL_FRAMES:-99991552}"
ATTEMPTS="${ATTEMPTS:-10}"

RUN_DIR="outputs/$GROUP/$RUN_NAME"

# Matched on the module rather than the interpreter: uv may exec python, python3
# or python3.13 depending on how the venv resolves, and a guard that misses lets
# a second run share the GPU and exhaust it. Bracketed first character so the
# pattern cannot match this script's own command line.
if pgrep -f "[-]m [s]rc.train" > /dev/null; then
  echo "ERROR: another src.train process is running and this run wants the same GPU." >&2
  exit 1
fi

# The supervisor resumes from train_state.pt by design, which is right after a
# crash and wrong on a deliberate relaunch, so a leftover state stops the run
# instead of being picked up silently.
if [ -f "$RUN_DIR/train_state.pt" ] && [ "${RESUME:-0}" != "1" ]; then
  echo "ERROR: $RUN_DIR/train_state.pt exists, so this would CONTINUE a previous run." >&2
  echo "       To resume it:  RESUME=1 $0" >&2
  echo "       To start over: rm -rf $RUN_DIR" >&2
  exit 1
fi

mkdir -p logs
LOG="logs/train-weighted-field-$(date +%Y%m%dT%H%M%S).log"

echo "Config:  $CONFIG_NAME +experiment=$EXPERIMENT"
echo "Run dir: $RUN_DIR"
echo "Budget:  $TOTAL_FRAMES frames"

# Missing decks and checkpoints are not checked here: src.train builds the deck
# pool and the eval opponents before collecting, so a bad path fails in the
# first seconds and the supervisor reports it as a failure a restart cannot fix.
./scripts/train_supervised.sh \
  --run-dir "$RUN_DIR" \
  --total-frames "$TOTAL_FRAMES" \
  --attempts "$ATTEMPTS" \
  -- --config-name "$CONFIG_NAME" \
  "+experiment=$EXPERIMENT" \
  "wandb.group=$GROUP" \
  "wandb.name=$RUN_NAME" \
  "$@" \
  2>&1 | tee "$LOG"

echo "Log: $LOG"
