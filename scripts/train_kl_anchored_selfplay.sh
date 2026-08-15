#!/bin/bash
# Launch conf/experiment/kl_anchored_selfplay.yaml under the crash supervisor.
#
# Every hyperparameter lives in that config. This script owns only what the
# supervisor needs: the run directory and the total frame budget. Anything
# passed here is forwarded to src.train as a Hydra override:
#
#   ./scripts/train_kl_anchored_selfplay.sh
#   ./scripts/train_kl_anchored_selfplay.sh agent.kl_anchor_coeff=0.1
#   RESUME=1 ./scripts/train_kl_anchored_selfplay.sh
#
# Launch inside tmux/screen so the run survives a disconnect. A crash restarts
# from train_state.pt; see scripts/train_supervised.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG_NAME="ppo_selfplay_multideck"
EXPERIMENT="kl_anchored_selfplay"

WANDB_GROUP="${WANDB_GROUP:-kl-anchored-selfplay-20260816}"
RUN_NAME="${RUN_NAME:-kl0.05-bcv6-expertpool}"
# Absolute target, not an increment: the supervisor subtracts what
# train_state.pt already records. A multiple of the 16384-frame batch keeps the
# last batch whole. Set past what 24h collects so the job uses the whole
# allocation and stops cleanly at the time limit.
TOTAL_FRAMES="${TOTAL_FRAMES:-99991552}"
ATTEMPTS="${ATTEMPTS:-10}"

# The clone supplies both the initial weights and the frozen KL reference. It
# has to exist before anything else: a missing file here means the run trains
# from scratch with an anchor to noise, which looks like a working run.
BC_CHECKPOINT="${BC_CHECKPOINT:-outputs/bc/bc-v6-submit.pt}"
if [ ! -f "$BC_CHECKPOINT" ]; then
  echo "ERROR: behaviour-cloned checkpoint not found: $BC_CHECKPOINT" >&2
  echo "       Copy it from the desktop, or set BC_CHECKPOINT." >&2
  exit 1
fi

if [ ! -d "decks/expert_pool" ]; then
  echo "ERROR: decks/expert_pool is missing; the deck pool is not in the repo." >&2
  exit 1
fi

WORKER_OVERRIDE=()
if [ -n "${NUM_WORKERS:-}" ]; then
  WORKER_OVERRIDE=("env.num_workers=$NUM_WORKERS")
fi

RUN_DIR="outputs/$WANDB_GROUP/$RUN_NAME"

# Matched on the module rather than the interpreter: uv may exec python, python3
# or python3.13, and a guard that misses lets a second run share the GPU and
# exhaust it. Bracketed first character so the pattern cannot match this
# script's own command line.
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
LOG="logs/train-kl-anchored-$(date +%Y%m%dT%H%M%S).log"

echo "Config:     $CONFIG_NAME +experiment=$EXPERIMENT"
echo "Run dir:    $RUN_DIR"
echo "Budget:     $TOTAL_FRAMES frames"
echo "Clone:      $BC_CHECKPOINT"

./scripts/train_supervised.sh \
  --run-dir "$RUN_DIR" \
  --total-frames "$TOTAL_FRAMES" \
  --attempts "$ATTEMPTS" \
  -- --config-name "$CONFIG_NAME" \
  "+experiment=$EXPERIMENT" \
  "wandb.group=$WANDB_GROUP" \
  "wandb.name=$RUN_NAME" \
  "agent.kl_anchor_checkpoint=$BC_CHECKPOINT" \
  "train.init_checkpoint=$BC_CHECKPOINT" \
  ${WORKER_OVERRIDE[@]+"${WORKER_OVERRIDE[@]}"} \
  "$@" \
  2>&1 | tee "$LOG"

echo "Log: $LOG"
