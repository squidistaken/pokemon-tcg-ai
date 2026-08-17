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

# decks/ is gitignored, so the pool does not arrive with a git pull.
DECK_POOL="${DECK_POOL:-decks/expert_pool_30}"
if [ ! -d "$DECK_POOL" ] || [ -z "$(ls -A "$DECK_POOL" 2>/dev/null)" ]; then
  echo "ERROR: deck pool $DECK_POOL is missing or empty." >&2
  echo "       rsync -av decks/expert_pool_30/ <host>:\$PWD/decks/expert_pool_30/" >&2
  exit 1
fi

# A missing eval opponent raises inside src.train at the first evaluation,
# which is 50,000 frames into an allocation. Fail here instead.
EVAL_OPPONENT_CHECKPOINT="${EVAL_OPPONENT_CHECKPOINT:-outputs/deck-pinned-150m-local/tf-ptr-pinned-selfplay-10m-s42/checkpoints/snapshot_000175702016.pt}"
if [ ! -f "$EVAL_OPPONENT_CHECKPOINT" ]; then
  echo "ERROR: eval opponent checkpoint not found: $EVAL_OPPONENT_CHECKPOINT" >&2
  echo "       Copy it across, or drop it from train.eval_opponents." >&2
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
#
# Skipped under Slurm, which hands each job its own GPU: pgrep sees every
# process this user owns on the node, so a second arm landing on the same node
# as a running job was refused although the two never shared a device. Job
# 30641994 died 18 seconds in on exactly that. The guard is for the desktop,
# where two launches do land on the one GPU.
if [ -z "${SLURM_JOB_ID:-}" ] && pgrep -u "$(id -u)" -f "[-]m [s]rc.train" > /dev/null; then
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
echo "Deck pool:  $DECK_POOL ($(ls -1 "$DECK_POOL" | wc -l) decks)"
echo "Eval ref:   $EVAL_OPPONENT_CHECKPOINT"

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
  "train.warmup_checkpoint=$BC_CHECKPOINT" \
  "train.eval_opponent_checkpoint=$EVAL_OPPONENT_CHECKPOINT" \
  "env.deck_pool=$DECK_POOL" \
  ${WORKER_OVERRIDE[@]+"${WORKER_OVERRIDE[@]}"} \
  "$@" \
  2>&1 | tee "$LOG"

echo "Log: $LOG"
