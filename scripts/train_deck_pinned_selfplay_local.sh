#!/bin/bash
# Launch the local 10M-frame deck-pinned self-play run under the crash supervisor.
#
# Overlays conf/experiment/deck_pinned_selfplay_local.yaml, which inherits
# conf/experiment/weighted_field.yaml, on conf/ppo_selfplay_multideck.yaml. The
# league, the anchor and the eval references are the Habrok arm's; the only
# difference is that the learner pilots one deck. Every hyperparameter lives in
# those files; this script owns only what the supervisor needs as flags: which
# directory the run writes to, and the absolute frame target. Anything passed
# here is forwarded to src.train as a Hydra override, so a variant needs no
# edit:
#
#   ./scripts/train_deck_pinned_selfplay_local.sh
#   ./scripts/train_deck_pinned_selfplay_local.sh env.num_workers=8 agent.lr=1e-4
#   RESUME=1 ./scripts/train_deck_pinned_selfplay_local.sh
#
# Writes to its own run directory, deliberately not the frozen-pool variant's:
# the PFSP league scans the run's own checkpoints/, so sharing a directory would
# admit the other arm's snapshots into this league.
#
# Launch inside tmux/screen so the run survives a disconnect. A crash restarts
# from train_state.pt; see scripts/train_supervised.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG_NAME="ppo_selfplay_multideck"
EXPERIMENT="deck_pinned_selfplay_local"

GROUP="${GROUP:-deck-pinned-150m-local}"
RUN_NAME="${RUN_NAME:-tf-ptr-pinned-selfplay-10m-s42}"
# Absolute target, not an increment: the supervisor subtracts the warm start's
# 150,011,904 frames and collects 10M more, ending at 160,011,904.
TOTAL_FRAMES="${TOTAL_FRAMES:-160011904}"
ATTEMPTS="${ATTEMPTS:-10}"

RUN_DIR="outputs/$GROUP/$RUN_NAME"

# Weights-only warm start from the 150M snapshot. The supervisor reads its frame
# count so TOTAL_FRAMES stays the absolute target above. This is the same
# checkpoint the config freezes into the league as train.warmup_checkpoint.
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/weighted-field-20260808/tf-ptr-weighted-15m-s42/checkpoints/snapshot_000150011904.pt}"

WORKER_OVERRIDE=()
if [ -n "${NUM_WORKERS:-}" ]; then
  WORKER_OVERRIDE=("env.num_workers=$NUM_WORKERS")
fi

# Default: warm-start from the snapshot. RESUME=1 with an existing
# train_state.pt continues that state instead, for when a previous invocation
# was killed outright and its partial frames are wanted.
SUPERVISOR_ARGS=()
if [ "${RESUME:-0}" = "1" ] && [ -f "$RUN_DIR/train_state.pt" ]; then
  echo "RESUME=1 and $RUN_DIR/train_state.pt exists; continuing it."
else
  SUPERVISOR_ARGS=(--init-checkpoint "$INIT_CHECKPOINT")
fi

# Matched on the module rather than the interpreter: uv may exec python, python3
# or python3.13 depending on how the venv resolves, and a guard that misses lets
# a second run share the GPU and exhaust it. Bracketed first character so the
# pattern cannot match this script's own command line.
if pgrep -f "[-]m [s]rc.train" > /dev/null; then
  echo "ERROR: another src.train process is running and this run wants the same GPU." >&2
  exit 1
fi

mkdir -p logs
LOG="logs/train-deck-pinned-selfplay-$(date +%Y%m%dT%H%M%S).log"

echo "Config:  $CONFIG_NAME +experiment=$EXPERIMENT"
echo "Run dir: $RUN_DIR"
echo "Budget:  $TOTAL_FRAMES frames (10M on top of the 150M warm start)"

# Missing decks and checkpoints are not checked here: src.train builds the deck
# pool and the eval opponents before collecting, so a bad path fails in the
# first seconds and the supervisor reports it as a failure a restart cannot fix.
./scripts/train_supervised.sh \
  --run-dir "$RUN_DIR" \
  --total-frames "$TOTAL_FRAMES" \
  --attempts "$ATTEMPTS" \
  ${SUPERVISOR_ARGS[@]+"${SUPERVISOR_ARGS[@]}"} \
  -- --config-name "$CONFIG_NAME" \
  "+experiment=$EXPERIMENT" \
  "wandb.group=$GROUP" \
  "wandb.name=$RUN_NAME" \
  ${WORKER_OVERRIDE[@]+"${WORKER_OVERRIDE[@]}"} \
  "$@" \
  2>&1 | tee "$LOG"

echo "Log: $LOG"
