#!/bin/bash
# Run one arm of the deck probe under the crash supervisor.
#
#   DECK=decks/heuristic-resolved/slowking/slowking.csv \
#   RUN_NAME=pinned-slowking \
#   CHECKPOINT=/scratch/.../snapshot_000176357376.pt \
#   TOTAL_FRAMES=276357376 \
#     ./scripts/train_deck_probe_arm.sh
#
# One deck, one run directory, one W&B run. slurm-conf/train_deck_probe_8way.sh
# starts eight of these at once on a single GPU, but nothing here is specific to
# Slurm: a single arm runs the same way on a desktop. Anything passed on the
# command line is forwarded to src.train as a Hydra override.
#
# Output goes to stdout rather than a log file of its own, because the caller
# decides where eight concurrent streams belong. Standalone:
#
#   ./scripts/train_deck_probe_arm.sh 2>&1 | tee logs/my-arm.log
#
# There is deliberately no "another src.train is running" guard, unlike
# scripts/train_tf_weighted_field.sh. Sharing one card is the design here, and
# the caller owns the arithmetic: one arm measured 7.5 GB of VRAM, so eight fit
# in the RTX Pro 6000's 96 GB with room to spare.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG_NAME="ppo_selfplay_multideck"
EXPERIMENT="deck_probe_8way"

GROUP="${GROUP:-deck-probe-8way}"
ATTEMPTS="${ATTEMPTS:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"

for required in DECK RUN_NAME CHECKPOINT TOTAL_FRAMES; do
  if [ -z "${!required:-}" ]; then
    echo "ERROR: $required is required; see the header of $0." >&2
    exit 2
  fi
done

# Checked here rather than left to src.train because eight arms start together:
# a typo in one deck path should stop that arm in the first second with a
# readable message, not surface as a Hydra error buried in a shared job log.
if [ ! -f "$DECK" ]; then
  echo "ERROR: DECK $DECK does not exist." >&2
  exit 2
fi
if [ ! -f "$CHECKPOINT" ]; then
  echo "ERROR: CHECKPOINT $CHECKPOINT does not exist." >&2
  exit 2
fi

# Its own directory, never shared: the PFSP league scans the run's own
# checkpoints/ (src/training/self_play.py), so two arms writing to one directory
# would recruit each other's snapshots and stop being independent.
RUN_DIR="outputs/$GROUP/$RUN_NAME"

# Default: warm-start from the shared checkpoint, weights only. RESUME=1 with an
# existing train_state.pt continues that state instead, which is what a requeued
# Slurm job wants: it keeps the frames and the optimizer moments the first
# allocation paid for.
SUPERVISOR_ARGS=(--init-checkpoint "$CHECKPOINT")
if [ "${RESUME:-0}" = "1" ] && [ -f "$RUN_DIR/train_state.pt" ]; then
  echo "RESUME=1 and $RUN_DIR/train_state.pt exists; continuing it."
  SUPERVISOR_ARGS=()
elif [ -f "$RUN_DIR/train_state.pt" ]; then
  echo "ERROR: $RUN_DIR/train_state.pt exists, so this would overwrite a previous arm." >&2
  echo "       To continue it: RESUME=1 $0" >&2
  echo "       To start over:  rm -rf $RUN_DIR" >&2
  exit 1
fi

echo "Config:     $CONFIG_NAME +experiment=$EXPERIMENT"
echo "Deck:       $DECK"
echo "Run dir:    $RUN_DIR"
echo "Anchor:     $CHECKPOINT"
echo "Budget:     $TOTAL_FRAMES frames (absolute)"
echo "Workers:    $NUM_WORKERS"

# train.warmup_checkpoint is passed as well as --init-checkpoint: the supervisor
# flag seeds the weights and the frame count, while the config value freezes the
# same file into the league as the anchor and resolves train.eval_opponents. A
# resume drops the flag but must keep the config value, or the league loses its
# anchor and the eval curve loses its yardstick halfway through the run.
./scripts/train_supervised.sh \
  --run-dir "$RUN_DIR" \
  --total-frames "$TOTAL_FRAMES" \
  --attempts "$ATTEMPTS" \
  ${SUPERVISOR_ARGS[@]+"${SUPERVISOR_ARGS[@]}"} \
  -- --config-name "$CONFIG_NAME" \
  "+experiment=$EXPERIMENT" \
  "train.warmup_checkpoint=$CHECKPOINT" \
  "env.eval_agent_deck=$DECK" \
  "env.num_workers=$NUM_WORKERS" \
  "wandb.group=$GROUP" \
  "wandb.name=$RUN_NAME" \
  "$@"
