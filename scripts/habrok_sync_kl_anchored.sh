#!/bin/bash
# Copies the code and the three files the KL-anchored self-play run needs.
# Run from the repo root on the desktop.
#
# The run starts from the behaviour-cloned policy rather than from a
# train_state.pt, so nothing here can overwrite a run already in progress on
# Habrok. That is why this script has no --code-only guard: it never sends
# resume state.
#
#   ./scripts/habrok_sync_kl_anchored.sh
#
# Then on the login node:
#   cd /scratch/s4325621/pokemon-tcg-ai
#   sbatch slurm-conf/train_kl_anchored_smoke.sh   # 30 min, any GPU
#   sbatch slurm-conf/train_kl_anchored.sh         # 24 h, RTX Pro 6000

set -euo pipefail

HOST="${HABROK_HOST:-s4325621@login1.hb.hpc.rug.nl}"
DEST="${HABROK_DEST:-/scratch/s4325621/pokemon-tcg-ai}"

CLONE=outputs/bc/bc-v6-submit.pt
EVAL_REF_DIR=outputs/deck-pinned-150m-local/tf-ptr-pinned-selfplay-10m-s42/checkpoints
EVAL_REF=snapshot_000175702016.pt

for required in "$CLONE" "$EVAL_REF_DIR/$EVAL_REF" decks/expert_pool_30; do
  if [ ! -e "$required" ]; then
    echo "ERROR: $required is missing locally; nothing to sync." >&2
    exit 1
  fi
done

ssh "$HOST" "mkdir -p $DEST/outputs/bc $DEST/$EVAL_REF_DIR /scratch/s4325621/logs"

echo "Code, configs and scripts"
# The data directories are anchored with a leading slash so they match only at
# the transfer root. Unanchored, rsync matches a name at any depth: plain
# 'decks' also skipped src/env/decks/, which shipped a tree missing that package
# and failed on import at job start. 'logs' stays unanchored on purpose, to keep
# --delete away from slurm-conf/logs/ on the remote.
rsync -avz --delete \
  --exclude '.git' --exclude '.venv*' --exclude '__pycache__' \
  --exclude '/outputs' --exclude '/wandb' --exclude '/submissions' \
  --exclude '/decks' --exclude '/decks-v2-baseline' \
  --exclude '/checkpoint___' --exclude 'logs' \
  ./ "$HOST:$DEST/"

# decks/ is gitignored, so the pool only reaches Habrok this way.
echo "Deck corpus, including decks/expert_pool_30"
rsync -avz decks/ "$HOST:$DEST/decks/"

echo "Behaviour-cloned policy, 10 MB: warm start and frozen KL reference"
rsync -avz "$CLONE" "$HOST:$DEST/outputs/bc/"

echo "Eval reference pin-175.7M, 8 MB"
rsync -avz "$EVAL_REF_DIR/$EVAL_REF" "$EVAL_REF_DIR/${EVAL_REF%.pt}.json" \
  "$HOST:$DEST/$EVAL_REF_DIR/" 2>/dev/null \
  || rsync -avz "$EVAL_REF_DIR/$EVAL_REF" "$HOST:$DEST/$EVAL_REF_DIR/"

echo
echo "Verifying the three files landed"
ssh "$HOST" "cd $DEST && ls -la $CLONE $EVAL_REF_DIR/$EVAL_REF && echo \"decks/expert_pool_30: \$(ls -1 decks/expert_pool_30 | wc -l) decks\""

echo
echo "Done. On the Habrok login node:"
echo "  cd $DEST"
echo "  sbatch slurm-conf/train_kl_anchored_smoke.sh   # 30 min, any GPU, prove it works"
echo "  sbatch slurm-conf/train_kl_anchored.sh         # 24 h, RTX Pro 6000"
