#!/bin/bash
# Copies the code and the minimum training state Habrok needs to continue the
# weighted-field arm. Run from the repo root on the desktop.
#
# Only 13 of the 173 snapshots go over. The league keeps
# paths[-pool_size:] (src/env/snapshot_opponent_pool.py:113), so the newest 12
# reproduce it exactly, and the lowest-frame one is what the `first_snapshot`
# eval reference resolves to (src/training/self_play.py:155). Sending all 173
# would move 1.3 GB to no effect.

set -euo pipefail

HOST=s4325621@login1.hb.hpc.rug.nl
DEST=/scratch/s4325621/pokemon-tcg-ai
RUN=outputs/weighted-field-20260808/tf-ptr-weighted-15m-s42
SUBMITTED=outputs/deck-pinned-20260808/mlp-ptr-pinned-15m-s42/checkpoints

# --code-only sends code and decks but never the training state. Habrok's
# train_state.pt is ahead of the desktop's whenever a job has run there since
# the last pull, and the unguarded push below would silently replace it with the
# older local copy, discarding those frames. Use --code-only to change config or
# decks under a run that is already further along.
CODE_ONLY=0
if [ "${1:-}" = "--code-only" ]; then
  CODE_ONLY=1
fi

ssh "$HOST" "mkdir -p $DEST/$RUN/checkpoints $DEST/$SUBMITTED /scratch/s4325621/logs /scratch/s4325621/wandb_cache"

echo "Code, configs and scripts"
rsync -avz --delete \
  --exclude '.git' --exclude '.venv*' --exclude '__pycache__' \
  --exclude 'outputs' --exclude 'wandb' --exclude 'submissions' \
  --exclude 'decks' --exclude 'decks-v2-baseline' \
  --exclude 'checkpoint___' --exclude 'logs' \
  ./ "$HOST:$DEST/"

echo "Deck corpus, 552 MB"
rsync -avz decks/ "$HOST:$DEST/decks/"

if [ "$CODE_ONLY" -eq 1 ]; then
  echo
  echo "--code-only: left Habrok's train_state.pt and checkpoints untouched."
  echo "Remote frame count:"
  ssh "$HOST" "ls -la $DEST/$RUN/train_state.pt; ls $DEST/$RUN/checkpoints/snapshot_*.pt | sort | tail -n 1"
else
  echo "Resume state, 23 MB"
  rsync -avz "$RUN/train_state.pt" "$HOST:$DEST/$RUN/"

  echo "League snapshots: newest 12 plus the first, 13 files"
  KEEP="$(cd "$RUN/checkpoints" && { ls snapshot_*.pt | sort | tail -n 12; ls snapshot_*.pt | sort | head -n 1; } | sort -u)"
  for pt in $KEEP; do
    rsync -avz "$RUN/checkpoints/$pt" "$RUN/checkpoints/${pt%.pt}.json" \
      "$HOST:$DEST/$RUN/checkpoints/"
  done

  echo "The fixed external eval reference, 11 MB"
  rsync -avz "$SUBMITTED/snapshot_000010141696.pt" "$SUBMITTED/snapshot_000010141696.json" \
    "$HOST:$DEST/$SUBMITTED/"
fi

echo
echo "Done. On the Habrok login node:"
echo "  cd $DEST"
echo "  sbatch --time=1-00:00:00 --mem=96G --export=ALL,TOTAL_FRAMES=150011904 \\"
echo "    slurm-conf/train_weighted_field.sh"
