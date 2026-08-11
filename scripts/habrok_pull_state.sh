#!/bin/bash
# Pulls the training state Habrok has produced back to the desktop. The reverse
# of habrok_sync_state.sh, and it keeps the same 13-snapshot rule: the newest 12
# rebuild the league (src/env/snapshot_opponent_pool.py:113) and the local
# lowest-frame snapshot already covers the `first_snapshot` eval reference.
#
# Safe to run while the job is training. train_state.pt is replaced atomically
# (src/training/train_state.py), so rsync either reads the previous complete
# file or the new one, never a half-written mix.

set -euo pipefail

HOST=s4325621@login1.hb.hpc.rug.nl
SRC=/scratch/s4325621/pokemon-tcg-ai
RUN=outputs/weighted-field-20260808/tf-ptr-weighted-15m-s42
KEEP_COUNT="${KEEP_COUNT:-12}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p "$RUN/checkpoints"

echo "Remote frame count"
ssh "$HOST" "ls $SRC/$RUN/checkpoints/snapshot_*.pt | sort | tail -n 1"

echo "Resume state, 23 MB"
rsync -avz "$HOST:$SRC/$RUN/train_state.pt" "$RUN/"

echo "Newest $KEEP_COUNT league snapshots"
KEEP="$(ssh "$HOST" "cd $SRC/$RUN/checkpoints && ls snapshot_*.pt | sort | tail -n $KEEP_COUNT")"
for pt in $KEEP; do
  rsync -avz "$HOST:$SRC/$RUN/checkpoints/$pt" "$RUN/checkpoints/"
  rsync -avz "$HOST:$SRC/$RUN/checkpoints/${pt%.pt}.json" "$RUN/checkpoints/"
done

echo "Trainer log"
rsync -avz "$HOST:$SRC/$RUN/train.log" "$RUN/train.log.habrok"

echo
echo "Local state is now at:"
uv run --frozen --no-sync python -c \
  "import torch; print(torch.load('$RUN/train_state.pt', map_location='cpu')['frames'], 'frames')"
