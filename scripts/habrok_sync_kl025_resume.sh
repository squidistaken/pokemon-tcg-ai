#!/bin/bash
# Sets up the kl_anchor_coeff=0.25 arm on Habrok in its own tree, seeded from
# the desktop run. Run from the repo root on the desktop.
#
#   ./scripts/habrok_sync_kl025_resume.sh
#   ./scripts/habrok_sync_kl025_resume.sh --code-only
#
# Then on the login node:
#   cd /scratch/s4325621/pokemon-tcg-ai-kl025
#   sbatch slurm-conf/train_kl025_anchored.sh
#
# Why a second tree rather than /scratch/s4325621/pokemon-tcg-ai: the kl0.05 arm
# is running out of that one, its supervisor re-reads the code on every restart,
# and the desktop branch differs from what was synced there (it adds
# train.warmup_checkpoint to conf/experiment/kl_anchored_selfplay.yaml, which
# would put the clone in that run's league mid-experiment). A separate root
# leaves the running job on the exact code it started with.
#
# decks/ and the uv environments are symlinked back to the main tree instead of
# copied: 559 MB of decks and two full environments, identical in both.
#
# This script sends resume state, so it refuses to overwrite a remote
# train_state.pt: once the Habrok job has run, the remote state is ahead of the
# desktop's and pushing over it would discard those frames. Use --code-only to
# update code under a run that is already further along.

set -euo pipefail

HOST="${HABROK_HOST:-s4325621@login1.hb.hpc.rug.nl}"
MAIN="${HABROK_MAIN:-/scratch/s4325621/pokemon-tcg-ai}"
DEST="${HABROK_DEST:-/scratch/s4325621/pokemon-tcg-ai-kl025}"

SOURCE_RUN="${SOURCE_RUN:-outputs/kl-coeff-ablation-20260816/kl0.25-local-16w}"
TARGET_RUN="${TARGET_RUN:-outputs/kl-coeff-ablation-20260816/kl0.25-bcv6-habrok32}"

CLONE=outputs/bc/bc-v6-submit.pt
EVAL_REF_DIR=outputs/deck-pinned-150m-local/tf-ptr-pinned-selfplay-10m-s42/checkpoints
EVAL_REF=snapshot_000175702016.pt

CODE_ONLY=0
if [ "${1:-}" = "--code-only" ]; then
  CODE_ONLY=1
fi

if [ "$CODE_ONLY" -eq 0 ] && [ ! -f "$SOURCE_RUN/train_state.pt" ]; then
  echo "ERROR: $SOURCE_RUN/train_state.pt does not exist; there is no state to seed from." >&2
  exit 1
fi

echo "Tree, symlinks and directories"
ssh "$HOST" "set -e
  mkdir -p $DEST/slurm-conf/logs $DEST/$TARGET_RUN/checkpoints $DEST/$EVAL_REF_DIR $DEST/outputs/bc
  for link in decks .venv .venv-rtx; do
    if [ ! -e $DEST/\$link ]; then ln -s $MAIN/\$link $DEST/\$link; fi
  done
  # Same filesystem, so the two checkpoints cost nothing to place here.
  cp -n $MAIN/$CLONE $DEST/$CLONE
  cp -n $MAIN/$EVAL_REF_DIR/$EVAL_REF $DEST/$EVAL_REF_DIR/$EVAL_REF
  cp -n $MAIN/$EVAL_REF_DIR/${EVAL_REF%.pt}.json $DEST/$EVAL_REF_DIR/ 2>/dev/null || true"

echo "Code, configs and scripts"
# The data directories are anchored with a leading slash so they match only at
# the transfer root. Unanchored, rsync matches a name at any depth: plain
# 'decks' also skipped src/env/decks/, which shipped a tree missing that package
# and failed on import at job start. No --delete: /decks, /outputs and the
# environments are symlinks on the far side and must survive the transfer.
rsync -avz \
  --exclude '.git' --exclude '.venv*' --exclude '__pycache__' \
  --exclude '/outputs' --exclude '/wandb' --exclude '/submissions' \
  --exclude '/decks' --exclude '/decks-v2-baseline' \
  --exclude '/checkpoint___' --exclude 'logs' \
  --exclude '.pytest_cache' --exclude '.ruff_cache' \
  --exclude '*.zip' --exclude '*.csv' \
  ./ "$HOST:$DEST/"

if [ "$CODE_ONLY" -eq 1 ]; then
  echo
  echo "--code-only: left the remote train_state.pt and checkpoints untouched."
  ssh "$HOST" "ls -la $DEST/$TARGET_RUN/train_state.pt 2>/dev/null || echo 'no remote state yet'"
  exit 0
fi

if ssh "$HOST" "test -f $DEST/$TARGET_RUN/train_state.pt"; then
  echo >&2
  echo "ERROR: $TARGET_RUN/train_state.pt already exists on Habrok." >&2
  echo "       That state is at or ahead of this one; overwriting it would discard frames." >&2
  echo "       Continue that run instead:  sbatch slurm-conf/train_kl025_anchored.sh" >&2
  echo "       Or pass --code-only to send code alone." >&2
  exit 1
fi

echo "Resume state, 30 MB"
rsync -avz "$SOURCE_RUN/train_state.pt" "$HOST:$DEST/$TARGET_RUN/"

# The whole league, not a tail: pool_size is 10 and this run has fewer than
# that, so every snapshot is still a live member
# (src/env/opponents/snapshot_opponent_pool.py:113).
echo "League snapshots"
rsync -avz "$SOURCE_RUN/checkpoints/" "$HOST:$DEST/$TARGET_RUN/checkpoints/"

echo
echo "Verifying"
ssh "$HOST" "cd $DEST && ls -la $CLONE $EVAL_REF_DIR/$EVAL_REF $TARGET_RUN/train_state.pt && \
  echo \"league: \$(ls -1 $TARGET_RUN/checkpoints/snapshot_*.pt | wc -l) snapshots\" && \
  echo \"decks/expert_pool_30: \$(ls -1 decks/expert_pool_30 | wc -l) decks\""

echo
echo "Done. On the Habrok login node:"
echo "  cd $DEST"
echo "  sbatch slurm-conf/train_kl025_anchored.sh"
