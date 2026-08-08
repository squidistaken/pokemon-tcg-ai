#!/bin/bash
# A/B test: is W&B's console redirect what hangs a ParallelEnv worker mid-run?
#
# WHAT IS BEING TESTED
#
# Twice during the 5M transformer run (8667ld3j) a single worker stopped
# answering while the other 31 sat idle waiting for work. Both sides then burned
# torchrl's 10000-second BATCHED_PIPE_TIMEOUT before anything was reported, so
# two recoverable faults cost 5.6 of the run's 8.9 hours.
#
# The silent worker was NOT dead. The parent checks `is_alive()` on every worker
# roughly a thousand times a second while it waits, and it caught a genuinely
# dead worker in ~60s on a third incident. It never caught these two, and it
# spent 99% of a core spinning throughout (31,745s CPU over 31,912s wall). So
# the worker was alive and wedged, and it left no traceback -- which is what a
# frozen process looks like.
#
# Three candidates were left standing:
#
#   1. Stuck inside the cabt engine.
#   2. Stuck writing a log line. W&B's console="redirect" replaces the process's
#      file descriptors, so all 32 forked workers write through pipes drained by
#      relay threads in the parent. A worker blocks forever if that path stalls.
#   3. Stuck on a lock inherited across fork. The workers are forked from a
#      parent running ~8 W&B threads; fork copies locks in whatever state they
#      were in, and the child jams the first time it touches one.
#
# 2 and 3 are both "W&B plus fork", and the one hang that IS fully diagnosed
# (workers unable to exit, see WandbForkGuard) is also W&B plus fork. This test
# separates them from 1 for the cost of one run.
#
# HOW TO READ THE RESULT
#
# console=wrap keeps W&B logging fully intact for the parent process; it only
# stops W&B from intercepting the workers' file descriptors. Everything else is
# held at the values of the run being reproduced.
#
#   * No worker hang in 5M frames -> candidate 2 or 3. Next step is to keep
#     console=wrap as the default and confirm over a second run.
#   * Hang still occurs -> the console redirect is exonerated; the engine
#     (candidate 1) is where to look next, and the trace to capture is a py-spy
#     dump of the silent worker while it is still wedged.
#
# The hang is roughly one per 1.5M frames, so a 5M-frame run expects ~3. Zero is
# a meaningful result; one is not a refutation on its own.
#
# WHAT THIS RUN NO LONGER COSTS
#
# collector.pipe_timeout is now 180s rather than torchrl's 10000s, so a hang
# that does occur costs ~3 minutes and a pool restart instead of 2h47m. Grep the
# log for "Worker pool died" to count them:
#
#   grep -c "Worker pool died" logs/train-console-wrap-*.log
#
# Expect ~3.5h rather than the 8.9h the original run took.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DECK_DIR="decks"
DECK_CORPUS="heuristic-resolved"
GROUP="worker-hang-console-20260807"
RUN_NAME="tf-pointer-2l-5m-s42-console-wrap"
TOTAL_FRAMES=5000000
SEED=42

if [ ! -d "$DECK_DIR/$DECK_CORPUS" ]; then
  echo "ERROR: '$DECK_DIR/$DECK_CORPUS' not found. Run ./scripts/fetch_decks.sh first." >&2
  exit 1
fi

mkdir -p logs
LOG="logs/train-console-wrap-$(date +%Y%m%dT%H%M%S).log"

echo "A/B arm: wandb.console=wrap (workers' stdout no longer routed through W&B)."
echo "Everything else matches run 8667ld3j. Log: $LOG"

# Identical to scripts/train_transformer_pointer_5m.sh except for
# wandb.console=wrap, which is the single variable under test.
uv run --frozen --no-sync python -m src.train --config-name ppo_selfplay_multideck \
  paths.data_dir=$DECK_DIR \
  deck_corpus=$DECK_CORPUS \
  model/backbone=transformer \
  model.backbone.option_tokens=true \
  model.backbone.num_layers=2 \
  model.backbone.ff_dim=512 \
  model/head=pointer \
  env.curriculum.enabled=true \
  env.curriculum.capacity=1500 \
  env.curriculum.explore_prob=0.3 \
  agent.device=cuda env.num_workers=32 collector.total_frames=$TOTAL_FRAMES \
  train.pool_size=2 \
  train.snapshot_interval=50000 \
  agent.sub_batch_size=1024 \
  agent.frames_per_batch=16384 \
  train.eval_interval=250000 train.eval_episodes=50 train.eval_per_archetype=false \
  seed=$SEED set_seed=true \
  wandb.console=wrap \
  wandb.group=$GROUP \
  wandb.name=$RUN_NAME \
  'wandb.tags=[transformer,pointer-head,5m,worker-hang-ab,console-wrap]' \
  2>&1 | tee "$LOG"

echo "Log: $LOG"
echo "Worker hangs this run: $(grep -c 'Worker pool died' "$LOG" || true)"
