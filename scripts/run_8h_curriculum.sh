#!/bin/bash
# 8-hour PLR-curriculum self-play run over the current deck corpus.
#
#   ./scripts/run_8h_curriculum.sh          # seed 0
#   SEED=1 ./scripts/run_8h_curriculum.sh   # another seed, own run dir
#
# Reproduces the hyperparameters of the 2M-frame reference run (W&B 7yvq9zy3)
# and only scales what the longer budget forces:
#
#   total_frames      2M -> 22M. At the reference's measured 985 fps collecting
#                     plus ~55 s per evaluation, this lands at ~7 h 30 m, which
#                     leaves headroom for the new corpus stepping slower.
#   snapshot_interval 50k -> 200k. pool_size is 5, so at 50k the self-play
#                     league only ever holds the last 250k frames -- 1.1% of a
#                     22M run, i.e. eight hours spent playing near-copies of the
#                     current policy. At 200k the league spans 1M frames, PFSP's
#                     per-member win rates get long enough to mature, and the
#                     checkpoint directory stays around 110 files / 220 MB.
#   eval_interval     100k -> 250k, eval_episodes 20 -> 40. Measurement only, no
#                     effect on learning: 88 eval points instead of 220, each
#                     averaged over twice the episodes, for the same share of
#                     the budget. The reference's 20-episode points swung 0.42
#                     to 0.60 between neighbours.
#
# Everything else is untouched: lr 3e-4 (no annealing), num_epochs 4,
# sub_batch_size 4096, clip_epsilon 0.2, entropy_coeff 0.01, gamma 0.99,
# lmbda 0.95, pool_size 5, pfsp sampling, 32 workers.
#
# There is no end-of-run model save -- the artifacts are the periodic self-play
# snapshots under the run's checkpoints/. Interrupting at the eight-hour mark is
# safe and costs at most snapshot_interval frames.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEED="${SEED:-0}"
TOTAL_FRAMES="${TOTAL_FRAMES:-22000000}"
GROUP="${GROUP:-decks-mar2026-8h}"
RUN_DIR="${RUN_DIR:-outputs/curriculum_8h_s${SEED}}"

if [ -e "$RUN_DIR" ]; then
  echo "ERROR: $RUN_DIR already exists; set RUN_DIR to a fresh path." >&2
  exit 1
fi

echo "Deck corpus: $(find decks -name '*.csv' ! -name 'example.csv' | wc -l) decks"
echo "Run dir:     $RUN_DIR"
echo "Frames:      ${TOTAL_FRAMES} (seed ${SEED}, W&B group ${GROUP})"

exec python -m src.train \
  agent=ppo env=curriculum_v2 train=ppo_selfplay callbacks=wandb \
  seed="${SEED}" set_seed=true \
  agent.device=cuda \
  env.num_workers=32 \
  collector.total_frames="${TOTAL_FRAMES}" \
  train.snapshot_interval=200000 \
  train.eval_interval=250000 \
  train.eval_episodes=40 \
  wandb.group="${GROUP}" \
  wandb.name="curriculum-8h-s${SEED}" \
  wandb.job_type=curriculum \
  hydra.run.dir="${RUN_DIR}"
