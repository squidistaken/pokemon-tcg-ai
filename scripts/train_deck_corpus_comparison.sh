#!/bin/bash
# Compare the two card-swap resolution strategies by training PPO self-play on
# each corpus in turn: decks-intermediate-20260805/heuristic-resolved, then
# decks-intermediate-20260805/mapping-resolved.
#
# Pinned to that frozen snapshot (not ./decks/) on purpose: the scraper may
# still be running and appending to ./decks/, and a run needs a fixed corpus
# for its duration rather than one that grows underneath it mid-training.
#
# Runs both sequentially in the foreground, not concurrently:
# env.num_workers=32 already oversubscribes this machine's 8 physical cores,
# and there is only one GPU for the PPO update. Launch this yourself inside
# your own tmux/screen session so it survives a disconnect.
#
#   ./scripts/train_deck_corpus_comparison.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DECK_SNAPSHOT="decks-intermediate-20260805"
GROUP="deck-corpus-comparison-20260805"
TOTAL_FRAMES=5000000
# Each corpus has ~135 archetypes (~18k matchups), far more than a 5M-frame
# run's ~38k-episode budget could ever cover if every matchup had to be
# prefilled and hit min_visits. CURRICULUM_CAPACITY is instead a bounded
# working set: PLR scores this many matchups at once, and EXPLORE_PROB is the
# per-episode chance of sampling a fresh, never-scored matchup from the full
# corpus so ones outside the working set keep getting discovered throughout
# the run, evicting the weakest scored entry to make room (see
# src/env/level_buffer.py's commit()/prefill() docstrings).
CURRICULUM_CAPACITY=1500
EXPLORE_PROB=0.3
# Without these the run inherits eval_interval=50000 (conf/train/ppo_selfplay.yaml)
# and eval_episodes=150 (conf/ppo_selfplay_multideck.yaml), which over two
# eval_opponents is 300 evaluation episodes per 50k frames. The evaluator plays
# episodes one at a time in a single env while collection runs 32 workers, so
# that puts ~78% of wall clock into evaluation and reports ~213 fps against the
# ~962 it is actually collecting at. See docs/training-performance.md section 6.
# Per-archetype is off because 40 episodes over ~48 archetypes is under one game
# each; the corpus comparison is settled by a head-to-head between the final
# checkpoints, not by the training-time eval curve.
EVAL_ARGS="train.eval_interval=250000 train.eval_episodes=40 train.eval_per_archetype=false"

if [ ! -d "$DECK_SNAPSHOT" ]; then
  echo "ERROR: '$DECK_SNAPSHOT' not found; this script trains against that frozen snapshot specifically." >&2
  exit 1
fi

mkdir -p logs

uv run --frozen --no-sync python -m src.train --config-name ppo_selfplay_multideck \
  paths.data_dir=$DECK_SNAPSHOT \
  deck_corpus=heuristic-resolved \
  agent.device=cuda env.num_workers=32 collector.total_frames=$TOTAL_FRAMES \
  env.curriculum.enabled=true env.curriculum.capacity=$CURRICULUM_CAPACITY env.curriculum.explore_prob=$EXPLORE_PROB \
  $EVAL_ARGS \
  seed=42 set_seed=true \
  wandb.group=$GROUP \
  wandb.name=ppo-mlp-heuristic-resolved-s42 \
  'wandb.tags=[deck-corpus-comparison,heuristic-resolved]' \
  2>&1 | tee "logs/train-heuristic-resolved-$(date +%Y%m%dT%H%M%S).log"

uv run --frozen --no-sync python -m src.train --config-name ppo_selfplay_multideck \
  paths.data_dir=$DECK_SNAPSHOT \
  deck_corpus=mapping-resolved \
  agent.device=cuda env.num_workers=32 collector.total_frames=$TOTAL_FRAMES \
  env.curriculum.enabled=true env.curriculum.capacity=$CURRICULUM_CAPACITY env.curriculum.explore_prob=$EXPLORE_PROB \
  $EVAL_ARGS \
  seed=42 set_seed=true \
  wandb.group=$GROUP \
  wandb.name=ppo-mlp-mapping-resolved-s42 \
  'wandb.tags=[deck-corpus-comparison,mapping-resolved]' \
  2>&1 | tee "logs/train-mapping-resolved-$(date +%Y%m%dT%H%M%S).log"
