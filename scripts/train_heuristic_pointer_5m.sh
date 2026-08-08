#!/bin/bash
# 5M-frame PPO self-play run on decks/heuristic-resolved with the pointer head.
#
# This is a rerun of the heuristic-resolved arm of
# scripts/train_deck_corpus_comparison.sh, so its result is directly comparable
# to W&B run ci4ddgzv (group deck-corpus-comparison-20260805). The training
# configuration is deliberately identical to that run except for three things:
#
#   1. model/head=pointer  -- the fix. The previous run used the flat head,
#      which reads the option table only through a permutation-invariant mean
#      and can therefore represent nothing beyond a prior over slot indices.
#      See docs/architecture/pointer-head.md.
#   2. eval_episodes 40 -> 200. Forty episodes is +/-7.9pp standard error, which
#      is wider than any effect worth seeing; the previous run's eval "curve"
#      was mostly noise. Evaluation does not feed back into training, so
#      raising it costs wall clock and nothing else.
#   3. decks/heuristic-resolved instead of the frozen snapshot, as requested.
#      Checked below for the archetype count the run expects, since ./decks/ is
#      the scraper's output directory and can change between runs.
#
# Everything else (optimizer, curriculum, league, seed) is held fixed on
# purpose: the architecture is the variable under test, and this run only
# answers the architecture question if nothing else moves with it.
#
# Runs in the foreground for ~2h. Launch inside tmux/screen so it survives a
# disconnect:
#
#   ./scripts/train_heuristic_pointer_5m.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DECK_DIR="decks"
DECK_CORPUS="heuristic-resolved"
GROUP="pointer-head-5m-20260806"
RUN_NAME="ppo-pointer-heuristic-resolved-s42"
TOTAL_FRAMES=5000000
SEED=42

# Bounded PLR working set over archetype matchups, unchanged from the run this
# reproduces. Note the honest caveat: with ~136 archetypes the level space is
# ~18k matchups against a ~48k-episode budget, and the previous run matured only
# 79 levels while 11,780 sat in probation -- the curriculum was close to inert.
# It is kept on for comparability, not because it was earning its keep. Set
# CURRICULUM=false to get the clean no-curriculum baseline instead.
CURRICULUM=true
CURRICULUM_CAPACITY=1500
EXPLORE_PROB=0.3

# 200 episodes per opponent (train.eval_opponents = [first_snapshot, random]),
# so 400 evaluation episodes every 250k frames, 20 rounds over the run. The
# evaluator plays one episode at a time in a single env while collection uses
# 32 workers, so this is the dominant non-collection cost -- see
# docs/training-performance.md section 6 before raising either number.
EVAL_ARGS="train.eval_interval=250000 train.eval_episodes=200 train.eval_per_archetype=false"

if [ ! -d "$DECK_DIR/$DECK_CORPUS" ]; then
  echo "ERROR: '$DECK_DIR/$DECK_CORPUS' not found. Run ./scripts/fetch_decks.sh first." >&2
  exit 1
fi

ARCHETYPES=$(find "$DECK_DIR/$DECK_CORPUS" -mindepth 1 -maxdepth 1 -type d | wc -l)
DECKS=$(find "$DECK_DIR/$DECK_CORPUS" -name '*.csv' | wc -l)
if [ "$ARCHETYPES" -lt 100 ]; then
  echo "ERROR: only $ARCHETYPES archetypes in '$DECK_DIR/$DECK_CORPUS'; expected ~136." >&2
  echo "       The corpus looks incomplete -- refusing to burn 2h on it." >&2
  exit 1
fi
echo "Training on $DECKS decks across $ARCHETYPES archetypes in $DECK_DIR/$DECK_CORPUS."

mkdir -p logs
LOG="logs/train-pointer-$DECK_CORPUS-$(date +%Y%m%dT%H%M%S).log"

uv run --frozen --no-sync python -m src.train --config-name ppo_selfplay_multideck \
  paths.data_dir=$DECK_DIR \
  deck_corpus=$DECK_CORPUS \
  model/head=pointer \
  agent.device=cuda env.num_workers=32 collector.total_frames=$TOTAL_FRAMES \
  env.curriculum.enabled=$CURRICULUM \
  env.curriculum.capacity=$CURRICULUM_CAPACITY \
  env.curriculum.explore_prob=$EXPLORE_PROB \
  $EVAL_ARGS \
  seed=$SEED set_seed=true \
  wandb.group=$GROUP \
  wandb.name=$RUN_NAME \
  'wandb.tags=[pointer-head,heuristic-resolved,5m]' \
  2>&1 | tee "$LOG"

echo "Log: $LOG"
