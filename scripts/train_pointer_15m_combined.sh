#!/bin/bash
# 15M-frame PPO self-play run on decks/heuristic-resolved with the pointer head,
# applying every change diagnosed from run pyo9oiis in one go.
#
# pyo9oiis (5M frames, same corpus, same head) stopped improving at roughly
# 750k frames. Its win rate against the fixed `random` reference averaged 0.828
# over the first four evaluation rounds and 0.824 over the last four, a slope
# indistinguishable from zero against a per-eval standard error of 0.028. The
# peak, 0.885, was set at 500k frames and never beaten across the following
# 4.5M. So roughly 85% of that run's wall clock bought nothing measurable, and
# simply extending the frame budget under the same configuration would have
# reproduced the plateau at three times the cost.
#
# Four changes, applied together. They address independent suspected causes, so
# a flat result here does not identify which one failed -- that was the
# tradeoff accepted when choosing a single combined run over a per-arm sweep.
#
#   1. train.pool_size 5 -> 20. The league is a strict FIFO over the newest
#      snapshots (snapshot_opponent_pool.py:101). At pool_size 5 and
#      snapshot_interval 50000 it spanned only the last 250k frames, so the
#      agent trained almost entirely against near-copies of its recent self and
#      strategies beaten a million frames earlier were free to cycle back.
#      Training win rate pinned at 0.492 for the whole run is that signature.
#      Twenty snapshots span 1M frames and also give opponent_sampling=pfsp a
#      league diverse enough for its hard weighting to select within.
#   2. Capacity: embed_dim 128 -> 256, backbone [256,256] -> [512,512],
#      adapter.card_embed_dim 8 -> 16. Eight dimensions per card across a
#      28,670-list corpus is thin, and a plateau that arrives early and then
#      sits flat rather than creeping upward is the shape of a representational
#      ceiling, which no additional frames can lift.
#   3. agent.entropy_coeff 0.02 -> 0.05. Policy entropy fell from 1.30 to 1.04
#      nats inside the first 10% of the run and then held at ~1.05 for the
#      remaining 90% -- about 2.9 effective actions against legal sets reaching
#      ~42 options. Entropy froze at the same point the win rate did. A fixed
#      bonus settles where the entropy and policy gradients balance; raising it
#      moves that equilibrium, and it acts at 750k where the plateau forms
#      rather than only at the end of the run.
#
# The curriculum is held at pyo9oiis's settings on purpose. It changes which
# matchups the agent sees, so switching it off would alter the training
# distribution alongside the three changes above and confound all of them.
#
# Evaluation interval is raised 250k -> 750k. Evaluation plays one episode at a
# time in a single environment while collection runs 32 workers, so it is pure
# wall clock -- see docs/evaluation-cost.md. At the old interval a 15M run would
# spend ~60 rounds x 400 episodes on it. At 750k it keeps the same 20 curve
# points pyo9oiis had, for about an hour total.
#
# Expect ~10h: the wider network collects more slowly than the 546 fps pyo9oiis
# averaged. Launch inside tmux/screen so it survives a disconnect:
#
#   ./scripts/train_pointer_15m_combined.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# The repo's own modules (submission/, src/, main.py) collide with same-named
# top-level modules on an inherited PYTHONPATH, so the run gets a clean one.
export PYTHONPATH=

DECK_DIR="decks"
DECK_CORPUS="heuristic-resolved"
GROUP="pointer-15m-combined-20260806"
RUN_NAME="ppo-pointer-15m-combined-s42"
TOTAL_FRAMES=15000000
SEED=42

# Held at pyo9oiis's values so the curriculum is not a variable in this run.
CURRICULUM=true
CURRICULUM_CAPACITY=1500
EXPLORE_PROB=0.3

# pyo9oiis's final agent (5,013,504 frames, key 473ff864bf18) as the evaluation
# baseline in place of first_snapshot. first_snapshot is this run's own untrained
# starting point, so beating it says only that training happened; the previous
# run's finished agent is the bar this one has to clear to be worth submitting.
# It has a different architecture (embed_dim 128, backbone [256,256],
# card_embed_dim 8), which load_actor_critic handles by rebuilding it from the
# config embedded in its own checkpoint.
BASELINE_CHECKPOINT="outputs/2026-08-06/12-36-09/checkpoints/snapshot_000005013504.pt"

if [ ! -f "$BASELINE_CHECKPOINT" ]; then
  echo "ERROR: baseline checkpoint '$BASELINE_CHECKPOINT' not found." >&2
  exit 1
fi

if [ ! -d "$DECK_DIR/$DECK_CORPUS" ]; then
  echo "ERROR: '$DECK_DIR/$DECK_CORPUS' not found. Run ./scripts/fetch_decks.sh first." >&2
  exit 1
fi

ARCHETYPES=$(find "$DECK_DIR/$DECK_CORPUS" -mindepth 1 -maxdepth 1 -type d | wc -l)
DECKS=$(find "$DECK_DIR/$DECK_CORPUS" -name '*.csv' | wc -l)
if [ "$ARCHETYPES" -lt 100 ]; then
  echo "ERROR: only $ARCHETYPES archetypes in '$DECK_DIR/$DECK_CORPUS'; expected ~136." >&2
  echo "       The corpus looks incomplete -- refusing to burn 10h on it." >&2
  exit 1
fi
echo "Training on $DECKS decks across $ARCHETYPES archetypes in $DECK_DIR/$DECK_CORPUS."

mkdir -p logs
LOG="logs/train-pointer-15m-combined-$(date +%Y%m%dT%H%M%S).log"

uv run --frozen --no-sync python -m src.train --config-name ppo_selfplay_multideck \
  paths.data_dir=$DECK_DIR \
  deck_corpus=$DECK_CORPUS \
  model/head=pointer \
  model.embed_dim=256 \
  'model.backbone.num_cells=[512,512]' \
  model.adapter.card_embed_dim=16 \
  agent.entropy_coeff=0.05 \
  train.pool_size=20 \
  env.curriculum.enabled=$CURRICULUM \
  env.curriculum.capacity=$CURRICULUM_CAPACITY \
  env.curriculum.explore_prob=$EXPLORE_PROB \
  agent.device=cuda env.num_workers=32 collector.total_frames=$TOTAL_FRAMES \
  train.eval_interval=750000 train.eval_episodes=200 train.eval_per_archetype=false \
  "train.eval_opponents=[$BASELINE_CHECKPOINT,random]" \
  seed=$SEED set_seed=true \
  wandb.group=$GROUP \
  wandb.name=$RUN_NAME \
  'wandb.tags=[pointer-head,heuristic-resolved,15m,combined-fix]' \
  2>&1 | tee "$LOG"

echo "Log: $LOG"
