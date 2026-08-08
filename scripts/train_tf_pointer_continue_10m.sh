#!/bin/bash
# Continue tf-pointer-2l-5m-s42-pool2 (W&B 8667ld3j) from its final 5.01M-frame
# checkpoint for another 10M frames, with the entropy bonus cut 0.02 -> 0.005.
# Ends at ~15M cumulative.
#
# WHY THIS RUN EXISTS
#
# The 5M run it continues improved against a random reference (eval/random 0.62
# -> ~0.90) but never pulled away from its own 50k-frame snapshot:
# eval/first_snapshot read 0.80 at 262k frames and 0.84 at 4.98M, with every
# wobble inside the +/-7.1pt standard error a 50-episode eval carries. Over 4.7M
# frames the policy did not measurably beat a much earlier copy of itself.
#
# The diagnosis is the entropy bonus, and it is quantitative rather than a
# hunch. Comparing the two terms the optimizer actually balances:
#
#   frames      loss_objective   loss_entropy   |entropy| / |objective|
#   0-1M          -0.0036          -0.0242            6.8x
#   2-3M          -0.0062          -0.0240            3.9x
#   4-5M          -0.0098          -0.0243            2.5x
#   5-6M          -0.0114          -0.0234            2.1x
#
# The bonus outweighed the policy objective for the entire run. Entropy fell to
# 1.08 nats around 1-1.5M and then *rose back* to 1.22 and parked there -- the
# regularizer was actively preventing the policy from committing. Halving the
# coefficient to 0.01 would only bring the two terms to parity (0.0117 vs
# 0.0114), which is why this run goes to 0.005: that puts the entropy term at
# roughly 0.5x the objective, the first time in this lineage that the policy
# gradient dominates. Verified in a smoke run at these settings:
# loss_entropy -0.0059 against loss_objective -0.0146, ratio 0.40.
#
# It also compounds in the right direction. The term is coeff x entropy, so as
# entropy falls the ratio keeps improving on its own, while loss_objective has
# been growing independently (0.0036 -> 0.0114 over the previous run).
#
# Not 0.0025. With no annealing, target_kl 0.03 is the only brake, and the
# compounding means 0.005 already has teeth. Watch for entropy diving under
# ~0.5 while eval stalls -- that is over-collapse, and the signal to stop rather
# than cut further.
#
# WHAT IS DELIBERATELY UNCHANGED
#
# entropy_coeff is the only hyperparameter that moves, so a flat result is
# attributable to it. In particular pool_size stays at 2 even though the memory
# fix has landed and 5 is now affordable: raising it changes the opponent
# distribution at the same time, and then neither the entropy change nor the
# league change could be read from the outcome. That is the trap the eight-arm
# sweep fell into. Restoring the league is its own run.
#
# Corpus, curriculum settings, lr 3e-4 with no annealing, gamma/lmbda,
# target_kl, batching and the 20% deck holdout are all inherited unchanged from
# the run being continued.
#
# WHAT CONTINUING ACTUALLY RESTORES
#
#   weights      train.init_checkpoint, strict load
#   curriculum   env.curriculum.init_state -- 259 scored matchups, visit counts
#                and win/loss tallies, episode counter at 67,551. Without this
#                the sampler rediscovers the corpus from nothing.
#   optimizer    NOT restored. The 5.01M snapshot predates train_state.pt and
#                carries weights only, so Adam's moments start from zero and the
#                first rollouts are noisier than the ones before them. Do not
#                read the first evaluation round. This run writes train_state.pt
#                every 250k frames, so the *next* continuation resumes properly.
#   league       Starts empty and refills from the warm-started weights, which
#                is better than a cold league but is not the previous run's.
#
# THE EVALUATION REFERENCE
#
# eval_opponents drops first_snapshot -- flat for 4.7M frames, it answered
# nothing -- and scores against the 5.01M checkpoint itself. That metric,
# snapshot_000005013504/win_rate, is literally "am I better than the agent
# currently on the Kaggle leaderboard", which is the question this run is being
# run to answer. random is kept because it is the one reference comparable
# across every run in this lineage; it is near saturation at ~0.90 and will not
# say much more on its own.
#
# WHAT TO WATCH
#
#   train/entropy                      should now fall below ~1.0 and keep
#                                      going. Parking at 1.2 again means
#                                      entropy was not the constraint, and the
#                                      next suspect is the league, not frames.
#   snapshot_000005013504/win_rate     above 0.5 and climbing is the result.
#   train/grad_norm                    sat at 0.46-0.62 against max_grad_norm
#                                      1.0. A sharper policy may push it up, and
#                                      clipping would change what is measured.
#
# Expect ~22h at the previous run's 126 fps -- long enough that the periodic
# train_state.pt below is the difference between a crash costing an hour and
# costing the run. Launch inside tmux/screen so it survives a disconnect:
#
#   ./scripts/train_tf_pointer_continue_10m.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# The repo's own modules (submission/, src/, main.py) collide with same-named
# top-level modules on an inherited PYTHONPATH, so the run gets a clean one.
export PYTHONPATH=

# GAE runs the critic over the whole collected batch in one forward and the
# epoch loop clones the batch once per epoch, so the allocator sees a handful of
# multi-GB transients against steady small ones -- the fragmentation case
# expandable_segments exists for.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PREV_RUN="outputs/2026-08-07/03-20-11-p30924"
INIT_CHECKPOINT="$PREV_RUN/checkpoints/snapshot_000005013504.pt"
INIT_CURRICULUM="$PREV_RUN/curriculum/curriculum_000005013504.pt"

DECK_DIR="decks"
DECK_CORPUS="heuristic-resolved"
GROUP="tf-pointer-continue-20260807"
RUN_NAME="tf-pointer-2l-15m-ent005-s43"
TOTAL_FRAMES=10000000

# Seed 43, not the 42 the previous run used. The env's deck, seat and opponent
# streams are seeded from this, so reusing 42 would deal the continued run the
# same matchup sequence it has already played 5M frames of.
SEED=43

# The change under test. See the loss-magnitude table above.
ENTROPY_COEFF=0.005

# Held at the previous run's values on purpose -- see "WHAT IS DELIBERATELY
# UNCHANGED".
POOL_SIZE=2
SNAPSHOT_INTERVAL=50000
SUB_BATCH=1024
FRAMES_PER_BATCH=16384
CURRICULUM_CAPACITY=1500
EXPLORE_PROB=0.3

# 250k frames between optimizer-state writes: ~40 writes over the run, each
# roughly 3x a weights-only snapshot, into one rolling file rather than a
# numbered series.
TRAIN_STATE_INTERVAL=250000

EVAL_INTERVAL=250000
EVAL_EPISODES=50

for required in "$INIT_CHECKPOINT" "$INIT_CURRICULUM"; do
  if [ ! -f "$required" ]; then
    echo "ERROR: '$required' not found; this run continues from it." >&2
    exit 1
  fi
done

if [ ! -d "$DECK_DIR/$DECK_CORPUS" ]; then
  echo "ERROR: '$DECK_DIR/$DECK_CORPUS' not found. Run ./scripts/fetch_decks.sh first." >&2
  exit 1
fi

# The curriculum state is addressed by archetype position, so it is only
# meaningful against the corpus it was built over. load_state re-checks this and
# refuses a mismatch, but failing here costs seconds instead of a model build.
ARCHETYPES=$(find "$DECK_DIR/$DECK_CORPUS" -mindepth 1 -maxdepth 1 -type d | wc -l)
DECKS=$(find "$DECK_DIR/$DECK_CORPUS" -name '*.csv' | wc -l)
if [ "$ARCHETYPES" -lt 100 ]; then
  echo "ERROR: only $ARCHETYPES archetypes in '$DECK_DIR/$DECK_CORPUS'; expected ~136." >&2
  echo "       The corpus looks incomplete -- refusing to burn 22h on it." >&2
  exit 1
fi
echo "Training on $DECKS decks across $ARCHETYPES archetypes in $DECK_DIR/$DECK_CORPUS."

mkdir -p logs
LOG="logs/train-tf-pointer-continue-$(date +%Y%m%dT%H%M%S).log"

echo "Continuing from $INIT_CHECKPOINT (5,013,504 frames)."
echo "Entropy coefficient $ENTROPY_COEFF (was 0.02); everything else held."
echo "Scoring against that checkpoint and random every $EVAL_INTERVAL frames."

uv run --frozen --no-sync python -m src.train --config-name ppo_selfplay_multideck \
  paths.data_dir=$DECK_DIR \
  deck_corpus=$DECK_CORPUS \
  model/backbone=transformer \
  model.backbone.option_tokens=true \
  model.backbone.num_layers=2 \
  model.backbone.ff_dim=512 \
  model/head=pointer \
  train.init_checkpoint=$INIT_CHECKPOINT \
  env.curriculum.init_state=$INIT_CURRICULUM \
  env.curriculum.enabled=true \
  env.curriculum.capacity=$CURRICULUM_CAPACITY \
  env.curriculum.explore_prob=$EXPLORE_PROB \
  agent.entropy_coeff=$ENTROPY_COEFF \
  agent.device=cuda env.num_workers=32 collector.total_frames=$TOTAL_FRAMES \
  train.pool_size=$POOL_SIZE \
  train.snapshot_interval=$SNAPSHOT_INTERVAL \
  train.train_state_interval=$TRAIN_STATE_INTERVAL \
  agent.sub_batch_size=$SUB_BATCH \
  agent.frames_per_batch=$FRAMES_PER_BATCH \
  "train.eval_opponents=[$INIT_CHECKPOINT,random]" \
  train.eval_interval=$EVAL_INTERVAL \
  train.eval_episodes=$EVAL_EPISODES \
  train.eval_per_archetype=false \
  seed=$SEED set_seed=true \
  wandb.group=$GROUP \
  wandb.name=$RUN_NAME \
  'wandb.tags=[transformer,pointer-head,option-tokens,continue,entropy-005,15m]' \
  2>&1 | tee "$LOG"

echo "Log: $LOG"
