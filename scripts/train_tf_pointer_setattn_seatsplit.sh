#!/bin/bash
# 15M-frame PPO self-play run from scratch on the transformer trunk with the
# pointer head, the seat-split fix, and -- the point of this arm -- attention
# that actually reaches the policy.
#
# NOT THE PRIMARY RUN. scripts/train_mlp_pointer_seatsplit_15m.sh is. This one
# is kept ready for when the GPU is free and the MLP arm's number is in.
#
# WHY THIS ARM EXISTS
#
# Every transformer run in this lineage set option_tokens=true with
# encoded_option_repr=false, which builds option_repr from the adapter's raw
# option rows and never routes them through the encoder. Measured on that exact
# config: perturbing every encoder weight by N(0, 0.5) moved state_repr by 4.0
# and option_repr by exactly 0.0. Attention reached the policy only through the
# shared state vector, identically for every option, so it could not compute
# anything about a *specific* option -- the same role the MLP trunk plays, at
# 2.4x the cost per frame. A flat backbone comparison is what that wiring
# predicts; it was never evidence against attention.
#
# token_groups=[options] + encoded_option_repr=true puts the 129 option rows
# into the sequence and reads option_repr back from the *attended* output, so
# an option's representation depends on the rest of the board and on the other
# options available. That is the first configuration in this repo where a win
# can be credited to attention rather than to option identity alone.
#
# The constructor rejects the half-configured versions of this: options in
# token_groups with option_tokens and encoded_option_repr left false would pay
# the attention cost and then discard the result.
#
# COST
#
# Measured unbatched CPU forward (the path the league opponent runs inside
# every environment worker): 1.90 ms for the 10-token default, 2.88 ms with
# every option routed through attention. Roughly +50% on the forward, not the
# quadratic blowup earlier comments assumed -- but the league opponent forward
# is already ~46% of throughput (docs/training-performance.md), so budget for
# it. Expect meaningfully fewer frames per hour than the MLP arm.
#
# TWO CHANGES AT ONCE
#
# This run varies the seat split and the option-attention wiring together, so a
# flat result will not say which failed. That is accepted deliberately: the
# seat split is a capability fix rather than an arm, and running it alone on
# the transformer would spend the same hours re-measuring what the MLP arm
# already answers more cheaply. If the MLP arm has already banked the seat
# split's effect by the time this runs, the delta here is attributable to the
# attention wiring.
#
# WHAT TO WATCH
#
#   train/explained_variance   the seat fix's earliest signal; see the MLP
#                              script's header.
#   train/grad_norm            norm_first stays false (post-LN, torch's
#                              default) to keep this comparable with the
#                              lineage, and post-LN at 2 layers with no warmup
#                              is a live suspect for tf_combined's monotonic
#                              0.96 -> 4.14 growth into the max_grad_norm 1.0
#                              clip. If it climbs, add
#                              model.backbone.norm_first=true
#                              model.backbone.final_norm=true and rerun -- but
#                              that is then a third change.
#   fps                        against the MLP arm's. If this arm is not ahead
#                              on eval at matched *wall clock*, it loses.
#
# DO NOT LAUNCH WHILE ANOTHER RUN HOLDS THE GPU. Check first:
#
#   pgrep -af "src.train"
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

DECK_DIR="decks"
DECK_CORPUS="heuristic-resolved"
GROUP="seatsplit-20260807"
RUN_NAME="tf-pointer-setattn-seatsplit-15m-s42"
TOTAL_FRAMES=15000000
SEED=42

# Held at the transformer lineage's values (8667ld3j and its continuation), so
# the trunk wiring and the seat split are the only things that move.
NUM_LAYERS=2
FF_DIM=512
POOL_SIZE=2
SNAPSHOT_INTERVAL=50000
SUB_BATCH=1024
FRAMES_PER_BATCH=16384
CURRICULUM_CAPACITY=1500
EXPLORE_PROB=0.3

# The transformer lineage's own conclusion, argued from loss magnitudes in
# train_tf_pointer_continue_10m.sh: at 0.02 the entropy bonus outweighed the
# policy objective for an entire run. The MLP lineage went the other way, to
# 0.05 -- that disagreement is unresolved, see the MLP script's header.
ENTROPY_COEFF=0.005

TRAIN_STATE_INTERVAL=250000
EVAL_INTERVAL=750000
EVAL_EPISODES=200

# The same frozen 5.01M-frame reference the MLP arm uses -- the agent
# tf-pointer-2l-15m-ent005-s43 was initialised from and scored against for its
# whole life, not the partial snapshot its interruption left behind. Holding it
# identical across both arms is what makes their eval curves comparable to each
# other as well as to the lineage. Loaded via the config embedded in its own
# checkpoint, so its different architecture is fine.
BASELINE_CHECKPOINT="outputs/2026-08-07/03-20-11-p30924/checkpoints/snapshot_000005013504.pt"

if [ ! -f "$BASELINE_CHECKPOINT" ]; then
  echo "ERROR: baseline checkpoint '$BASELINE_CHECKPOINT' not found." >&2
  exit 1
fi

# Bracketed first character so the pattern cannot match the shell that
# happens to carry this script's own command line.
if pgrep -f "[p]ython3 -m src.train" > /dev/null; then
  echo "ERROR: another src.train process is running and this run wants the same GPU." >&2
  echo "       Check it with: pgrep -af 'src.train'" >&2
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
  echo "       The corpus looks incomplete -- refusing to burn hours on it." >&2
  exit 1
fi
echo "Training on $DECKS decks across $ARCHETYPES archetypes in $DECK_DIR/$DECK_CORPUS."
echo "Scoring against $BASELINE_CHECKPOINT and random every $EVAL_INTERVAL frames."

mkdir -p logs
LOG="logs/train-tf-pointer-setattn-seatsplit-$(date +%Y%m%dT%H%M%S).log"

uv run --frozen --no-sync python -m src.train --config-name ppo_selfplay_multideck \
  paths.data_dir=$DECK_DIR \
  deck_corpus=$DECK_CORPUS \
  model/backbone=transformer \
  model/head=pointer \
  model.backbone.option_tokens=true \
  'model.backbone.token_groups=[options]' \
  model.backbone.encoded_option_repr=true \
  model.backbone.num_layers=$NUM_LAYERS \
  model.backbone.ff_dim=$FF_DIM \
  model.adapter.pokemon_seat_split=true \
  agent.entropy_coeff=$ENTROPY_COEFF \
  train.pool_size=$POOL_SIZE \
  train.snapshot_interval=$SNAPSHOT_INTERVAL \
  env.curriculum.enabled=true \
  env.curriculum.capacity=$CURRICULUM_CAPACITY \
  env.curriculum.explore_prob=$EXPLORE_PROB \
  agent.device=cuda env.num_workers=32 collector.total_frames=$TOTAL_FRAMES \
  agent.sub_batch_size=$SUB_BATCH \
  agent.frames_per_batch=$FRAMES_PER_BATCH \
  train.train_state_interval=$TRAIN_STATE_INTERVAL \
  train.eval_interval=$EVAL_INTERVAL \
  train.eval_episodes=$EVAL_EPISODES \
  train.eval_per_archetype=false \
  "train.eval_opponents=[$BASELINE_CHECKPOINT,random]" \
  seed=$SEED set_seed=true \
  wandb.group=$GROUP \
  wandb.name=$RUN_NAME \
  'wandb.tags=[transformer,pointer-head,option-attention,seat-split,15m,from-scratch]' \
  2>&1 | tee "$LOG"

echo "Log: $LOG"
