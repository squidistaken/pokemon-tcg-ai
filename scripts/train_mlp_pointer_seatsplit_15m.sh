#!/bin/bash
# 15M-frame PPO self-play run from scratch on the MLP trunk with the pointer
# head, adding the seat-split observation fix. This is the primary run.
#
# WHY FROM SCRATCH
#
# adapter.pokemon_seat_split widens the `pokemon` group 192 -> 384, which
# resizes the trunk's first projection. No earlier checkpoint can be loaded
# against it, so warm-starting is not an option and the strict load would fail
# loudly rather than silently mismatch.
#
# THE FIX
#
# `pokemon` is the only observation group holding both players' rows (agent
# active + bench, then opponent active + bench). It was collapsed by one masked
# pool over all 18 rows, so the pooled path was seat-blind: swapping the two
# boards left state_repr bit-identical, measured at 0.000e+00. Every backbone
# reads the board only through that vector, and the critic reads *nothing* but
# state_repr -- it was being asked to predict win probability from a summary
# that cannot say whose Pokemon is whose. The entity-token path already carried
# seat identity via group_segment_ids, but token_groups is empty in every run
# to date, so it never applied.
#
# The split pools the two halves separately and concatenates. Pooling stays
# *within* a seat, so bench order remains irrelevant -- the property the single
# pool had, and the one the split must not cost.
#
# WHY THE MLP TRUNK, NOT THE TRANSFORMER
#
# Measured on the exact config of the transformer lineage (8667ld3j and its
# continuation): perturbing every encoder weight by N(0, 0.5) moved state_repr
# by 4.0 and option_repr by exactly 0.0. With encoded_option_repr false, the
# per-option tokens never enter the encoder, so attention reaches the policy
# only through the shared state vector -- identically for every option. It
# cannot compute anything about a *specific* option, which is the same role the
# MLP trunk plays at 2.4x the cost per frame. transformer-diagnosis-45 §3 could
# not distinguish the two backbones per-frame either, and at matched wall clock
# the MLP gets 2.4x the frames.
#
# scripts/train_tf_pointer_setattn_seatsplit.sh is the transformer arm that
# fixes that wiring, kept ready but not the primary run.
#
# WHAT IS DELIBERATELY UNCHANGED
#
# Everything else is held at scripts/train_pointer_15m_combined.sh's values --
# capacity, entropy, curriculum, corpus, optimizer, eval cadence -- so the seat
# split is the single learning-relevant change against that run and a flat
# result is attributable to it.
#
# Two deliberate second changes, both representational capacity the model was
# measurably missing:
#
#   option_target_state  puts the targeted Pokemon's live state (HP, attached
#     energy/tool counts, is-active) on each option's own row. target_id already
#     said *which card* an option targets, but a card ID is shared by every copy
#     of it, so two options over two copies of the same Pokemon differed only in
#     a raw index scalar -- and no feedforward network can use that index to
#     look the Pokemon up in the `pokemon` table. Measured on real battles:
#     30.2% of same-target-card option pairs differ in this state and were
#     previously unrankable. The remaining 67.8% were verified genuinely
#     interchangeable -- 0 of 15,949 differed in attached energy types, energy
#     card IDs or tool IDs -- so this block is sufficient, not a partial fix.
#
#   ENTITY_DIM 64 -> 128 (see below). It is bundled
# rather than run as its own arm because both changes are representational
# capacity the model was measurably missing, and splitting them costs a second
# 5-hour run to resolve a question neither result would be ambiguous about --
# the seat split shows up in explained_variance, the entity width in the
# pointer head's ability to separate options. If the run comes out flat on
# both, that is the case for splitting them.
#
# Two operational deviations, neither of which changes what the agent learns:
# train_state_interval writes optimizer state periodically so a crash costs an
# hour instead of the run, and the league buys its 1M-frame span through
# snapshot_interval rather than pool_size (see POOL_SIZE below -- the reference
# run's setting exhausted memory and never finished).
#
# NEW METRICS TO WATCH
#
# The trainer now reports the PPO diagnostics it previously discarded. The one
# this run exists for:
#
#   train/explained_variance   whether the critic predicts returns at all. If
#                              the seat fix works, this is where it shows first
#                              and earliest -- well before eval win-rate moves.
#                              Near 0 means the critic is no better than
#                              predicting the mean.
#   train/clip_fraction        share of the batch the surrogate is clipping.
#   train/ESS                  importance-weight degeneracy across the epochs.
#   train/kl_approx            drift off-policy; target_kl stops the epoch loop
#                              at 1.5 x 0.03.
#
# DO NOT LAUNCH WHILE ANOTHER RUN HOLDS THE GPU. Check first:
#
#   pgrep -af "src.train"
#
# Launch inside tmux/screen so it survives a disconnect:
#
#   ./scripts/train_mlp_pointer_seatsplit_15m.sh
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
RUN_NAME="mlp-pointer-seatsplit-15m-s42"
TOTAL_FRAMES=15000000
SEED=42

# Held at train_pointer_15m_combined.sh's values; see "WHAT IS DELIBERATELY
# UNCHANGED".
EMBED_DIM=256
BACKBONE_CELLS="[512,512]"
CARD_EMBED_DIM=16

# Width of one per-entity encoding, and so of the option tokens the pointer
# head scores. A raw option row is 163 features; at the shipped 64 it was
# compressed by 2.5x before the head ever saw it, and the head's whole job is
# telling options apart. Measured on real battles: the *raw* rows never collide
# (0 in 10,779 rows across 1,800 decisions), so the information is there to
# preserve -- the 64-dim projection was the only place it could be lost.
# Costs 1.75M -> 2.81M parameters.
ENTITY_DIM=128
CURRICULUM_CAPACITY=1500
EXPLORE_PROB=0.3

# League diversity is pool_size * snapshot_interval, and it is bought here the
# cheap way. train_pointer_15m_combined.sh raised pool_size 5 -> 20 at the
# inherited 50k interval to span 1M frames instead of 250k; the reasoning was
# right but the mechanism was the expensive one. Every worker's
# SnapshotOpponentPool keeps pool_size networks *loaded in its own process*
# (snapshot_opponent_pool.py::_refresh), so at 32 workers and a 6.7 MB snapshot
# that is 20 x 6.7 MB x 32 = 4.3 GB of league weights on a 23 GB box, on top of
# 32 torch worker processes.
#
# Both runs that tried it died early and neither wrote a clean shutdown line:
# outputs/2026-08-06/16-47-15 at 2.5% and outputs/2026-08-06/17-01-22 at 21%
# (3.08M/15M). Every run at pool_size <= 5 completed.
#
# 5 x 200k spans the same 1M frames at 1.07 GB -- a quarter of the memory for
# the same league. This is what conf/ppo_transformer.yaml already does, for the
# same stated reason.
POOL_SIZE=5
SNAPSHOT_INTERVAL=200000

# The two lineages disagree here and the disagreement is unresolved, so this is
# the first knob to try if the run stalls with entropy pinned:
#   MLP lineage (train_pointer_15m_combined.sh) raised 0.02 -> 0.05, arguing the
#     policy froze at ~1.05 nats when the win rate froze.
#   Transformer lineage (train_tf_pointer_continue_10m.sh) cut 0.02 -> 0.005,
#     arguing from loss magnitudes that the bonus outweighed the policy
#     objective for an entire run.
# 0.05 keeps the seat split as the single delta against the MLP reference.
ENTROPY_COEFF=0.05

# 250k frames between optimizer-state writes: one rolling file, so an
# interrupted run continues without restarting Adam's moments.
TRAIN_STATE_INTERVAL=250000

EVAL_INTERVAL=750000
EVAL_EPISODES=200

# The frozen reference, deliberately *not* whatever the interrupted
# tf-pointer-2l-15m-ent005-s43 wrote on its way out. This is the 5.01M-frame
# agent that run was initialised from and scored against for its whole life, so
# the new run's eval/snapshot_000005013504/win_rate series is directly
# comparable with that one's rather than being measured against a moving and
# arbitrarily-truncated target.
#
# It has a different architecture (transformer trunk, no seat split), which
# load_actor_critic handles by rebuilding it from the config embedded in its own
# checkpoint rather than from this run's.
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
LOG="logs/train-mlp-pointer-seatsplit-$(date +%Y%m%dT%H%M%S).log"

uv run --frozen --no-sync python -m src.train --config-name ppo_selfplay_multideck \
  paths.data_dir=$DECK_DIR \
  deck_corpus=$DECK_CORPUS \
  model/head=pointer \
  model.embed_dim=$EMBED_DIM \
  "model.backbone.num_cells=$BACKBONE_CELLS" \
  model.adapter.card_embed_dim=$CARD_EMBED_DIM \
  model.adapter.entity_dim=$ENTITY_DIM \
  model.adapter.option_target_state=true \
  model.adapter.pokemon_seat_split=true \
  agent.entropy_coeff=$ENTROPY_COEFF \
  train.pool_size=$POOL_SIZE \
  train.snapshot_interval=$SNAPSHOT_INTERVAL \
  env.curriculum.enabled=true \
  env.curriculum.capacity=$CURRICULUM_CAPACITY \
  env.curriculum.explore_prob=$EXPLORE_PROB \
  agent.device=cuda env.num_workers=32 collector.total_frames=$TOTAL_FRAMES \
  train.train_state_interval=$TRAIN_STATE_INTERVAL \
  train.eval_interval=$EVAL_INTERVAL \
  train.eval_episodes=$EVAL_EPISODES \
  train.eval_per_archetype=false \
  "train.eval_opponents=[$BASELINE_CHECKPOINT,random]" \
  seed=$SEED set_seed=true \
  wandb.group=$GROUP \
  wandb.name=$RUN_NAME \
  'wandb.tags=[pointer-head,mlp,seat-split,heuristic-resolved,15m,from-scratch]' \
  2>&1 | tee "$LOG"

echo "Log: $LOG"
