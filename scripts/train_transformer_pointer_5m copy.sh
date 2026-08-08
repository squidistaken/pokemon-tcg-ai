#!/bin/bash
# 5M-frame PPO self-play run on the transformer backbone with the pointer head.
#
# This is the backbone comparison against run pyo9oiis (5M frames, MLP backbone,
# same pointer head, W&B group pointer-head-5m-20260806). Every setting below is
# held at that run's values on purpose -- deck corpus, curriculum, league,
# optimizer, evaluation, seed -- so the trunk is what varies:
#
#   MLPBackbone  ->  TransformerBackbone + backbone.option_tokens=true
#
# option_tokens is the mechanism, not attention. It projects each option row
# with one Linear plus a segment embedding and hands it to the head, so the
# policy scores what an option *is* rather than which slot it sits in. It was
# the only one of nine arms in docs/architecture/transformer-diagnosis-45.md to
# beat its reference outside the sweep's noise floor (eval/first_snapshot 0.928
# vs 0.614, eval/random 0.862 vs 0.737 at 4M frames, -14% fps). Note the head
# already carries that idea on the MLP trunk here, so what this run actually
# tests is whether attention over the ten observation-group summaries adds
# anything on top of it.
#
# Nothing else from that sweep is enabled. Pre-LN, CLS pooling, per-entity
# tokens, the sub-batch change and the LR/entropy schedules were all refuted or
# fell inside a noise floor of sigma = 0.084 -- two byte-identical runs differed
# by 0.19. The arm that stacked eight changes at once scored well and taught
# nothing, because no result could be attributed to any of them.
#
# The one deliberate second change: num_layers 2 / ff_dim 512
# (conf/experiment/ptr_tf_capacity.yaml) rather than the 1 layer / ff_dim 256
# tf_pointer used. Depth is genuinely unresolved -- the flat head's capacity arm
# scored +0.056 rand / +0.069 snap over its reference while its byte-identical
# duplicate landed 0.084 below, so the two runs straddle the control -- and with
# token_groups [] the encoder attends over ten masked-mean group summaries, so
# the second layer has pooled averages rather than entities to relate. Cost is
# small (-8% fps measured) but the tradeoff is real: this run varies the trunk
# and its depth together, so a flat result will not say which.
#
# Watch train/grad_norm. The capacity arm ran 2-3x the reference (0.18 -> 0.70
# -> 0.42) against max_grad_norm 1.0, and it is one of four candidates for the
# monotonic 0.96 -> 4.14 growth that had tf_combined training ~95% of its run
# clipped.
#
# Two things pyo9oiis's own diagnosis asked for are NOT applied, because both
# would confound the backbone comparison this run exists to make:
#
#   * agent.entropy_coeff stays 0.02 (not the 0.05 of
#     train_pointer_15m_combined.sh). That answers an MLP-specific entropy
#     freeze at ~1.05 nats; the transformer pointer arm held 0.53-0.58 nats
#     at 0.02.
#
# RELAUNCH after 3o2jilac died at 770k/5M frames (~35 min in, 2026-08-06 22:48).
# The league is cut from pool_size 5 to 2 to lower memory pressure. This is a
# deliberate break from the original design note below, which held the league
# fixed so the trunk was the only variable -- pool_size changes the opponent
# distribution, so this run is no longer a clean control against pyo9oiis. It
# is a memory-headroom run first and a backbone comparison second.
#
# Cost of the cut: the league now spans the most recent 100k frames (2 x 50k)
# instead of 250k, so self-play is more incestuous. snapshot_interval stays at
# 50000 on purpose -- raising it to 125000 would restore the 250k span, but it
# would also move the first snapshot from 50k to 125k frames, and
# eval/first_snapshot is the anchor every earlier run is scored against.
#
# Corpus and curriculum are pyo9oiis's: the full heuristic-resolved corpus
# (~136 archetypes, 28,670 lists), capacity 1500, explore_prob 0.3. The
# curriculum is close to inert at that level count -- ~18k archetype pairs
# against ~14k episodes, and that run matured 79 levels while 11,780 sat in
# probation -- but it was equally inert for the MLP run, which is what makes it
# a fair control. conf/env/curriculum_v2.yaml (19 archetypes, 361 levels, 18,598
# lists) is where the curriculum actually functions; that is its own experiment.
#
# Expect ~3.5h (~3h collection plus ~25 min across 20 evaluation rounds, now
# at 50 episodes each rather than 200). Launch
# inside tmux/screen so it survives a disconnect -- the 15M run died at 3.08M
# frames when WSL restarted, and there is no resume path:
#
#   ./scripts/train_transformer_pointer_5m.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# The repo's own modules (submission/, src/, main.py) collide with same-named
# top-level modules on an inherited PYTHONPATH, so the run gets a clean one.
export PYTHONPATH=

# GAE runs the critic over the whole collected batch in one forward
# (ppo_trainer.py:393), and the epoch loop clones the batch once per epoch
# (ppo_trainer.py:420), so the allocator sees a handful of multi-GB transients
# per update against steady small ones. That is the fragmentation case
# expandable_segments was built for: run 26560 hit
# "OOM ... trying to allocate 8210350080 bytes (free: 0)" on a 24GB 4090 while
# nvidia-smi showed 23.8GB already reserved by that same process.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DECK_DIR="decks"
DECK_CORPUS="heuristic-resolved"
GROUP="tf-pointer-5m-20260806"
RUN_NAME="tf-pointer-2l-5m-s42-pool2"
TOTAL_FRAMES=5000000
SEED=42

# Memory headroom. Each of the 32 env workers holds its own copy of every
# league member, so resident opponent models are POOL_SIZE x num_workers.
# 5 -> 2 drops that from 160 to 64. snapshot_interval stays at 50000 so the
# eval/first_snapshot anchor lands where every earlier run's did.
POOL_SIZE=2
SNAPSHOT_INTERVAL=50000

# SUB_BATCH is the gradient minibatch and the knob that moves learner VRAM;
# FRAMES_PER_BATCH is the on-policy batch each update is drawn from. 1024 -> 512
# roughly halves activation memory per step. It is the least invasive cut
# available: every update still sees the same 16384 frames over the same 4
# epochs, just in 32 minibatches instead of 16, so the PPO objective and the
# data are unchanged and only the gradient-step granularity moves. Lowering
# FRAMES_PER_BATCH instead would change the update itself, so leave it alone.
SUB_BATCH=1024
FRAMES_PER_BATCH=16384

# pyo9oiis's curriculum settings. capacity 1500 against ~18k archetype pairs
# means the buffer cannot hold the corpus, so explore_prob > 0 is required or
# the sampler raises rather than discovering the rest lazily.
CURRICULUM=true
CURRICULUM_CAPACITY=1500
EXPLORE_PROB=0.3

# Evaluation, cut from pyo9oiis's 200 episodes to 50 to buy back wall clock.
# 3o2jilac spent ~2 min on first_snapshot and ~3 min on random per round, so
# 20 rounds cost ~100 min of a ~4h run; at 50 episodes that drops to ~25 min.
#
# The price is eval noise. Standard error on a win rate is ~7.1 points at 50
# episodes against ~3.5 at 200, so a single round is now noisier than the
# sigma = 0.084 floor this comparison already lives with. Read the trend over
# the 20 rounds rather than any one point, and treat a lone round-to-round
# swing under ~14 points as nothing. If a number needs to be quoted precisely,
# re-evaluate that checkpoint offline at high episode count instead of paying
# for precision 20 times during training.
#
# eval_per_archetype stays off for the reason it was already off -- 246 extra
# W&B series at ~1.5 games per archetype per round.
EVAL_INTERVAL=250000
EVAL_EPISODES=50
EVAL_ARGS="train.eval_interval=$EVAL_INTERVAL train.eval_episodes=$EVAL_EPISODES train.eval_per_archetype=false"

if [ ! -d "$DECK_DIR/$DECK_CORPUS" ]; then
  echo "ERROR: '$DECK_DIR/$DECK_CORPUS' not found. Run ./scripts/fetch_decks.sh first." >&2
  exit 1
fi

ARCHETYPES=$(find "$DECK_DIR/$DECK_CORPUS" -mindepth 1 -maxdepth 1 -type d | wc -l)
DECKS=$(find "$DECK_DIR/$DECK_CORPUS" -name '*.csv' | wc -l)
if [ "$ARCHETYPES" -lt 100 ]; then
  echo "ERROR: only $ARCHETYPES archetypes in '$DECK_DIR/$DECK_CORPUS'; expected ~136." >&2
  echo "       The corpus looks incomplete -- refusing to burn 4h on it." >&2
  exit 1
fi
echo "Training on $DECKS decks across $ARCHETYPES archetypes in $DECK_DIR/$DECK_CORPUS."

mkdir -p logs
LOG="logs/train-tf-pointer-5m-$(date +%Y%m%dT%H%M%S).log"

echo "League: pool_size $POOL_SIZE, snapshot every $SNAPSHOT_INTERVAL frames."
echo "Batching: sub_batch $SUB_BATCH over $FRAMES_PER_BATCH-frame updates."
echo "Eval: $EVAL_EPISODES episodes per opponent every $EVAL_INTERVAL frames."

# Everything not overridden below is pyo9oiis's, inherited from
# conf/ppo_selfplay_multideck.yaml: lr 3e-4 with no annealing, entropy_coeff
# 0.02, 4 epochs of minibatches over the on-policy batch, target_kl 0.03,
# max_grad_norm 1.0, gamma 0.999, lmbda 0.98, a PFSP league, and a 20% deck
# holdout. The league size and the batching are set above, not inherited.
uv run --frozen --no-sync python -m src.train --config-name ppo_selfplay_multideck \
  paths.data_dir=$DECK_DIR \
  deck_corpus=$DECK_CORPUS \
  model/backbone=transformer \
  model.backbone.option_tokens=true \
  model.backbone.num_layers=2 \
  model.backbone.ff_dim=512 \
  model/head=pointer \
  env.curriculum.enabled=$CURRICULUM \
  env.curriculum.capacity=$CURRICULUM_CAPACITY \
  env.curriculum.explore_prob=$EXPLORE_PROB \
  agent.device=cuda env.num_workers=32 collector.total_frames=$TOTAL_FRAMES \
  train.pool_size=$POOL_SIZE \
  train.snapshot_interval=$SNAPSHOT_INTERVAL \
  agent.sub_batch_size=$SUB_BATCH \
  agent.frames_per_batch=$FRAMES_PER_BATCH \
  $EVAL_ARGS \
  seed=$SEED set_seed=true \
  wandb.group=$GROUP \
  wandb.name=$RUN_NAME \
  'wandb.tags=[transformer,pointer-head,option-tokens,capacity,5m,pool2]' \
  2>&1 | tee "$LOG"

echo "Log: $LOG"
