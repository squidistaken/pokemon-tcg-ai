#!/bin/bash
# 15M-frame PPO self-play run piloting ONE deck against the full field.
#
# WHAT CHANGED AND WHY
#
# Three changes against scripts/train_mlp_pointer_seatsplit_15m.sh, each from a
# measurement on that run (W&B 049zz7kg, died at 3.78M frames).
#
# 1. ENTROPY_COEFF 0.05 -> 0.005
#
# The real action space is 4.90 legal options on average (median 4, 14.5% of
# decisions forced to one), so a coin-flipping policy sits at ln(4.90) = 1.589
# nats. train/entropy on 049zz7kg never left 1.40 -> 1.25 across 3.8M frames:
# exp(1.25) = 3.5 effective choices out of 4.9, i.e. ~79% as random as guessing,
# at the *end* of the run. The entropy term measured 14.0x the policy objective
# at 0.5M frames and still 3.9x at 3.5M. PPO was paid several times more to stay
# random than to win, and the eval curve is flat to match (0.745 -> 0.68 vs
# random) while the ent=0.02 transformer arm climbed 0.62 -> 0.94.
#
# 2. AGENT_DECK pinned, opponents left at full width
#
# 134 archetypes drawn on both seats is 17,956 matchups. 049zz7kg played 50,317
# episodes: 2.8 per matchup. Nothing learns a matchup from 2.8 episodes, and the
# curriculum could not either -- train/curriculum/size reached 97 with 11,692
# entries still in probation and score_max 0.157, so it was steering nothing.
#
# The submitted agent pilots exactly one list, so the generalist objective was
# never the one being scored. env.agent_deck pins the agent's seat and leaves
# the opposing field at all 134 archetypes -- the ladder it actually faces --
# collapsing the space from 17,956 matchups to 134.
#
# How those 134 are covered is NOT uniform. The draw is uniform over the
# 22,936 deck *lists*, and lists per archetype run 1 to 1,444, so opponent
# episodes land roughly in proportion to how many lists the scraper collected
# -- which tracks meta share. Over a 50k-episode run the top 10 archetypes
# take 48% of episodes (2,400-3,150 each) and the bottom 40 share 1% (~2
# each). That is defensible if the ladder is meta-weighted too and wrong if
# it samples archetypes evenly; we do not know which. env.deck_sampling and
# env.deck_weighting are the knobs, neither of which balances by archetype
# today.
#
# AGENT_DECK_FIELD_PROB keeps a fifth of episodes on a pool-drawn agent deck.
# Without it the card embeddings outside the pinned list stop receiving
# gradient, and the encoder degrades on exactly the cards the opponent plays.
#
# 3. CURRICULUM OFF
#
# A level is an ordered (agent, opponent) archetype pair, so pinning the agent
# discards the half of every level the curriculum drew; env_factory rejects the
# combination rather than scoring matchups nobody played. At 134 opponents a
# uniform draw already gives ~375 episodes each, which is where the curriculum
# had nothing to add for the archetypes that get thousands of episodes.
# Revisit once that saturates -- and note it would not have rescued the tail
# either, since a level needs min_visits=5 to mature at all.
#
# eval_deck_matchup goes mirror -> independent: under a pin the agent always
# brings its own deck, so a mirror draw would only ever produce the pinned
# list on both seats. eval_per_archetype goes off for the same reason --
# Evaluator attributes an episode to the archetype the *agent* piloted, which
# is now constant, so the breakdown would be one bucket. Breaking it down by
# the opponent's archetype instead is the useful version and is not wired yet.
#
# EVAL_EPISODES 200 -> 50 buys back wall clock: 049zz7kg spent 35 of its 153
# minutes (23%) in serial, single-process evaluation. At 50 episodes the
# per-round win-rate carries about +/- 0.07, so read the trend across rounds
# rather than any single one.
#
# WHAT WE MEASURE AGAINST
#
#   random          Uniform over the action mask. The only absolute yardstick
#                   we have: it depends on no earlier run, so its series is
#                   comparable across every run past and future.
#   first_snapshot  This run's own earliest snapshot. Measures improvement
#                   inside this run and inherits nothing from the lineages
#                   diagnosed above.
#   $BASELINE_CHECKPOINT
#                   The 5.01M transformer we last shipped, kept only as a
#                   'did we beat what is deployed' gate. It was trained under
#                   the seat-blind observation and the 17,956-matchup spread,
#                   so beating it is necessary, not sufficient.
#
# Every reference here is either uniform-random or a network we trained
# ourselves. There is no independent baseline in the repository, so the only
# external signal remains the Kaggle leaderboard.
#
# Under the pin, eval is: the pinned deck versus opponents dealt from the
# 5,734 held-out lists the run never trains on.
#
# DECK CHOICE
#
# rockets-honchkrow-2 is the most-observed single list in the scraped corpus
# (466 observations, 2.6x the next). Note the standing caveat: the in-engine
# probe for run 8zpnz876 scored this archetype 0.362, the worst of that
# shortlist, while it looked best on paper. Override AGENT_DECK to
# decks/heuristic-resolved/lucario-hariyama/lucario-hariyama-63.csv (383.8 on
# the leaderboard, the best result to date) to run the safer arm.
#
# The run is supervised: a crash restarts from train_state.pt rather than
# ending the run. See scripts/train_supervised.sh.
#
# Launch inside tmux/screen so it survives a disconnect:
#
#   ./scripts/train_meta_deck_pinned.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DECK_DIR="decks"
DECK_CORPUS="heuristic-resolved"
AGENT_DECK="${AGENT_DECK:-decks/heuristic-resolved/rockets-honchkrow/rockets-honchkrow-2.csv}"
AGENT_DECK_FIELD_PROB=0.2

GROUP="deck-pinned-20260808"
RUN_NAME="mlp-ptr-pinned-15m-s42"
TOTAL_FRAMES=15000000
SEED=42

ENTROPY_COEFF=0.005

EMBED_DIM=256
BACKBONE_CELLS="[512,512]"
CARD_EMBED_DIM=16
ENTITY_DIM=128

# 16 rather than 32: each worker holds its own SnapshotOpponentPool of
# pool_size networks in its own process, and 049zz7kg's throughput settled
# near 385 fps with 32 -- the engine, not the GPU, is the bottleneck, so the
# extra processes bought contention rather than frames.
NUM_WORKERS=16

POOL_SIZE=5
SNAPSHOT_INTERVAL=200000
TRAIN_STATE_INTERVAL=250000
EVAL_INTERVAL=500000
EVAL_EPISODES=50

BASELINE_CHECKPOINT="outputs/2026-08-07/03-20-11-p30924/checkpoints/snapshot_000005013504.pt"

if [ ! -f "$AGENT_DECK" ]; then
  echo "ERROR: agent deck '$AGENT_DECK' not found." >&2
  exit 1
fi
if [ ! -f "$BASELINE_CHECKPOINT" ]; then
  echo "ERROR: baseline checkpoint '$BASELINE_CHECKPOINT' not found." >&2
  exit 1
fi

# Matched on the module rather than the interpreter: uv may exec python,
# python3 or python3.13 depending on how the venv resolves, and a guard that
# misses lets a second run share the GPU and exhaust it. Bracketed first
# character so the pattern cannot match this script's own command line.
if pgrep -f "[-]m [s]rc.train" > /dev/null; then
  echo "ERROR: another src.train process is running and this run wants the same GPU." >&2
  exit 1
fi

ARCHETYPES=$(find "$DECK_DIR/$DECK_CORPUS" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [ "$ARCHETYPES" -lt 100 ]; then
  echo "ERROR: only $ARCHETYPES archetypes in '$DECK_DIR/$DECK_CORPUS'; expected ~136." >&2
  exit 1
fi

RUN_DIR="outputs/$GROUP/$RUN_NAME"

# This run trains from scratch: pokemon_seat_split resized the trunk's first
# projection, so no earlier checkpoint loads against it anyway. The supervisor
# resumes from train_state.pt by design, which is right after a crash and
# wrong on a deliberate relaunch -- so a leftover state stops the run instead
# of being picked up silently. Pass RESUME=1 to continue an interrupted run.
if [ -f "$RUN_DIR/train_state.pt" ] && [ "${RESUME:-0}" != "1" ]; then
  echo "ERROR: $RUN_DIR/train_state.pt exists, so this would CONTINUE a previous run." >&2
  echo "       To resume it:        RESUME=1 $0" >&2
  echo "       To start over:       rm -rf $RUN_DIR" >&2
  exit 1
fi

mkdir -p logs
LOG="logs/train-deck-pinned-$(date +%Y%m%dT%H%M%S).log"

echo "Agent pilots $(basename "$AGENT_DECK"); $AGENT_DECK_FIELD_PROB of episodes fall back to a"
echo "pool-drawn deck. Opponents drawn from all $ARCHETYPES archetypes. Run dir: $RUN_DIR"

./scripts/train_supervised.sh \
  --run-dir "$RUN_DIR" \
  --total-frames "$TOTAL_FRAMES" \
  --attempts 10 \
  -- --config-name ppo_selfplay_multideck \
  paths.data_dir=$DECK_DIR \
  deck_corpus=$DECK_CORPUS \
  env.agent_deck="$AGENT_DECK" \
  env.agent_deck_field_prob=$AGENT_DECK_FIELD_PROB \
  env.curriculum.enabled=false \
  env.eval_deck_matchup=independent \
  model/head=pointer \
  model.embed_dim=$EMBED_DIM \
  "model.backbone.num_cells=$BACKBONE_CELLS" \
  model.adapter.card_embed_dim=$CARD_EMBED_DIM \
  model.adapter.entity_dim=$ENTITY_DIM \
  model.adapter.option_target_state=true \
  model.adapter.pokemon_seat_split=true \
  agent.entropy_coeff=$ENTROPY_COEFF \
  agent.device=cuda \
  env.num_workers=$NUM_WORKERS \
  train.pool_size=$POOL_SIZE \
  train.snapshot_interval=$SNAPSHOT_INTERVAL \
  train.train_state_interval=$TRAIN_STATE_INTERVAL \
  train.eval_interval=$EVAL_INTERVAL \
  train.eval_episodes=$EVAL_EPISODES \
  train.eval_per_archetype=false \
  "train.eval_opponents=[first_snapshot,random,$BASELINE_CHECKPOINT]" \
  seed=$SEED set_seed=true \
  wandb.group=$GROUP \
  wandb.name=$RUN_NAME \
  'wandb.tags=[pointer-head,mlp,deck-pinned,seat-split,heuristic-resolved,15m]' \
  2>&1 | tee "$LOG"

echo "Log: $LOG"
