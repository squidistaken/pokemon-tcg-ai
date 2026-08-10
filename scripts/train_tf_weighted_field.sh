#!/bin/bash
# 15M-frame PPO self-play run over an observation-weighted field.
#
# WHAT CHANGED AND WHY
#
# 1. NO AGENT-DECK PIN
#
# The previous run pinned the agent to one list. Under self-play that is
# self-defeating: the opponent is a snapshot of the same network, and it is
# dealt field decks -- decks a pinned network barely practises -- so it misplays
# them and the learner trains against an opponent it crippled itself. The effect
# grows as the pin succeeds. agent_deck_field_prob softens it but cannot remove
# it, because 20% field exposure still leaves the opponent bad at 80% of what it
# is handed.
#
# Both seats now draw from the pool, so the network stays competent at whatever
# either seat is dealt and its snapshots are genuine adversaries.
#
# The pin's real motivation was that spreading training across 22,936 lists
# wastes it. That is what deck_weighting fixes, below -- properly, and without
# damaging the opponent.
#
# The submission deck is chosen afterwards, by probing candidates in-engine with
# the finished agent, rather than baked into 13 hours of training.
#
# 2. DECK_WEIGHTING=OBSERVATION
#
# Every list was previously equally likely, and 76.7% of the corpus was observed
# exactly once in a real tournament -- so most training went to decks nobody
# plays. Weighting deals each list in proportion to its manifest
# observation_count, which is the distribution the ladder actually presents:
# Dragapult ~6.5% of episodes (~13,000 games over this run), the long tail a
# handful each. Applies to both seats.
#
# 3. TRANSFORMER TRUNK WITH POKEMON ENTITY TOKENS
#
# Chosen for capability, not size. MLPBackbone has no per-entity token path
# except option_repr, so its pooled pokemon/zone features stay seat- and
# zone-blind in every configuration -- it cannot tell which Pokemon is where or
# whose it is. token_groups=[pokemon] routes the 18 pokemon slots through
# attention as individual tokens carrying their seat/zone segment embedding.
#
# It is also the smaller model: 0.93M parameters against the MLP's 2.85M. That
# costs nothing measurable -- at 325 fps a worker produces a frame every ~49 ms
# and the opponent forward is ~1.9 ms of it, so the model is ~4% of wall clock.
#
# 4. LR 3e-4 -> 2e-4
#
# The previous run's KL early-stop fired on 269 of ~619 updates: 43% of batches
# were cut short of their 4 epochs. conf/agent/ppo.yaml warns about exactly this
# ("it early-stops on every batch and quietly gives back the extra updates the
# minibatching just bought"). A lower constant LR keeps mean approx-KL under the
# 0.045 threshold so the epochs actually run. Constant, not annealed.
#
# 5. NO DECK HOLDOUT
#
# multideck sets deck_holdout_frac 0.2, which withheld 5,734 lists carrying
# 10,124 observations -- 20.1% of the corpus's mass, including several of its
# most-played lists (dragapult-dudunsparce-73 at 149 observations,
# dragapult-147 at 127). That is a fifth of the training signal spent to buy a
# memorization check that cannot fail here: 28,670 lists over ~205k episodes is
# ~9 episodes per list, and a 60-step hidden-information game with shuffle
# randomness is not memorizable at that rate with parameters shared across 134
# archetypes.
#
# It also made the eval panel worse. Drawn from a random 20% slice the panel
# covered 8 distinct archetypes; drawn from the whole corpus it is the true
# top-10 most-played lists and covers 10.
#
# The cost is that eval decks are also trained on, so the win-rate is nominally
# optimistic. It matters little: the most-played opponent list appears in ~1,900
# of 205k episodes, and two of the three eval opponents (random, the deployed
# checkpoint) are policy references rather than deck references, so deck
# familiarity shifts every arm equally and the comparisons between them hold.
#
# 6. EVALUATION THAT READS LIKE A SUBMISSION
#
# Eval previously dealt a different held-out list every episode, so a round was
# N matchups played once each and the deck draw swamped the policy signal. Now:
#
#   eval_agent_deck    the agent pilots one fixed deck, as a submission does.
#                      Safe here in a way a training pin is not, since eval
#                      never feeds back into learning.
#   eval_panel_size    opponents are the 10 most-observed held-out lists, drawn
#                      round-robin, so every round plays the same matchups and
#                      the curve is comparable across rounds.
#
# Both decks fixed means the only remaining variance is shuffle and coin flips.
# The per-archetype breakdown now buckets by the OPPONENT's archetype (the
# agent's is constant under the eval pin), which is what says where the policy
# is weak.
#
# At 50 episodes over a 10-deck panel each opponent gets 5 games per round, so
# read a single round's per-opponent numbers as noise and the accumulation
# across rounds as signal.
#
# eval_deck_matchup stays mirror, and that is load-bearing rather than
# cosmetic. AgentDeckSampler always hands the opponent the field sampler's
# *second* deck, so under `independent` the round-robin cursor advances twice
# per episode and the opponent only ever meets the odd half of the panel --
# measured: 5 of the 10 lists, at 10 episodes each. Under `mirror` the field
# draws one deck for both seats before the agent's is overwritten by the pin,
# so the cursor advances once and every panel entry is played exactly 5 times.
# The name is misleading here: with the agent's deck pinned there is no mirror
# match, it just means "deal the opponent the next panel deck".
#
# 7. COLLECTOR: MULTI_SYNC (issue #86)
#
# The first 8.73M frames ran on collector.type=sync, where the policy lives in
# this process and every worker waits for it at every step. That pegged the main
# process at 99% of one core while the machine sat at 307% of 1600%.
#
# collector.type=multi_sync gives each worker its own copy of the policy, so the
# forward that was serialized now happens 16 times in parallel. Measured on this
# box against this exact configuration, league populated, PPO update included:
# 251 -> ~400 fps, a 1.6x speedup, with the learning diagnostics unchanged
# (ESS 0.9937 vs 0.9934, clip_fraction 0.0292 vs 0.0329). Batches stay
# on-policy, because multi_sync's workers idle between handing a batch over and
# being told to continue, so no V-trace and no other hyperparameter changes.
#
# agent.collector_device=cpu is required, not an optimisation. The workers are
# forked and CUDA cannot initialize in a forked child, so a CUDA collection
# policy fails outright; the trainer now rejects it with that message. The PPO
# update keeps the GPU via agent.device=cuda, which is worth 7.8x and is now
# 56% of the loop.
#
# env.num_workers stays 16. Between 10 and 16 the end-to-end differences are
# inside run-to-run noise. See docs/training-performance.md section 6.
#
# 8. VRAM CAP OFF
#
# agent.cuda_memory_fraction=null, as requested. The cap existed because an
# uncapped allocator on WSL fails in the driver rather than in Python, killing
# the CUDA context and with it the run. That risk is real but bounded here: the
# supervisor restarts from train_state.pt, so a lost context costs at most
# train.train_state_interval (250k) frames. Measured peak usage under this
# configuration is in docs/training-performance.md; if it is comfortably under
# the card, the cap was never binding anyway.
#
# 9. LEAGUE MEMORY: POOL_SIZE 5 -> 12, PFSP_MIN_WEIGHT 0.05 -> 0.15
#
# The first 8.77M frames made strong relative progress and no absolute
# progress. Training win rate fell 0.680 -> 0.541, which is what self-play
# should do as the league fills with recent copies of the learner. But over the
# same span every fixed reference was flat or worse: vs random 0.900 -> 0.756
# (about 4 standard errors on a 5-round mean, so real), vs its own first
# snapshot 0.880 -> 0.824, vs the deployed MLP 0.388 -> 0.400.
#
# It is not under-training. KL climbed 0.0064 -> 0.0227 against target_kl 0.03
# and clip_fraction 0.072 -> 0.195, so updates got larger, not smaller. Entropy
# held at ~0.93 and explained variance at ~0.74, so neither the policy nor the
# critic collapsed. That combination -- rising update magnitude, converging
# self-play win rate, static absolute skill -- is the signature of chasing a
# moving target.
#
# Two settings compounded into maximum recency bias. POOL_SIZE 5 at
# SNAPSHOT_INTERVAL 200000 gave the league a 1M-frame memory, 3.5% of this
# run's budget. And PFSP `hard` weighting, (1 - p)^2 + min_weight, by
# construction deprioritises members the learner already beats, which is
# exactly the older strategies worth retaining. At min_weight 0.05 a mastered
# member (p ~ 0.9) drew weight 0.06 against 0.30 for an even one.
#
# POOL_SIZE 12 widens the memory to 2.4M frames; PFSP_MIN_WEIGHT 0.15 flattens
# that 5:1 tilt to 2.5:1 so history keeps a real share of episodes. Cost is
# memory only: snapshots are 7.4 MB, so 12 per worker is ~770 MB across 16
# workers.
#
# This is a hypothesis, not a demonstrated fix. What would falsify it:
# eval/random/win_rate still drifting down over the first 4-6 rounds after the
# restart (2-3M frames). If it is, the league is not the cause and running to
# 28.7M is not worth the hours. Confound worth remembering while reading it:
# eval pins the agent to one deck while training spreads over 135 weighted
# archetypes, so these curves may understate progress on the training
# distribution.
#
# WHAT WE MEASURE AGAINST
#
#   random             Uniform over the action mask. The only absolute
#                      yardstick, comparable across every run past and future.
#   first_snapshot     This run's own earliest snapshot: improvement within the
#                      run.
#   $SUBMITTED_AGENT   The 10.14M MLP submitted as
#                      mlp-ptr-pinned-10m-honchkrow-greedy. The "are we better
#                      than what is deployed" gate, and the only reference here
#                      that cannot drift with this run's league, since it comes
#                      from a finished one.
#
# Checkpoint opponents are always played by GreedyPolicyOpponent, so this arm is
# the greedy variant by construction -- the same action selection the submitted
# bundle uses, not the sampled one. Its architecture is rebuilt from the config
# the checkpoint embeds, so an MLP opponent inside this transformer run is fine.
#
# Two caveats on reading it. It pilots decks from the eval panel rather than the
# rockets-honchkrow-2 list it was submitted with, so this measures the network,
# not the submitted bundle. And it is still a network we trained ourselves:
# there is no independent baseline in the repository, so the Kaggle leaderboard
# remains the only external signal.
#
# The run is supervised: a crash restarts from train_state.pt. See
# scripts/train_supervised.sh. Launch inside tmux/screen so it survives a
# disconnect:
#
#   ./scripts/train_tf_weighted_field.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DECK_DIR="decks"
DECK_CORPUS="heuristic-resolved"
EVAL_AGENT_DECK="${EVAL_AGENT_DECK:-decks/heuristic-resolved/dragapult-blaziken/dragapult-blaziken-3.csv}"
EVAL_PANEL_SIZE=10

GROUP="weighted-field-20260808"
RUN_NAME="tf-ptr-weighted-15m-s42"
# Extended from 15M to ~100M so the run keeps going unattended for days. The
# supervisor treats this as an absolute target and subtracts what train_state.pt
# already records, so it must be the final total rather than the increment. The
# value is a multiple of the 16384-frame batch, which keeps the last batch whole.
TOTAL_FRAMES="${TOTAL_FRAMES:-99991552}"
SEED=42

LR=2.0e-4
ENTROPY_COEFF=0.005

EMBED_DIM=256
CARD_EMBED_DIM=16
ENTITY_DIM=128

NUM_WORKERS=16
# 12, not the previous 5: see section 9. At SNAPSHOT_INTERVAL the league's
# memory is POOL_SIZE * 200000 frames, so 5 remembered only the last 1M.
POOL_SIZE=12
# Floor on every league member's PFSP weight. The default 0.05 lets `hard`
# weighting squeeze a mastered member to ~1/5 the share of an even one, which
# is how the league forgets what it already learned to beat.
PFSP_MIN_WEIGHT=0.15
SNAPSHOT_INTERVAL=200000
TRAIN_STATE_INTERVAL=250000
EVAL_INTERVAL=500000
EVAL_EPISODES=50

# The network behind mlp-ptr-pinned-10m-honchkrow-greedy (checkpoint key
# 60e28feb298b), so the eval curve answers "have we beaten what is deployed".
SUBMITTED_AGENT="outputs/deck-pinned-20260808/mlp-ptr-pinned-15m-s42/checkpoints/snapshot_000010141696.pt"

if [ ! -f "$EVAL_AGENT_DECK" ]; then
  echo "ERROR: eval agent deck '$EVAL_AGENT_DECK' not found." >&2
  exit 1
fi
if [ ! -f "$SUBMITTED_AGENT" ]; then
  echo "ERROR: submitted-agent checkpoint '$SUBMITTED_AGENT' not found." >&2
  exit 1
fi

# Matched on the module rather than the interpreter: uv may exec python, python3
# or python3.13 depending on how the venv resolves, and a guard that misses lets
# a second run share the GPU and exhaust it. Bracketed first character so the
# pattern cannot match this script's own command line.
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

# The supervisor resumes from train_state.pt by design, which is right after a
# crash and wrong on a deliberate relaunch -- so a leftover state stops the run
# instead of being picked up silently. Pass RESUME=1 to continue.
#
# Continuing is the expected mode now: this run stopped at 8.73M of 15M frames,
# and RESUME=1 picks up the weights, Adam's moments and the frame count, so the
# remaining ~6.27M are collected under the section 7 collector change.
if [ -f "$RUN_DIR/train_state.pt" ] && [ "${RESUME:-0}" != "1" ]; then
  echo "ERROR: $RUN_DIR/train_state.pt exists, so this would CONTINUE a previous run." >&2
  echo "       To resume it:  RESUME=1 $0" >&2
  echo "       To start over: rm -rf $RUN_DIR" >&2
  exit 1
fi

# Absolute, so src/train.py takes it as given instead of creating it
# exclusively under the Hydra run directory. A resume reuses the run directory,
# so the relative default would find checkpoints/ already there and abort with
# "another run owns it" -- and the league in it is precisely what this run
# should carry forward rather than start empty.
CHECKPOINT_DIR="$ROOT/$RUN_DIR/checkpoints"

mkdir -p logs
LOG="logs/train-weighted-field-$(date +%Y%m%dT%H%M%S).log"

echo "Training unpinned over $ARCHETYPES archetypes, observation-weighted on both seats."
echo "Eval: $(basename "$EVAL_AGENT_DECK") vs a fixed $EVAL_PANEL_SIZE-deck held-out panel."
echo "Run dir: $RUN_DIR"

./scripts/train_supervised.sh \
  --run-dir "$RUN_DIR" \
  --total-frames "$TOTAL_FRAMES" \
  --attempts 10 \
  -- --config-name ppo_selfplay_multideck \
  paths.data_dir=$DECK_DIR \
  deck_corpus=$DECK_CORPUS \
  env.agent_deck=null \
  env.deck_weighting=observation \
  env.eval_agent_deck="$EVAL_AGENT_DECK" \
  env.eval_panel_size=$EVAL_PANEL_SIZE \
  env.deck_holdout_frac=0.0 \
  env.eval_deck_sampling=round_robin \
  env.eval_deck_matchup=mirror \
  env.curriculum.enabled=false \
  model/backbone=transformer \
  model/head=pointer \
  "model.backbone.token_groups=[pokemon]" \
  model.backbone.option_tokens=true \
  model.embed_dim=$EMBED_DIM \
  model.adapter.card_embed_dim=$CARD_EMBED_DIM \
  model.adapter.entity_dim=$ENTITY_DIM \
  model.adapter.option_target_state=true \
  model.adapter.pokemon_seat_split=true \
  agent.lr=$LR \
  agent.entropy_coeff=$ENTROPY_COEFF \
  agent.device=cuda \
  agent.collector_device=cpu \
  agent.cuda_memory_fraction=null \
  collector.type=multi_sync \
  env.num_workers=$NUM_WORKERS \
  train.checkpoint_dir="$CHECKPOINT_DIR" \
  train.pool_size=$POOL_SIZE \
  train.pfsp_min_weight=$PFSP_MIN_WEIGHT \
  train.snapshot_interval=$SNAPSHOT_INTERVAL \
  train.train_state_interval=$TRAIN_STATE_INTERVAL \
  train.eval_interval=$EVAL_INTERVAL \
  train.eval_episodes=$EVAL_EPISODES \
  train.eval_per_archetype=true \
  "train.eval_opponents=[first_snapshot,random,$SUBMITTED_AGENT]" \
  seed=$SEED set_seed=true \
  wandb.group=$GROUP \
  wandb.name=$RUN_NAME \
  'wandb.tags=[pointer-head,transformer,pokemon-tokens,observation-weighted,unpinned,15m]' \
  2>&1 | tee "$LOG"

echo "Log: $LOG"
