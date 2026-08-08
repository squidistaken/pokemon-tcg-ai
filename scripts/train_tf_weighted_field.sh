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
TOTAL_FRAMES=15000000
SEED=42

LR=2.0e-4
ENTROPY_COEFF=0.005

EMBED_DIM=256
CARD_EMBED_DIM=16
ENTITY_DIM=128

NUM_WORKERS=16
POOL_SIZE=5
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
if [ -f "$RUN_DIR/train_state.pt" ] && [ "${RESUME:-0}" != "1" ]; then
  echo "ERROR: $RUN_DIR/train_state.pt exists, so this would CONTINUE a previous run." >&2
  echo "       To resume it:  RESUME=1 $0" >&2
  echo "       To start over: rm -rf $RUN_DIR" >&2
  exit 1
fi

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
  env.num_workers=$NUM_WORKERS \
  train.pool_size=$POOL_SIZE \
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
