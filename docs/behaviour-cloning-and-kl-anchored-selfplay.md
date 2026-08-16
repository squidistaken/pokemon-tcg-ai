# Behaviour cloning and KL-anchored self-play

Written 2026-08-16.

Two stages, in the order AlphaGo and AlphaStar used. Stage 1 clones expert play
from Kaggle's published episode exports. Stage 2 runs self-play from that clone
while a KL term holds the policy near it.

Stage 1 is finished and submitted. Stage 2 is prepared and not yet run.

| | |
| --- | --- |
| clone checkpoint | `outputs/bc/bc-v6-submit.pt`, SHA-256 prefix `61b740c2df7c` |
| held-out accuracy | 0.7249 against a 0.3603 base rate |
| against the previous agent | 0.967 over 60 games (58-2) |
| Kaggle score | 873.9, against a previous best of 569.9 |
| submitted | `bc-v6-expert-top1`, Kaggle ref 55536749 |
| next | `sbatch slurm-conf/train_kl_anchored.sh` |

## Why behaviour cloning

Self-play PPO reached 180M frames and stopped improving against anything except
its own league. Three measurements show why:

- pin-175.7M scores 0.95 on its pinned deck and 0.05 to 0.33 on other decks,
  below its own initialization.
- Snapshots 2M frames apart agree on 0.40 of their decisions at 20M frames and
  0.52 at 170M. The policy is a fresh best response, not an accumulation.
- League Bradley-Terry rating rose from -94.2 to +76.0 over 180M frames with 0
  intransitive triples, while the Kaggle score stayed near 621.8.

Every game AI that beat strong humans bought its improvement operator from
search or from human data. The one system that bought neither, OpenAI Five, used
about 1e11 frames. This project has one RTX 4090.

## Relation to AlphaGo and AlphaStar

AlphaGo trained a supervised network on about 30 million positions from human
games. That network alone beat every earlier Go program. Self-play came second
and the value network third, trained on games the reinforcement-learning policy
played. Two details carry over:

- AlphaGo used the **supervised** policy as the search prior, not the
  reinforcement-learning one, because self-play makes the policy peaked and a
  peaked prior explores worse.
- It trained the value network on one position per game. Positions inside a game
  share an outcome, so a network fitted on all of them memorizes the game
  instead of judging the board.

AlphaStar trained on about 971,000 human replays, then fine-tuned on the
strongest players. During reinforcement learning it kept a KL penalty against
the frozen supervised policy for the whole run, used a league of main agents and
exploiters with prioritized fictitious self-play, and conditioned the policy on
a statistic describing the strategy being imitated.

This project already had the league. It now has stage one and the KL penalty. It
does not condition on a strategy statistic, and it does not use search.

# Stage 1: cloning

## Data

Source: `kaggle/pokemon-tcg-ai-battle-episodes-2026-08-13` and `-2026-08-14`,
the official daily exports listed in
`kaggle/pokemon-tcg-ai-battle-episodes-index`. Each day holds about 4,600
top-of-ladder episodes. Median player score is 1043 and the top is 1262, against
this project's 621.8.

`logs/bc/fetch_episodes.sh` downloads them to
`logs/kaggle_episodes/<date>/<episode>.json`. 9,062 games, 42 GB.

`tools/bc/bc_extract.py` turns them into decisions. Every replay step stores the
acting agent's full observation, so the same `StructuredObservationEncoder` used
in reinforcement learning encodes it. Nothing is re-simulated, so the clone
cannot disagree with the trainer about what a state looks like.

One row per accumulation position:

| field | content |
| --- | --- |
| `observation` | encoded from the acting seat, `already_chosen_option_count` = position |
| `action` | option index the player picked, or 128 for stop |
| `action_mask` | copy of `TCGEnv._build_mask` |
| `outcome` | episode result from the acting seat, +1 or -1 |
| `episode` | source game, used for the split |

A selection of k picks becomes k rows, plus a stop row when
`minCount <= k < min(maxCount, n_options)`. Declining an optional selection is
that stop row at k=0. Stop rows are 0.68% of the corpus.

Result: 1,632,486 decisions in 10 memory-mapped shards, 45 GB. Rows are 28 KB,
so the corpus does not fit in 23 GB of RAM. `ShardedDataset` in
`tools/bc/bc_train.py` pages rows from disk per batch.

## Model architecture

`StructuredObsAdapter` turns the observation into tokens. It embeds card IDs
(16 dims), attack IDs (8) and categories (4), pools each zone (hand, discard,
prizes, deck view) by mean, max and sum, and splits Pokemon tokens by seat.

A transformer backbone reads those tokens: 2 layers, 4 heads, width 256,
feed-forward 512, pre-layer-norm with a final norm, no dropout. It emits one
token per legal option plus a mean-pooled state vector.

`PointerHead` scores option slot i from the pair `[state, option_i]` through a
128-128 MLP with tanh, and produces one extra logit for stop. Scoring each
option from its own token is what lets the policy react to what an action does
rather than to where it sits in the list. The value head is a separate 256-256
MLP on the pooled state.

| component | value |
| --- | --- |
| backbone | transformer, 2 layers, 4 heads |
| width / feed-forward | 256 / 512 |
| normalization | pre-layer-norm, final norm |
| dropout | 0.0 |
| policy head | `PointerHead`, 128-128, tanh, plus a stop logit |
| value head | 256-256 MLP on the pooled state |
| card / attack / category embeddings | 16 / 8 / 4 |
| zone pooling | mean, max, sum |
| parameters | 2,525,675 |

`conf/experiment/kl_anchored_selfplay.yaml` restates this architecture exactly,
because `train.init_checkpoint` fails on any size mismatch.

## Cloning hyperparameters

| parameter | value | note |
| --- | --- | --- |
| optimizer | AdamW | |
| learning rate | 3e-4, constant | no annealing |
| weight decay | 1e-4 | |
| batch size | 512 | |
| gradient clip | 1.0 | |
| epochs | 12 | best was epoch 9 |
| seed | 42 | |
| value loss weight | 0.5 | added to the policy cross-entropy |
| validation fraction | 0.05 | 453 whole games, 78,799 decisions |
| split | by game | not by row |
| checkpoint selected on | validation policy loss | not `policy + 0.5 * value` |

Weights start from scratch. The loss is masked cross-entropy on the policy head
plus `0.5 * MSE` on the value head against the episode outcome.

Two choices in that table are load-bearing. The split is by **game** because
states inside one game are far too correlated for a row split to measure
generalization. The checkpoint is selected on validation **policy** loss because
the value head degrades every epoch while the policy improves, so selecting on
the sum saves a worse policy.

## Metrics and their baselines

Raw accuracy is misleading, because 36.03% of expert decisions are option 0.

| metric | meaning | baseline |
| --- | --- | --- |
| `val/accuracy` | top-1 over all held-out decisions | 0.3603, always answering 0 |
| `val/nontrivial_accuracy` | top-1 where the answer is not option 0 | none; this is the number that tracks play skill |
| `val/predicts_zero` | how often the model answers 0 | 0.3603; above that means falling back on the default |
| `val/top5` | answer within the 5 highest-scoring legal actions | 0.7569; weak here, since mean legal actions is 7.54 and 47% of rows offer 5 or fewer |
| `val/value_loss` | MSE against the +1/-1 outcome | 0.9954, the variance of the outcome |

Report `nontrivial_accuracy` next to `accuracy`. Raw accuracy rises when the
model gets better at the majority class, which is not play skill.

## Results

Run `bc-v6-aligned`, W&B group `bc-v6-20260815`, entity `pokemon-tcg-ai`.

| epoch | train acc | val acc | train loss | val loss | nontrivial | predicts 0 | value loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.5992 | 0.6574 | 1.1224 | 0.9855 | 0.5725 | 0.3804 | 0.7202 |
| 3 | 0.7029 | 0.7092 | 0.8505 | 0.8461 | 0.6322 | 0.3803 | 0.8869 |
| 6 | 0.7261 | 0.7168 | 0.7914 | 0.8289 | 0.6423 | 0.3791 | 0.9532 |
| 9 | 0.7383 | 0.7249 | 0.7506 | 0.8131 | 0.6488 | 0.3839 | 0.9877 |

Training and validation accuracy stay within 0.014 of each other, so the policy
head does not overfit. `predicts_zero` sits at 0.3839 against a base rate of
0.3603, so the policy does not fall back on the default action. The value head
does overfit: its validation loss rises every epoch.

### Against the previous agent

The reference opponent is pin-175.7M
(`outputs/deck-pinned-150m-local/tf-ptr-pinned-selfplay-10m-s42/checkpoints/snapshot_000175702016.pt`),
the strongest self-play-only checkpoint available locally. It was never itself
submitted; the closest submitted relative is a 158M snapshot from the same run,
which scored 521.5. The best submission before cloning was 569.9
(`fixed-deck-frozen-30m-s42`).
`tools/bc/head_to_head.py` alternates seats so neither side gets the first-player
advantage, and plays both policies greedily, which is how they are deployed.

| test | clone | result |
| --- | --- | --- |
| 60 games, both on `expert_top1` | epoch 9 | 0.967 (58-2) |
| 24 games on each of 5 expert decks, pin on `alakazam-dudunsparce` | epoch 4 | 119 wins in 120 |
| 40 games, clone on `expert_top1`, pin on its own best deck | epoch 4 | 1.000 (40-0) |
| 40 games, both on `alakazam-dudunsparce` | epoch 4 | 0.350 (14-26) |

The five-deck sweep rules out a single favourable matchup. The last row is the
honest limit: on a deck the corpus barely contains, the clone is worse than pin.
Expert decks overlap that list by 16 of 60 cards on average, and a third of
expert decklists share no cards with it.

Beating pin-175.7M is necessary, not sufficient: self-play-only checkpoints from
that family scored 521 to 570 on Kaggle, while the corpus this clone learned
from averages 1043.

The clone submitted as `bc-v6-expert-top1` scored **873.9**, against a previous
best of 569.9.

## The checkpoint

`outputs/bc/bc-v6-submit.pt`, SHA-256 prefix `61b740c2df7c`, epoch 9 of 12.

The trainer writes a file only when validation policy loss improves. Epoch 9
reached 0.8131; epochs 10 and 11 ran and were evaluated but did not beat it, so
their weights were never written. There is no final-epoch file by design, only
the best one.

Epoch 11 was 0.0024 better on raw accuracy and 0.0013 worse on nontrivial
accuracy, against an accuracy standard error of 0.0018 over 78,799 held-out
decisions. The two are not distinguishable. Epoch 11 also predicted option 0
more often (0.3916 against 0.3839), so the loss and that diagnostic agree.

Three files hold identical bytes: `bc-v6-aligned.pt` (the trainer's output
path), `bc-v6-submit.pt` (a frozen copy) and the `model.pt` inside the submitted
bundle. **Use `bc-v6-submit.pt` everywhere.** The trainer's output path is
overwritten by any later cloning run, which would silently change what a resumed
reinforcement-learning run is anchored to.

Keep this checkpoint after reinforcement learning as well. It is the search
prior, for the AlphaGo reason above.

## The value head

Validation MSE is 0.99 against a constant-predictor baseline of 0.9954, which
looks useless. It is not. Measured on 20,000 held-out rows:

- correlation with outcome 0.4739
- sign agreement 0.716, so it names the winner from a mid-game state 72% of the
  time
- prediction standard deviation 0.8670, against an outcome standard deviation of
  0.9999

The head is overconfident, not uninformative. MSE punishes confident and wrong
far harder than it rewards confident and right. Scaling the output by
`cov / var = 0.5465` moves MSE to 0.7753, which explains 22.5% of the variance.
Apply that scale before using the head as a search target.

The cause is label correlation: 1,632,486 rows come from 9,062 games, so 180
rows share one +1 or -1 label. This is the problem AlphaGo solved by sampling a
single position per game. Do the same before relying on this head.

# Stage 2: KL-anchored self-play

## The anchor

`src/training/supervised_kl_anchor.py` holds `SupervisedKLAnchor`. It keeps a
frozen copy of the clone and adds `coefficient * KL(current || reference)` to
the PPO loss on the states the collector visited. The mask restricts the
divergence to legal actions, because illegal slots carry arbitrary logits.

This is AlphaStar's KL term. A warm start alone does not help: nothing stops the
policy from leaving expert play once training resumes, which is the drift
measured above. The penalty has to hold for the whole run.

Configuration, in `conf/agent/ppo.yaml`:

```yaml
kl_anchor_coeff: 0.0        # 0 disables the anchor
kl_anchor_checkpoint: null  # null anchors to the learner's initial weights
```

The trainer logs `kl_to_bc`, the unweighted divergence, so the coefficient can
be tuned against a number that does not move when the coefficient changes.

Verified: an unchanged policy gives KL 0.000000. A policy perturbed by Gaussian
noise of scale 0.05 gives KL 1.921722 and loss 0.192172 at coefficient 0.1.
Gradients reach the learner.

## Choice of coefficient

The main run uses 0.05. This is reasoned, not measured. PPO's own `target_kl` is
0.03 per update. At coefficient 0.05 a drift of 0.5 nats from the clone costs
0.025, which shapes the objective without dominating it. Coefficient 0.25 would
contribute 0.125 and probably pin the policy to the clone.

The sizing was originally argued against the entropy term at coefficient 0.02,
which was itself an accident. Measured over the first 8.5M frames the anchor
contributed 0.05 × 0.12 = 0.0060 against the entropy term's 0.0177, so the
comparison the reasoning relied on did not hold and the anchor was the smaller
term by 3x. `entropy_coeff` moved to 0.005 for that reason; see "Changes made
during the run".

## Decks

Cloning ignores decks. It imitates whatever list the player brought, and the
board tells it which archetype that is, so the clone learned from all 198
distinct lists in the corpus. Self-play has to choose a pool,
because it generates its own games.

At the start of every episode each seat draws one deck from the pool
**independently** and **uniformly**. The two draws are separate, so with a pool
of N decks about 1/N of episodes are mirrors and the rest are cross-archetype.
Nothing is pinned and nothing is held out. Real ladder shares are not used as
weights.

```yaml
env:
  deck_pool: decks/expert_pool
  deck_matchup: independent   # each seat drawn separately, not forced mirrors
  deck_holdout_frac: 0.0      # nothing withheld
  # deck_sampling: uniform    # inherited from conf/env/multideck.yaml
```

Pinning one deck is the failure already measured on pin-175.7M, which reached
0.95 on its pinned list and 0.05 to 0.33 on every other. Under self-play a pin
also cripples the opponent, which is a snapshot of the same network handed a
deck it never practised.

### How large the pool should be

Counted over all 9,062 games: 18,124 decklists, two per game, 198 distinct,
evenly split between the two days (9,230 and 8,894).

| pool size | share of decklists |
| --- | --- |
| top 1 | 14.3% |
| top 5 | 50.9% |
| top 10 | 61.0% |
| top 20 | 73.7% |
| top 30 | 80.0% |
| top 50 | 88.5% |

Every deck in any of these pools is by construction already in the clone's
training data, because the census counts the same games the clone learned from.
That matters for the anchor: a KL penalty is only meaningful where the reference
policy is competent, and anchoring to the clone on a deck it never saw would
penalize the learner for leaving an uninformed prior.

`decks/expert_pool/` holds the top five, which is 50.9%. That is too narrow. The
agent would never meet half the field, and the clone already plays all 198
lists, so a five-deck pool narrows the policy against its own starting point on
the rest. That is the forgetting the KL term exists to prevent, reintroduced
through the deck pool.

`decks/expert_pool_30/` holds the top 30, at 80.0%. Use that one.

Uniform weighting inside the pool is deliberate. Weighting by observed share
would train the agent into whichever deck is most common today, and the
submission picks its own list, so even competence across the pool keeps that
choice open. The alternative worth testing is weighting the opponent seat by
observed share while the agent's seat stays uniform, so the field matches the
ladder while our own practice stays broad. The sampler supports per-deck weights
already. This run does not use them.

## Self-play hyperparameters

Inherited from `conf/agent/ppo.yaml` and `conf/train/ppo_selfplay.yaml` unless
`conf/experiment/kl_anchored_selfplay.yaml` overrides them. The architecture is
the clone's, restated above. Everything marked **explicit** is written out in
the experiment file; see "The eight defaults that were wrong" below for why.

| parameter | value | note |
| --- | --- | --- |
| `kl_anchor_coeff` | 0.05 | the one new term; see below |
| `kl_anchor_checkpoint` | `outputs/bc/bc-v6-submit.pt` | frozen reference |
| `train.init_checkpoint` | `outputs/bc/bc-v6-submit.pt` | whole network, Adam restarts from zero |
| `clip_epsilon` | 0.2 | |
| `entropy_coeff` | 0.005 | explicit |
| `gamma` | 0.999 | reward is terminal only, so this predicts win probability |
| `lmbda` | 0.98 | GAE, averaged, 8 chunks |
| `lr` | 2.0e-4, constant | explicit; no annealing |
| `weight_decay` | 0.0 | deliberate; see below |
| `num_epochs` | 4 | PPO passes per batch |
| `frames_per_batch` | 16,384 | |
| `sub_batch_size` | 1,024 | |
| `max_grad_norm` | 1.0 | |
| `target_kl` | 0.03, multiplier 1.5 | PPO's own trust region, separate from the anchor |
| `snapshot_interval` | 500,000 frames | explicit; league snapshots |
| `pool_size` | 10 | explicit; league size, so the window is 5M frames |
| `pfsp_min_weight` | 0.15 | explicit; floor share for a beaten league member |
| `opponent_sampling` | pfsp, `hard` weighting | |
| `opponent_action_selection` | greedy | |
| `eval_interval` / `eval_episodes` | 500,000 frames / 40 | |
| `eval_deterministic` | true | explicit; argmax, matching how the policy is submitted |
| `eval_opponents` | first_snapshot, checkpoint, random | see below |
| `set_seed` | true | explicit; worker seeds and torch RNG survive a resume |
| `agent.device` | cuda | explicit; the default leaves the allocated GPU idle |
| `agent.collector_device` | cpu | required under `multi_sync`: CUDA cannot initialize in a forked child |
| `env.mp_start_method` | fork | inherited from `conf/env/default.yaml:199` |
| `collector.type` | multi_sync | explicit; the throughput setting, see below |
| `env.num_workers` | 32 | matches the Slurm allocation; every 900 fps run on these nodes used 32 |
| `collector.total_frames` | 300,000,000 | past what 24h collects, so the job uses the whole allocation |

`collector_device: cpu` with `mp_start_method: fork` is not a compromise. That
exact pair is running in production on
`fixed-deck-expert30-frozen-kl005-30m-s42`, this run's architecture and anchor,
which has reached 16.3M frames at 536 fps. The PPO update keeps the GPU; only
the forked collector workers stay on CPU.

`train.init_checkpoint` loads the **whole** network: shared trunk, policy head
and value head. `_load_warm_start_weights` calls `load_state_dict(strict=False)`
only to tolerate newly added LayerNorm parameters; anything else missing or
unexpected raises. The optimizer is not restored, so Adam's moments start at
zero.

That means the run inherits the clone's value head, which is overconfident:
correlation 0.474 with the outcome but predictions at standard deviation 0.867
where 0.47 would be calibrated. PPO retrains the critic against its own returns
immediately, and r=0.474 is a better start than noise, but the overconfidence
inflates early advantage magnitudes. Watch `loss_critic` and explained variance
over the first 1M frames.

Weight decay stays off, even though earlier runs used 1e-4. AdamW shrinks
weights toward zero, while this run starts at the clone and exists to stay near
it, so decay and the anchor pull in different directions. The anchor is the
regularizer here, and it constrains the policy in function space rather than
parameter space, which is the better place for it. Cloning did use 1e-4, which
was right: no anchor existed and the weights started from scratch.

Note that `target_kl` and `kl_anchor_coeff` measure different things.
`target_kl` bounds how far one update moves the policy from itself and
early-stops the epoch loop. `kl_anchor_coeff` penalizes distance from the clone,
which accumulates over the whole run.

## The eight defaults that were wrong

The experiment file originally set only the anchor, the decks, the architecture
and the eval panel. Everything else came from the shared config groups, and
eight of those inherited values were wrong for this run. It trained about 8M
frames that way. Not one was a considered choice later revised; each was a
default the experiment file never mentioned. All eight are now written out
explicitly, which is the habit worth keeping: state anything the experiment
depends on, even where the default is already right.

| setting | inherited | now | inherited from |
| --- | --- | --- | --- |
| `collector.type` | sync | multi_sync | `conf/collector/default.yaml:23` |
| `agent.lr` | 3.0e-4 | 2.0e-4 | `conf/agent/ppo.yaml:58` |
| `agent.entropy_coeff` | 0.02 | 0.005 | `conf/agent/ppo.yaml:11` |
| `train.pool_size` | 5 | 10 | `conf/train/default.yaml:32` |
| `train.snapshot_interval` | 50,000 | 500,000 | `conf/train/ppo_selfplay.yaml:8` |
| `train.pfsp_min_weight` | 0.05 | 0.15 | `conf/train/default.yaml:85` |
| `train.eval_deterministic` | false | true | `conf/train/ppo_selfplay.yaml:22` |
| `set_seed` | false | true | `conf/config.yaml:18` |

The run continued from `train_state.pt` rather than restarting. That file stores
only `state_dict`, `optimizer`, `frames` and `torch_rng_state`
(`src/training/callbacks/train_state_callback.py:128`), so every new value comes
from the config on resume while the weights, Adam moments and frame count carry
over.

**`collector.type` cost most of the throughput.** Under `sync` the collector
builds one batched environment and runs the policy forward in the main process,
which the Slurm script pins to one thread with `OMP_NUM_THREADS=1`. The 32
workers only step environments, so a deeper network serializes the whole
collector. Under `multi_sync` each worker holds its own policy copy
(`src/training/collectors.py:271`). Measured on the rtx6000 nodes:

| run | collector | workers | layers / ff | fps |
| --- | --- | --- | --- | --- |
| `tf-ptr-weighted-15m-s42`, six runs | multi_sync | 32 | 1 / 256 | 894-925 |
| `fixed-deck-frozen-30m-s42` | multi_sync | 16 | 1 / 256 | 536-554 |
| `fixed-deck-expert30-frozen-kl005-30m-s42` | multi_sync | 16 | 2 / 512 | 536 |
| e4p5tnwg, first segment | sync | 32 | 2 / 512 | 258 |

Row three is this run's architecture and anchor, holding 536 fps out to 16.3M
frames, and it matches the small model exactly. Neither the second transformer
layer nor the KL anchor costs throughput. Only the collector did.

**The league was 250,000 frames wide.** `pool_size` times `snapshot_interval` is
how much history the learner plays against, because the pool keeps the newest
members and evicts the rest
(`src/env/opponents/snapshot_opponent_pool.py:113`). At 5 x 50,000 that is about
four minutes of training at 900 fps, so every opponent is a near-copy of the
learner and `train/win_rate` pins near 0.5 whatever the policy does. For
comparison, `weighted_field.yaml` uses 24 x 1,000,000 and
`ppo_fixed_deck_finetune.yaml` uses 10 x 500,000.

10 x 500,000 rather than 24 x 1,000,000 because a snapshot is 10.1 MB at
2,525,675 parameters and is held per worker: 24 members across 32 workers adds
about 6 GB against the 64 G request, where 10 adds about 1.6 GB. On resume the
pool refills from the 10 newest snapshots on disk, written at the old cadence,
so the league starts at a 500,000-frame span and widens toward 5M.

**`eval_deterministic: false` made the first segment's eval numbers a floor.**
The learner sampled its actions while every reference played argmax. Entropy sat
at 0.882, an effective `exp(H / 0.855)` = 2.80 options against 4.90 mean legal,
so a large share of the learner's evaluation actions were not its argmax. No
`eval/*` number below the resume is comparable with anything logged after it.

**`entropy_coeff` ran at 0.02, and outweighed the anchor 3 to 1.** The entropy
term contributed `entropy_coeff × H` = 0.02 × 0.882 = 0.0177 to the loss against
the anchor's `kl_anchor_coeff × kl_to_bc` = 0.05 × 0.12 = 0.0060, which made the
run's one new mechanism the minority term in its own experiment. At 0.005 the
entropy contribution is about 0.0044 and the ordering reverses. Expect
`train/entropy` to fall from 0.882 and the effective option count to drop from
2.80 toward 1.5 to 2.0.

One diagnostic disagreed and is recorded because it may still be right:
`|loss_entropy| / |loss_objective|` measured 0.355 median (p10-p90 0.31-0.40)
over the first segment, inside the 0.24-0.38 band earlier runs recorded at
0.005, because this run's `loss_objective` is about 3x larger. By that measure
the entropy term was already pulling as hard as in the runs that worked, and
0.005 puts it at roughly 0.089, below anything previously run.

**None of this diagnoses the flat evaluation curves.** Over the first segment
the first eight evaluation points against the last eight moved 0.341 to 0.328 on
`first_snapshot`, 0.822 to 0.813 on `checkpoint` and 0.925 to 0.934 on `random`,
all inside the 0.075 standard error of a 40-episode measurement, while
`train/kl_to_bc` held at 0.12, explained variance sat at 0.70 and no non-finite
gradients appeared. The machinery ran correctly and the policy did not visibly
improve. Because the learner was sampling against argmax references, those
numbers are a floor, so whether it improved is not known either way.

`first_snapshot` survives the resume: it resolves by globbing `checkpoint_dir`
and taking the lowest frame number (`src/training/self_play.py:207`), and the
checkpoint directory lives in the run directory, which the resume reuses. The
reference stays the 50,000-frame snapshot on both sides of the boundary.

## Evaluation during the run

`train.eval_interval: 500000` frames, `eval_episodes: 40`, by argmax
(`eval_deterministic: true`) so evaluation matches how the policy is submitted
and matches `opponent_action_selection: greedy` on the other seat. Three fixed
opponents, logged under `first_snapshot/`, `checkpoint/` and `random/`:

- `first_snapshot` is the first snapshot the run freezes, which is the clone.
  This is the number that answers whether reinforcement learning improved on
  what it started from.
- `checkpoint` is pin-175.7M, the strongest self-play-only checkpoint. The clone
  already beats it 0.967 over 60 games, so this tracks whether that margin
  holds. It is rebuilt from the config it embeds, so its 1-layer, ff-256
  architecture keeps loading against this run's 2-layer, ff-512 model.
- `random` stays comparable across runs and does not depend on which snapshot
  was frozen, but it saturates once the policy reliably beats it.

The collected `win_rate` is pinned near 0.5 by construction under self-play and
says nothing about improvement. Ignore it.

## Launching on Habrok

The project runs from scratch storage, `/scratch/s4325621/pokemon-tcg-ai`, not
from home. `paths.data_dir` and `paths.output_dir` are relative (`decks` and
`outputs`), so running from that directory resolves everything to scratch with
no overrides. Hyperparameters live in
`conf/experiment/kl_anchored_selfplay.yaml`.

### Sync

```bash
./scripts/habrok_sync_kl_anchored.sh
```

It sends the code with `rsync --delete`, then three things the code sync
deliberately excludes:

| what | why it needs sending separately |
| --- | --- |
| `decks/` | gitignored, so it never arrives with the code |
| `outputs/bc/bc-v6-submit.pt` | `/outputs` is excluded from the code sync |
| `.../snapshot_000175702016.pt` | the eval reference, same exclusion |

The script fails before touching anything if one of the three is missing
locally, and lists them on the remote afterwards. It never sends
`train_state.pt`, so it cannot overwrite a run already in progress.

### Submit

```bash
cd /scratch/s4325621/pokemon-tcg-ai
sbatch slurm-conf/train_kl_anchored_smoke.sh   # 30 min, any GPU
```

The smoke job requests no GPU type, so it takes whatever is free and queues
briefly. 200,000 frames with evaluation forced to every 50,000 frames and 6
episodes, which exercises all three eval opponents. `ATTEMPTS=1`, so a failure
surfaces instead of being retried away. It prints `SMOKE OK` on success.

Only after that passes:

```bash
sbatch slurm-conf/train_kl_anchored.sh         # 24 h, RTX Pro 6000
```

It uses a separate run directory and W&B group, so the smoke test cannot
collide with it.

```bash
squeue -u $USER
tail -f slurm-conf/logs/pokemon-tcg-kl-smoke_<jobid>.out
tail -f slurm-conf/logs/pokemon-tcg-kl-anchored_<jobid>.out
```

### Resuming, and changing hyperparameters while doing it

```bash
RESUME=1 sbatch slurm-conf/train_kl_anchored.sh
```

This continues from `train_state.pt` after the time limit, after a crash, or
after a deliberate `scancel`. Without `RESUME=1` the wrapper refuses to start
when `train_state.pt` exists, rather than silently continuing a run
(`scripts/train_kl_anchored_selfplay.sh:77`).

The resume picks up any config change. `scripts/train_supervised.sh:178`
appends `train.resume_state=<run dir>/train_state.pt`, and
`_resolve_warm_start` prefers `resume_state` over `init_checkpoint`
(`src/trainer_builder.py:380`), so the clone is not reloaded over the trained
weights. Only weights, Adam moments, frame count and the RNG state come from the
file; everything else is read fresh from the config. `--total-frames` stays an
absolute target, with the supervisor subtracting what the state already records.

Two things do not survive a mid-run change of this kind. Evaluation curves are
only comparable across the boundary if `eval_deterministic`, `eval_episodes`,
`eval_opponents`, `env.deck_pool` and `env.eval_agent_deck` all stay fixed;
changing any of them makes the metric a different quantity. And the queue is the
real cost: cancelling an allocation to resubmit means waiting for another
`rtx_pro_6000`, so check `squeue --start -u $USER` before cancelling anything.

### What stops the run failing

| guard | catches |
| --- | --- |
| clone checkpoint check | a run that would train from scratch anchored to random weights and still look healthy |
| deck pool check | an empty or missing `decks/expert_pool_30` |
| eval reference check | a missing `snapshot_000175702016.pt`, which otherwise raises at the first evaluation, 50,000 frames in |
| `pgrep -u "$(id -u)"` | a second run of your own competing for the GPU, without tripping on other users' jobs |
| CUDA and engine preflight | a node where PyTorch cannot see the GPU, or the vendored engine will not load |
| `--attempts 10` | a crash mid-run; the supervisor restarts from `train_state.pt` |

`train_state.pt` is written every 250,000 frames and league snapshots every
50,000, so a crash costs a few minutes at most and there is always a
submittable checkpoint.

### A short local probe

```bash
uv run python -m src.train --config-name ppo_selfplay_multideck \
  +experiment=kl_anchored_selfplay collector.total_frames=100000
```

# Known limitations

Deck choice is not cloned. The extractor drops the deck-selection step, so the
clone plays but does not choose a list. Submission uses a fixed deck.

The corpus mixes hundreds of teams and 198 decklists, so the network fits a
population average over teachers who disagree. The visible board identifies the
archetype in part, which limits the damage, but AlphaStar conditioned on a
strategy statistic for this reason.

Both seats of every game are cloned, including the loser. `--winners-only`
exists in `tools/bc/bc_extract.py` and has not been tested.

The clone is strong on decks it has seen and mediocre elsewhere: 0.967 on
`expert_top1` against 0.350 on `alakazam-dudunsparce`. Two days of ladder data
fix the meta it knows.

Behaviour cloning cannot pass its teachers. The corpus median is 1043, so
perfect imitation lands near 1043. Going beyond needs search or the anchored
self-play above.
