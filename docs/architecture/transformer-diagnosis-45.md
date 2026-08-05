# Transformer diagnosis 45 — nine arms, one result

Run 2026-08-02. W&B group [`transformer-diagnosis-45`](https://wandb.ai/pokemon-tcg-ai/pokemon-tcg-ai/groups/transformer-diagnosis-45)
(13 runs, 9 arms, 4 of them with a duplicate). Launched by
`scripts/run_tf_diagnosis.sh`; raw data downloaded by
`scripts/fetch_wandb_runs.py` into `analysis/wandb/`.

**Conclusions**

1. **One arm out of eight beat the reference outside noise: `tf_pointer`.**
   Capacity, pre-LN, CLS pooling, per-entity tokens, sub-batch size and the
   LR/entropy schedules are all statistically indistinguishable from the
   control.
2. **The sweep measured its own noise floor and it is larger than seven of the
   eight effects.** Two runs of a byte-identical config at seed 42 differ by
   0.19 (eval/random) and 0.22 (eval/first_snapshot) on their last four eval
   rounds. A 9-arm single-seed sweep cannot resolve what it was built to
   resolve.
3. **The premise "the transformer underperforms the MLP" is unproven, and the
   best available evidence contradicts it.** Per frame the two are within
   0.036 (noise floor ±0.234). The transformer's real deficit is throughput:
   439 vs 1068 fps, a 2.4× penalty.
4. **The one arm that worked does not use attention.** `option_tokens=True`
   projects option rows with a single `Linear` and hands them straight to the
   head — they never enter the encoder
   ([transformer.py:492-494](../../src/models/transformer.py#L492-L494), the
   `encoded_option_repr=False` default path — see "What changed since" below).
   The pointer result should port to the MLP backbone unchanged, at 2.4× the
   throughput.

---

**What changed since (2026-08-05).** This document's findings and numbers are
unchanged — it analyzes the runs as they actually happened — but a follow-up
adversarial review found, and fixed, two correctness defects these arms could
not have known about, both now covered by regression tests (367 passed; the
suite's one failure, `tests/test_callbacks.py`, is a pre-existing wandb
startup/collector-shutdown issue unrelated to this work):

- **No per-entity token carried seat/zone identity.** Swapping the two
  players' Pokémon rows left `state_repr` bit-identical (measured
  `0.000e+00`); the same card in `my.hand[0]` vs `my.discard[0]` produced
  identical tokens. This directly hits `tf_tokens` (§2, §9.10) and the
  `token_groups: [pokemon]` half of `tf_combined` — see those sections'
  "what this run could not have tested" notes in
  `conf/experiment/tf_tokens.yaml` / `tf_combined.yaml`. Fixed by an
  unconditional per-slot segment embedding
  (`StructuredObsAdapter.group_segment_ids`); seat swap now moves
  `state_repr` by `4.857e-02`.
- **Option validity was `card_id != 0`, which is 0 for the stop action
  itself** (an END-type option, like YES/NO/NUMBER/RETREAT) as well as for
  padding. The pointer head gave the stop slot and every padded slot the
  identical logit (`0.056268`) — this hits `tf_pointer` and the
  `option_tokens: true` half of `tf_combined` (§2, conclusion 4). Fixed by
  keying validity on the encoder's own presence signal (`cats[..., 0] != 0`)
  plus an always-true stop slot, and giving the stop slot its own segment
  embedding.

Also fixed: `replace_pooled` and `encoded_option_repr` now exist (§9.10's redesign
request and the "still untested" option-self-attention item respectively),
`MLPBackbone` can emit `option_repr` (§9 P1.5), and construction now validates
`token_groups` eagerly (conclusion-adjacent, not itself a conclusion here).
None of this changes any number in this document — it changes what the
**next** sweep can measure. That sweep is `pointer-transfer-45`
(`scripts/run_ptr_sweep.sh`, arms in `conf/experiment/ptr_*.yaml`), which
re-runs `tf_pointer` and the redesigned `tf_tokens`/option-attention arms at
three seeds with both defects fixed. It grew to nine 4M arms plus a two-arm
12M tier after a pass over §9's "Still untested" list turned up two items the
original nine arms (mapped 1:1 in that list) didn't close: `pooling:
attention` (`ptr_tf_poolattn`) and a matched MLP control past 4M
(`ptr_mlp_long`, alongside `ptr_tf_long`). See that list for the full mapping.

---

## Setup

| | |
|---|---|
| Arms | `reference` (control) + 8 overlays from `conf/experiment/tf_*.yaml` |
| Base config | `conf/ppo_transformer.yaml` — transformer trunk, PPO self-play, PLR curriculum |
| Seeds | **1** (seed 42, `set_seed: true`) |
| Frames | 4,000,000 per arm (~977 logged steps) |
| Workers | 32 on 16 CPUs, A100 per arm |
| Curriculum | `env=curriculum_v2`, 19-deck pool → **361 levels** |
| League | self-play, PFSP (`hard`, η=2), snapshot every 200k frames, pool 5 |
| Evaluation | every 250k frames (15 rounds), **40 episodes** vs `first_snapshot` and `random` |
| Cross-play | off (`train.cross_play: false`) |

Two runs outside the group are used as context: `ppo-transformer-linear-s42`
(`transformer_tests`, the pre-sweep run that died at 1.32M) and
`curriculum-8h-s0` (`decks-mar2026-8h`, MLP backbone, 22M frames) as the
nearest available MLP reference. **The sweep contains no matched MLP arm** —
`conf/ppo_transformer.yaml`'s own header says to launch `model/backbone=mlp` as
the counterpart, and it was never launched.

### The noise floor comes first

Three arms have a duplicate. `tf-schedule` has **two complete 4M runs whose
configs differ in nothing but the W&B run id and output directory** — same
`seed: 42`, same `lr`, verified key-by-key against `index.json`. The
differences below are therefore pure nondeterminism: 32 forked `ParallelEnv`
workers, per-worker PFSP tallies, cuDNN.

| pair | metric | \|Δ run mean\| | \|Δ last-4\| |
|---|---|---|---|
| `tf-schedule` (4.00M vs 4.00M) | eval/random | 0.141 | **0.191** |
| `tf-schedule` | eval/first_snapshot | 0.142 | **0.219** |
| `tf-capacity` (4.00M vs 3.90M) | eval/random | 0.140 | 0.132 |
| `tf-capacity` | eval/first_snapshot | 0.088 | 0.062 |
| `tf-combined` (4.00M vs 2.24M, 8 common rounds) | eval/random | 0.056 | 0.025 |
| `tf-combined` | eval/first_snapshot | 0.081 | 0.000 |

Pooled run-to-run **σ ≈ 0.084** (eval/random) and **0.076**
(eval/first_snapshot) on a 15-round, 600-episode run mean. So the 95% interval
on a **difference between two single runs is ±0.234 / ±0.210**.

The binomial SE on one 40-episode round is 0.056 at p=0.85; on a 600-episode
run mean it would be 0.015. The observed σ is **5.8× that**, so evaluation
sampling is not the dominant noise source — trajectory divergence between
nominally identical runs is. **More eval episodes will not fix this. Seeds
will.**

---

## 1. Results

Ranked by `eval/first_snapshot/win_rate` run mean, the channel that separates
(see §4 for why `eval/random` does not). `last4` = mean of the final four eval
rounds.

| # | Arm | id | state | frames | rand mean | rand last4 | **snap mean** | snap last4 | train/win_rate | fps | A100-h/4M |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **pointer** | `dabvuo2o` | finished | 4.00M | 0.862 | 0.900 | **0.928** | 0.969 | 0.615 | 376 | 2.96 |
| 2 | **combined** | `5cordvnn` | finished | 4.00M | **0.920** | **0.925** | 0.830 | 0.887 | 0.601 | 303 | 3.67 |
| 2b | combined *(dup)* | `znw5buk3` | crashed | 2.24M | 0.881 | 0.900 | 0.856 | 0.888 | 0.795 | 194 | — |
| 3 | capacity | `tbypebf3` | finished | 4.00M | 0.793 | 0.818 | 0.683 | 0.675 | 0.541 | 405 | 2.75 |
| — | *MLP ref, 22M* | `mk7vio33` | finished | 22.0M | 0.744 | 0.794 | 0.665 | 0.719 | 0.517 | **1068** | **1.04** |
| 4 | **reference** | `dmmao1bh` | finished | 4.00M | 0.737 | 0.659 | 0.614 | 0.592 | 0.497 | 439 | 2.53 |
| 4b | capacity *(dup)* | `l1wi6yae` | crashed | 3.90M | 0.653 | 0.686 | 0.595 | 0.613 | 0.518 | 336 | — |
| 5 | schedule *(dup)* | `90yd8wpx` | finished | 4.00M | 0.554 | 0.603 | 0.593 | 0.688 | 0.508 | 387 | 2.87 |
| 6 | preln | `6olv083e` | finished | 4.00M | 0.706 | 0.749 | 0.538 | 0.612 | 0.511 | 427 | 2.60 |
| 7 | subbatch | `dyea1wgw` | finished | 4.00M | 0.720 | 0.658 | 0.504 | 0.453 | 0.478 | 380 | 2.93 |
| 8 | tokens | `h8qxc73y` | finished | 4.00M | 0.708 | 0.688 | 0.495 | 0.569 | 0.499 | 409 | 2.72 |
| 9 | pooling | `ipwqag8g` | finished | 4.00M | 0.763 | 0.773 | 0.461 | 0.484 | 0.485 | 383 | 2.90 |
| 10 | schedule | `0yq4ev3k` | finished | 4.00M | 0.695 | 0.794 | 0.452 | 0.469 | **0.401** | 400 | 2.78 |

`tf-subbatch` `lzprnvcs` (0.25M, SIGINT) logged no eval rounds and is excluded.

**The ranking inverts between the two schedule duplicates — ranks 5 and 10,
identical config, identical seed.** Everything below rank 3 is noise ordering,
not effect ordering.

---

## 2. Per arm: plus, minus, verdict

### `tf_pointer` — the only arm that worked ✅

Hypothesis: `LinearPolicyHead` maps a pooled state to 129 slot logits, so the
policy can only learn "prefer slot k", never "prefer the option that KOs".
**Confirmed.**

| channel | pointer | best non-pointer | non-pointer range | margin | largest same-config Δ | margin / Δ |
|---|---|---|---|---|---|---|
| snap mean | **0.928** | 0.683 | 0.452–0.683 | +0.245 | 0.142 | **1.7×** |
| rand `awr_worst_quartile` last4 | **0.599** | 0.295 | 0.095–0.295 | +0.304 | 0.196 | **1.6×** |
| rand `mean_episode_length` last4 | **65.8** | 86.3 | 86.3–114.9 | −20.5 | 12.4 | **1.7×** |
| rand mean | 0.862 | 0.793 | 0.554–0.793 | +0.069 | 0.141 | 0.5× |

**Plus:** the three runs containing the pointer head (pointer, combined ×2)
occupy the **top 3 of 12** on all of rand mean, snap mean,
`awr_worst_quartile`, `awr_std` (lower), `mean_episode_length` (lower) and
`train/win_rate`, with zero overlap against the nine non-pointer runs. Under
the null that is p = 1/C(12,3) = 0.0045 per channel. It reaches eval/random
≥0.85 at **0.51M frames** (reference: 1.02M) and holds it for 10 of 15 rounds
(reference: 4). Healthiest dynamics in the sweep: grad_norm flat 0.18–0.27,
entropy stable at 0.53–0.58 nats, critic loss 0.021, and **no late regression**.
Costs only −14% fps.

**Minus:** the arm changes two things at once (`model/head=pointer` **and**
`backbone.option_tokens=true`), so the attribution is "per-option
representation + dot-product scoring", not "attention". On `eval/random` and
`train/win_rate` alone its margin is inside the noise floor — the case rests on
the joint evidence across channels, not on any single number.

### `tf_combined` — best eval/random, and under-trained ✅⚠️

**Plus:** eval/random mean **0.920**, best in the sweep, reproduced across both
duplicates (0.920 / 0.881 over common rounds). It answers its own question —
yes, there is an effect worth attributing.

**Minus:** its margin over `tf_pointer` alone (+0.058 rand, **−0.098** snap) is
inside the noise floor, so **the other seven changes contribute nothing
measurable on top of the pointer head**. Worst throughput in the sweep (303
fps, 3.67 A100-h). And its `train/grad_norm` grows monotonically
**0.96 → 4.14**, above the `max_grad_norm: 1` clip from ~0.25M onward in both
duplicates — so it trained ~95% of the run clipped, at an effective LR
decayed a further 4× on top of `lr_anneal`. **Its 0.920 is a floor, not a
ceiling.** Which of the eight stacked changes drives the growth is
unattributed: pointer alone is flat at 0.18–0.27, subbatch alone flat at
0.60–0.77, leaving pre-LN / `final_norm` / 2 layers / CLS as candidates.

### `tf_capacity` — nominally positive, inside noise ⚠️

**Plus:** the only non-pointer arm above the reference on both metrics
(+0.056 rand, +0.069 snap) at just −8% fps.

**Minus:** its duplicate lands 0.084 *below* the reference on rand. **The two
runs of this arm straddle the control in both directions**, so the effect is
undetectable. Its grad_norm also runs 2–3× the reference (0.18 → 0.70 → 0.42).
Best remaining candidate, but needs seeds to say anything.

### `tf_preln` — refuted as necessary ❌

**Minus:** −0.031 rand, −0.076 snap. More importantly, **the pathology it fixes
does not exist**: no grad_norm spike, no loss divergence, no NaN in any of the
15 runs. A 1-layer, d_model=128 post-LN encoder is far below the depth where
warmup matters. **Plus:** it settles the question cheaply (−3% fps) and rules
out normalization as the explanation.

### `tf_pooling` — refuted ❌

**Minus:** +0.026 rand (noise) but **−0.153 snap, third-worst in the sweep**.
Entropy re-inflates late (0.252 → 0.744 nats), the signature of a policy being
pushed around by a strengthening league rather than converging. The mechanism
cannot even engage as the config header describes: with `token_groups: []`,
`padding_mask` is `None`
([transformer.py:470](../../src/models/transformer.py#L470)) and all 10 group
tokens are valid, so CLS replaces one uniform 10-way average with one learned
11-way attention over the same content, through a single layer.

### `tf_tokens` — the headline hypothesis, no signal ❌

**Plus:** the premise is correct and confirmed in code —
`StructuredObsAdapter.__init__` has `pool: bool = True`
([structured_obs_adapter.py:150](../../src/models/structured_obs_adapter.py#L150))
and no run sets `model.adapter.pool`, so the trunk has only ever attended over
ten masked-mean group summaries, never over entities. And expanding 10 → 28
tokens costs only **−7% fps**: the quadratic-attention worry in the config
header is unfounded at this token count.

**Minus:** it does not help. −0.029 rand, **−0.119 snap**, `awr_min` last4 =
**0.00** (dragapult, down from the reference's 1.00), and no archetype-specific
gain anywhere (§7). Note the arm *appends* the 18 entity tokens while keeping
the pooled `pokemon` token
([transformer.py:440-445](../../src/models/transformer.py#L440-L445)), so it
adds capacity rather than replacing pooling — a cleaner test would drop the
pooled token. **Formally untestable at this budget:** a 0.03 effect is 3.5×
below the noise floor, and — unreported at the time — the per-entity tokens
this arm expanded carried no seat identity at all, so even an infinite budget
could not have found the effect it was looking for (see "What changed since"
above).

### `tf_subbatch` — refuted ❌

**Minus:** 16× more optimizer steps bought −0.017 rand and **−0.110 snap**, with
`train/win_rate` drifting *down* 0.583 → 0.477. It measurably changed
optimization — grad_norm pinned at 0.60–0.77 all run (reference 0.15–0.48) and
`loss_objective` 10–15× larger in magnitude, as expected from higher-variance
256-sample advantages. **Plus:** lowest critic loss in the sweep
(0.025 → 0.013), and it establishes that the `sub_batch_size: 4096` drift from
the design doc's 256 was not costing anything. Side effect worth remembering:
**`train/loss_objective` is not comparable across arms with different
`sub_batch_size` or `lr`.**

### `tf_schedule` — untestable, and its premise was wrong ❌

**Plus:** `ent_anneal` works exactly as designed — entropy falls to 0.013–0.016
nats by 4M against the reference's 0.492.

**Minus:** the motivating observation is factually wrong. The reference's
entropy did **not** "sit at ~1.0 nats and barely move" — it decayed 1.28 → 0.49
nats unaided (the MLP reference: 1.33 → 0.31). There was less residual
stochasticity to remove than assumed, and run `0yq4ev3k` has the sweep's
**worst `train/win_rate` (0.401)** after entropy hit 0.016 nats while the PFSP
league kept growing. The two duplicates disagree by 0.14 on both metrics with
opposite signs — this arm is where the noise floor came from.

---

## 3. Transformer vs MLP: the comparison the sweep is missing

`curriculum-8h-s0` matches `tf-reference` on **every hyperparameter that could
confound the learning comparison** — verified key-by-key: `num_workers` 32,
`snapshot_interval` 200k, `eval_interval` 250k, `eval_episodes` 40,
`pool_size` 5, all PFSP settings, `lr` 3e-4, `sub_batch_size` 4096,
`num_epochs` 4, `clip_epsilon`, `entropy_coeff`, `max_grad_norm`, `gamma`,
`lmbda`, all `env.curriculum.*`, `embed_dim` 128, adapter settings, value head,
`LinearPolicyHead`, holdout fraction and split seed. The 22M budget does not
contaminate its first 4M because `lr_anneal` and `ent_anneal` are off in both.

Two confounds remain, and they are real: **seed 0 vs 42** (worth ±0.165 on a
single-run mean by this sweep's own measurement) and **a 15-deck vs 19-deck
corpus** — 225 vs 361 curriculum levels, a different task and a different eval
deck distribution, a systematic offset of unknown sign rather than noise. (Per
`decks/` being a fetched artifact, the corpus shifted between the two runs.)
Comparison is legitimate only at **frames ≤ 4.07M**, where the identical eval
intervals make rounds line up.

| | rand mean | rand last4 | snap mean | snap last4 | fps |
|---|---|---|---|---|---|
| MLP ref, first 4.07M (16 rounds) | 0.773 | 0.694 | 0.592 | 0.637 | **1068** |
| tf-reference, 4.00M (15 rounds) | 0.737 | 0.659 | 0.614 | 0.592 | 439 |
| Δ (transformer − MLP) | −0.036 | −0.035 | +0.022 | −0.045 | **−59%** |
| 95% on a two-run difference | ±0.234 | — | ±0.210 | — | — |

**Every per-frame difference is 5–10× inside the noise floor.** Read this not
as "the transformer matches the MLP" but as "this experiment cannot distinguish
them, and the burden of proof is on the transformer because it costs 2.4× per
frame". At matched wall clock the MLP gets 2.4× the frames, which is a real
disadvantage the sweep never priced in.

`ppo-transformer-linear-s42` differs from `tf-reference` in only 7 keys, 5 of
which are backbone knobs that did not exist yet and default to the reference
values, so it is effectively a **fourth duplicate of the reference arm**:
eval/random 0.800 over its 5 rounds vs the reference's first-5 mean of 0.745 —
another point inside the noise band.

---

## 4. Metric caveats

**`eval/random` is compressed, not censored.** Over 86 rounds and 22M frames the
MLP reference reaches ≥0.90 only 3 times and never exceeds it; the reference
arm reaches 0.900 once. But `tf_combined` scores ≥0.90 in **13 of 15 rounds and
hits 1.000**. So 0.85–0.90 is a **capability plateau of `LinearPolicyHead`**
that the pointer head breaks — not a task ceiling. The consequence is that
eval/random discriminates poorly in the 0.6–0.9 band, which is exactly where
all eight non-pointer arms live.

**`eval/first_snapshot` is a per-run moving target.** It is the run's *own*
snapshot at 200k frames, so a run with a weaker early checkpoint scores higher
for free. Here the round-1 values (pointer 0.550, combined 0.200, reference
0.550) show the 200k snapshots are not anomalously weak, so the jump to
0.975/0.850 by round 2 is real self-improvement — but that cannot be
established from the metric alone. **Do not resolve this by trusting
first_snapshot.** Three channels with a run-independent referent all agree with
it: eval/random (fixed opponent), `eval/random/mean_episode_length`
(65.8 vs 86.3–114.9 — pointer closes games 24–43% faster, immune to both
opponent-strength confounds and win-rate compression), and
`awr_worst_quartile` (0.599–0.709 vs 0.095–0.295).

**Per-arm fps is not attributable to the arm.** Same-config duplicates differ by
up to **1.56×** from node co-tenancy (combined 303 vs 194 fps; capacity 405 vs
336). Up to five arms ran concurrently. Any fps delta below ~1.3× is inside
that confound — which covers every single-change arm. What survives: the
**2.43× transformer-vs-MLP penalty** (5.5× the largest same-config ratio),
tokens at ≤7%, and 64 optimizer steps/batch at ≤13%.

**A "final round" ranking reads a downslope.** The reference peaks at 0.900 at
2.03M then falls to a 0.659 last-4 mean. The same shape appears in the MLP
reference (0.900 @ 2.03M → 0.694), subbatch, and capacity-`l1wi6yae` — 4 of 4
flat-head runs. As PFSP fills the pool with harder snapshots, the flat-head
policy specializes against the league and loses ground against random. Pointer
(0.675 → 0.950) and combined (flat at ceiling) do **not** show it. This is why
`rand_final` and `rand_last4` disagree so violently (schedule-`90yd8wpx`:
last4 0.603, final 0.436).

---

## 5. Learning dynamics

Entropy in nats (`−train/loss_entropy / entropy_coeff`):

| arm | 0.1M | 1.0M | 2.0M | 3.0M | 4.0M |
|---|---|---|---|---|---|
| reference | 1.280 | 1.014 | 0.695 | 0.509 | 0.492 |
| MLP ref | 1.330 | 1.021 | 0.479 | 0.592 | 0.311 |
| pointer | 1.432 | 0.624 | 0.582 | 0.526 | 0.578 |
| combined | 1.379 | 0.862 | 0.765 | 0.355 | **0.013** |
| schedule ×2 | 1.23/1.32 | 1.12/0.53 | 0.99/0.51 | 0.49/0.25 | **0.016/0.013** |
| pooling | 1.164 | 0.428 | 0.252 | 0.464 ↑ | **0.744 ↑** |
| subbatch | 1.326 | 0.790 | 0.795 | 0.707 | **0.925 ↑** |
| preln | 1.236 | 0.627 | 0.522 | 0.557 | **0.631 ↑** |

No collapse anywhere except by design. But **three arms re-inflate entropy
late** — pooling, subbatch, preln — the policy becoming *more* random after
mid-training. All three are among the four worst on snap. Pointer is flat and
healthy.

`train/grad_norm` (rolling-20 mean, `max_grad_norm: 1`):

| arm | 0.1M | 1.0M | 2.0M | 3.0M | 4.0M |
|---|---|---|---|---|---|
| reference | 0.16 | 0.21 | 0.27 | 0.15 | 0.19 |
| pointer | 0.10 | 0.21 | 0.21 | 0.25 | 0.18 |
| capacity | 0.18 | 0.35 | 0.70 | 0.50 | 0.42 |
| subbatch | 0.76 | 0.60 | 0.59 | 0.71 | 0.77 |
| **combined** | **0.96** | **1.81** | **2.31** | **2.70** | **4.14** |
| combined *(dup)* | 0.93 | 2.05 | 2.37 | 2.62 | — |

Critic loss is tightly clustered with no divergence anywhere (reference
0.037 → 0.013, pointer 0.039 → 0.021, combined 0.019 → 0.023, MLP
0.039 → 0.033). **No arm shows value-function blow-up.**

---

## 6. Crashes: all infrastructure, none a learning failure

| run | ended | frames | state | `summary/*` written |
|---|---|---|---|---|
| `l1wi6yae` (capacity) | **17:09:50** | 3.90M (97.5%) | crashed | **no** |
| `znw5buk3` (combined) | **17:09:30** | 2.24M | crashed | **no** |
| `ipwqag8g` (pooling) | 17:09:54 | 4.00M | finished | yes |
| `lzprnvcs` (subbatch) | 14:04:13 | 0.25M | finished | yes |
| `6olv083e` (preln) | started **14:04:52** | 4.00M | finished | yes |

`Trainer.train()` has a `finally:` that always calls `on_train_end`, which is
what writes the five `summary/*` keys
([trainer.py:273-291](../../src/training/trainer.py#L273-L291)). Every other run
has them; **these two have none**, so no Python exception propagated — the
processes were killed externally. They died **19 seconds apart** after 3.2 h of
independent execution, while a third run on the same node-group completed at
17:09:54. A shared node-level event is the only parsimonious explanation.

No metric anomaly precedes either death: fps flat to the last row (337.2 → 336.7
and 195.5 → 194.1), grad_norm in each run's normal band, no NaN, no eviction.
**No collector restart occurred in any run** — `train/frames` increments by
exactly 4096 for every consecutive row across all 15 histories, so
`collector.max_restarts: 20` was never exercised and the engine's
`FixedList::checkIndex` abort did not recur.

`lzprnvcs` is a different case and not a crash: it *did* write `summary/*`, and
the only early-exit path reaching the `finally` with `frames < budget` is
`except KeyboardInterrupt` ([trainer.py:267](../../src/training/trainer.py#L267)).
It took a **SIGINT at 444 s**, and `tf-preln` started 39 s later — consistent
with the job being cancelled or preempted and its GPU taken by the next queued
arm.

**Net: 3 of 14 runs (21%) lost to infrastructure, one at 97.5% completion.**
No SLURM stdout/stderr exists for 2 Aug — `logs/` holds only an 11 Jul train
log and two scraper logs, and the scratch output dirs have no captured streams —
so the proximate cause is unrecoverable.

---

## 7. Curriculum health, and the curriculum itself

Every arm is identical on the structural metrics: `size == matured == 361`,
**zero evictions and zero orphan commits in all 15 runs**, `sampling_fidelity`
0.89–1.26 with no arm outside, `score_std` 0.0072–0.0119 with a between-arm
spread (0.005) smaller than the same-config duplicate spread (0.0034). **No arm
interacted badly with PLR.**

Two findings about the curriculum:

1. **`env.curriculum.capacity: 4000` against 361 levels means the buffer can
   never evict — PLR's replacement mechanism is dead code in this
   configuration.**
2. **Prioritisation is close to uniform after ~1M frames.** `score_std/score_mean`
   ≈ 0.58 with `score_max` only 0.034–0.071, and with `score_temperature: 0.9`
   the weights are nearly flat. Early on there was real spread (0.25M:
   `score_std` 0.0569, `score_max` 0.4395), which collapses ~5× by 4M. **The
   "PLR deck curriculum" is, past ~1M frames, approximately uniform sampling
   over 361 fixed matchups** — a simpler environment than the label suggests,
   and a near-stationary task distribution on which to test claims about
   attention over entities.

One unexplained divergence: `curriculum/anchor_games` reaches 20,461 by 2.22M in
`znw5buk3` but only 6,324 by 3.99M in `5cordvnn` — 3.2× more with half the
frames, diverging from ~1.48M. Same config. The eval metrics agree between the
pair, so it changes no conclusion, but it is a real behavioural difference worth
a look.

---

## 8. Per-archetype: the gain is broad, and entity attention shows nothing

`eval/random` aggregates, last-4 mean:

| arm | `awr_mean` | `awr_min` | `awr_std` | **`awr_worst_quartile`** | snap `awr_worst_quartile` |
|---|---|---|---|---|---|
| reference | 0.663 | 0.000 | 0.375 | 0.095 | 0.162 |
| schedule ×2 | 0.610 / 0.774 | 0.000 / 0.125 | 0.372 / 0.302 | 0.099 / 0.295 | 0.232 / 0.000 |
| capacity ×2 | 0.751 / 0.690 | 0.000 / 0.000 | 0.346 / 0.342 | 0.175 / 0.109 | 0.099 / 0.154 |
| subbatch | 0.671 | 0.083 | 0.356 | 0.135 | 0.055 |
| preln | 0.741 | 0.125 | 0.322 | 0.247 | 0.021 |
| pooling | 0.724 | 0.000 | 0.352 | 0.153 | 0.021 |
| **tokens** | 0.717 | **0.000** | 0.342 | 0.179 | 0.016 |
| **pointer** | **0.909** | **0.250** | **0.218** | **0.599** | **0.896** |
| **combined** | **0.937** | **0.438** | **0.158** | **0.709** | **0.569** |

**No arm has an archetype-specific profile.** Pointer improves 16 of 18
archetypes, and its gain shows up as **compression of the spread** —
`awr_std` 0.375 → 0.218, `awr_min` 0.000 → 0.250,
`awr_worst_quartile` 0.095 → 0.599. It fixes the tail rather than winning a
matchup class. Combined goes further (`awr_std` 0.158, `awr_min` 0.438).

**`tf_tokens` shows the opposite of what the entity-attention hypothesis
predicts.** If relating individual Pokémon mattered, the gain should
concentrate on board-complexity-sensitive archetypes and lift the tail. Instead
`awr_min` is **0.00** (dragapult, from the reference's 1.00) and
`awr_worst_quartile` moves 0.095 → 0.179, a change below the same-config
duplicate spread on that metric (0.196). **This is the sweep's cleanest
negative, and it lands directly on the Issue #45 premise.**

Caveat: `archetype_count` ≈ 14 of 18 per round, so each cell is ~3 games per
round and ~11 over four rounds (SE ≈ 0.15). Only the aggregates are
trustworthy. The MLP reference has no per-archetype columns at all
(`train.eval_per_archetype` absent from its config), so no MLP comparison is
possible here.

---

## 9. What to do next

### P0 — fix the experiment before running more arms

1. **Multi-seed, and drop arms to pay for it.** Measured σ = 0.084 exceeds every
   single-change arm's effect. Three seeds brings the 95% interval on a
   difference to ~±0.135; five to ~±0.105. **Three seeds × 4 arms costs what
   this 9-arm sweep cost** and would resolve effects of the size being hunted.
   The only arm this design found is the one with a ~0.3 effect, which a 2-arm
   comparison would also have found.
2. **Run the matched MLP control.** It is the stated point of the ablation and it
   was never launched. One job, ~1 h (the MLP is 2.4× faster). Until it exists,
   "the transformer underperforms" is unproven.
3. **Add a run-independent strength metric.** `first_snapshot` is a per-run
   moving target; `random` is compressed above 0.85. Freeze one checkpoint as a
   league anchor shared across arms, or enable `train.cross_play` for sweeps, or
   add a scripted opponent. Raise `eval_episodes` too — but that is second-order
   (binomial 0.056/round vs trajectory σ 0.084). **Seeds first, episodes
   second.**
4. **Capture SLURM stdout/stderr and add checkpoint-resume.** Nothing wrote a job
   log; `l1wi6yae` lost 3.2 h at 97.5% and the cause is unrecoverable.

### P1 — the one real lead

5. **Test the pointer head on the MLP backbone.** Highest-value single experiment
   available. Option tokens never pass through the encoder, so nothing about
   the gain requires attention. If `MLPBackbone` + `option_tokens` +
   `PointerPolicyHead` reproduces ~0.86 eval/random, the Issue #45 transformer
   effort is answering the wrong question and the fix is a **head** change at
   1068 fps. If it does not reproduce, the pointer head genuinely needs the
   trunk — also a real finding. Either outcome is decisive. ~1 h × 3 seeds.
6. **Long `tf_pointer`: 15–20M frames, 3 seeds.** Only arm outside noise,
   separating from 0.5M, healthiest dynamics, −14% fps, and not plateaued.
7. **Longer `tf_combined` with `max_grad_norm` raised to 5–10.** It was clipped
   for ~95% of training, so 0.920 is a floor. Or first attribute the grad_norm
   growth — pointer and subbatch alone do not reproduce it, leaving pre-LN /
   `final_norm` / 2 layers / CLS.

### P2 — dead ends, do not re-run as configured

8. **`tf_preln`** (fixes an instability that does not exist at 1 layer /
   d_model 128), **`tf_pooling`** (−0.153 snap; mechanism barely engages with
   `token_groups: []`), **`tf_subbatch`** (16× the steps for −0.017/−0.110,
   grad_norm pinned, train win rate drifting down), **`tf_schedule` as
   configured** (premise factually wrong; its worst run has the sweep's lowest
   `train/win_rate` after entropy hit 0.016 nats).
9. **`tf_capacity` deserves one more look** — better duplicate above the
   reference on both metrics at −8% fps — but bundle it with the pointer work,
   at 3 seeds, not alone.
10. **`tf_tokens` needs redesigning, not rerunning.** (a) Drop the redundant
    pooled `pokemon` token so entity tokens *replace* rather than augment the
    summary; (b) go past `num_layers: 1` — one attention op is the bare minimum
    for a relational claim; (c) pair with a fixed anchor opponent so a 0.03
    effect is measurable at all.

### Still untested (as of this run; see "What changed since" for what now covers each)

- **Whether attention over entities helps at all.** `tf_tokens` at 1 layer,
  1 seed, 4M, augmenting-not-replacing, against a 0.084 noise floor cannot
  answer it. → `ptr_tf_entities`, redesigned per item 10 and re-run at 3 seeds.
- Whether the pointer gain requires the transformer (item 5). →
  `ptr_mlp_pointer`, 3 seeds.
- Whether depth helps (capacity's duplicates straddle). → `ptr_tf_capacity`,
  bundled with the pointer head per item 9, 3 seeds.
- `pooling: attention` — the lighter variant `tf_pooling`'s own header names as
  a fallback was never run. → `ptr_tf_poolattn`, 3 seeds; not part of the
  original nine, added when this list was reviewed for gaps
  (`pointer-transfer-45`).
- Whether the transformer beats the MLP at any budget — no matched control at
  any frame count. → `ptr_mlp_linear`/`ptr_tf_linear` and
  `ptr_mlp_pointer`/`ptr_tf_pointer` at 4M; `ptr_mlp_long`/`ptr_tf_long` at 12M,
  matched frames rather than matched wall clock (see `ptr_mlp_long.yaml` for
  why). `ptr_mlp_long` is likewise new, added for the same reason as
  `ptr_tf_poolattn`: this item said "any budget", and past 4M nothing closed it
  until now.
- Option-token *self-attention* rather than the linear projection the pointer
  arm uses — the design doc's Phase-2 Set Transformer trunk. → `ptr_tf_setattn`,
  3 seeds, gated behind `encoded_option_repr` + `replace_pooled`.

---

## 10. What this data cannot answer

- **Why `l1wi6yae` and `znw5buk3` were killed.** No SLURM logs for 2 Aug. The
  evidence supports "external kill, no Python unwind, same 19-second window";
  node fault vs OOM vs preemption is unrecoverable.
- **Who sent `lzprnvcs` its SIGINT.** The code path is unambiguous; the sender
  is not in the data.
- **Any MLP per-archetype comparison** — the MLP reference lacks
  `train.eval_per_archetype`.
- **Attribution within `tf_combined`** — eight simultaneous changes, and its
  grad_norm growth is reproduced by none of the single arms.
- **Whether the reference's late regression is a self-play pathology or a PFSP
  artifact.** It appears in 4 of 4 flat-head runs, which suggests systemic, but
  there is no matched no-self-play run.
- **Any effect smaller than ~0.2 win rate.** With n=1 and σ = 0.084 that is the
  sweep's resolution. Seven of eight arms produced effects below 0.12.

---

## Reproducing this analysis

```bash
# Download every transformer group plus the MLP reference (needs WANDB_API_KEY in .env)
python -m scripts.fetch_wandb_runs --out analysis/wandb \
    --group transformer-diagnosis-45 --group transformer_tests --group decks-mar2026-8h

# List matching groups without downloading
python -m scripts.fetch_wandb_runs --list
```

`analysis/wandb/index.json` holds per-run config/summary/tags/state;
`analysis/wandb/history/<name>-<runid>.csv` holds the full step series (the run
id is in the filename because relaunched arms reuse `wandb.name`);
`analysis/wandb/arm_summary.csv` holds the per-run aggregate table behind §1.
