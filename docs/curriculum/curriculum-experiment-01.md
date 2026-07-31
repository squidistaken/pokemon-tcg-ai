# Curriculum experiment 01 — PLR vs uniform, single seed

# TODO THIS WILL BE UPDATED WITH NEWER RESULTS

Run 2026-07-31. W&B group `curriculum_ab_20260731_185835`
(`curriculum-s0` = `efot5bq9`, `uniform-s0` = `v5k2mgpk`).

**Conclusions**

1. **The PLR implementation is validated.** Every mechanical check passes; the
2. **The run was too short for PLR to show an effect.** The buffer did not
   finish maturing until 65% of the way through training, so prioritisation was
   only active for the final third.

---

## Setup

| | |
|---|---|
| Arms | `curriculum` (`env=curriculum_v2`) vs `uniform` (`env=multideck_v2`) |
| Difference | `env.curriculum.enabled` only — same corpus, split, and seed |
| Seeds | **1** (seed 0) |
| Frames | 2,002,944 per arm (~15,000 episodes) |
| Workers | 32 |
| Corpus | merged `decks/` + `decks_v2/`, archetypes with ≥15 lists: 37 archetypes, 1,759 decks, **1,369 levels** |
| Holdout | 20% of lists, evaluation only |
| Curriculum | `min_visits=3`, `score_temperature=0.9`, `staleness_coefficient=0.3`, `capacity=4000` |
| League | self-play, PFSP (`hard`, η=2), snapshot every 50k frames |
| Evaluation | every 100k frames (19 rounds), 20 episodes each vs **two** references |

Command:

```bash
python -m scripts.curriculum_ab --arms curriculum,uniform \
    --seeds 1 --frames 2000000 --workers 32 \
    --eval-interval 100000 --eval-episodes 20
```

The `fixed` (single-deck) arm was dropped: it evaluates on its own training deck
while the pooled arms evaluate on held-out lists, so its number was never a
like-for-like comparison, and the budget was better spent elsewhere.

### Observation encoding changed for this experiment

Both arms ran with a **reworked `StructuredObsAdapter`**, so these win rates are
not comparable with any run predating it.

Previously every padded table was flattened and concatenated, so empty option
slots and empty zone positions each consumed input width. Now each entity —
one option row, one Pokémon row, one zone card — is projected to `entity_dim=64`
by a shared `Linear`, and the group is collapsed by **masked-mean pooling over
real entities only**, with padding excluded via the existing masks. Zones also
contribute a fill fraction. The `nn.Embedding` tables for cards, attacks and
categories are unchanged.

| | before | after |
|---|---|---|
| adapter output width | 13,297 | **824** |
| first-layer MLP weights | 3.4M | **211k** |
| throughput | 688 fps | 741 fps (+7.7%) |

Output width no longer depends on `max_options` or zone capacity, so padding
costs no parameters. Setting `pool=False` instead returns the per-entity token
sequence for a future Transformer backbone; that path was **not** used here.

This is a confound against historical runs, not between the arms — both used the
identical adapter, so the curriculum-vs-uniform comparison is unaffected.

## 1. The curriculum worked

| check | value | interpretation |
|---|---|---|
| `sampling_fidelity` | mean 0.769, median 0.708, 70% of batches > 0.5 | workers drew from the published distribution |
| `curriculum/matured` | **1369 / 1369** | every matchup measured |
| `curriculum/visits_min` | 3 | no level left unvisited |
| `curriculum/visits_mean` | 12.9 | ~13 episodes per matchup |
| `curriculum/orphan_commits` | 0 | no episode mis-attributed |
| r(draws, buffer index) | ≈ 0.00 | the pre-fix defect is gone (was +0.89) |

For contrast, the pre-fix run had one matchup absorbing 12.8% of all training
while the highest-scoring level received 3 episodes. This is the first genuine
PLR run in the project.

## 2. Outcome: no detectable difference

`eval/random/win_rate` is the only metric comparable **across** arms (see §4).

| window | curriculum | uniform | diff |
|---|---|---|---|
| all 19 points | 0.691 | 0.620 | **+0.071** |
| last 10 | 0.705 | 0.633 | +0.072 |
| last 5 | 0.700 | 0.657 | +0.043 |

Paired per-round differences: mean **+0.071**, sd 0.177, range [−0.150, +0.400],
sign split **11 positive / 7 negative / 1 tied**.

Pooling episodes over the last 5 rounds (~100 per arm):

> curriculum 70/100, uniform 66/100 → **diff +0.04, 95% CI [−0.089, +0.169]**

The interval comfortably spans zero. The direction favours the curriculum; the
effect is not distinguishable from noise.


## 3. Convergence

The buffer's maturation schedule:

| levels matured | frames | fraction of run |
|---|---|---|
| 25% | 364,544 | 18% |
| 50% | 638,976 | 32% |
| 90% | 1,122,304 | 56% |
| **100%** | **1,302,528** | **65%** |

Until a level clears `min_visits` it scores `inf`, so the sampler is performing
optimistic *exploration* rather than prioritisation. **PLR only genuinely
prioritised for the final ~35% of training.**

This is much worse than the pre-run estimate, which was 27% of the run. That
estimate used the theoretical floor: 1,369 levels × 3 visits = 4,107 episodes,
i.e. every episode landing on a distinct not-yet-matured level.

PLR cannot hit that floor. It samples from a distribution **with replacement**,
so episodes keep landing on levels that are already matured or re-hitting the
same unmatured one; the score term deliberately concentrates repeat visits, and
staleness only nudges neglected levels back rather than sweeping the space
systematically. Measured cost:

| | episodes |
|---|---|
| theoretical floor (1,369 × 3) | 4,107 |
| **actually needed** | **11,299** |
| ratio | **2.75×** |

### Fixed afterwards

`LevelBuffer.distribution()` now runs two regimes. While any level is below
`min_visits` it puts all mass **uniformly on the least-visited unmeasured
levels**, sweeping the space in tiers instead of letting `inf` scores compete in
the rank distribution. Once everything is measured, the normal rank + staleness
mixture takes over.

Re-measured on the same config (650k frames, 16 workers;
[run](https://wandb.ai/pokemon-tcg-ai/pokemon-tcg-ai/runs/7yvq9zy3)):

| | maturation | episodes | vs floor | exploitation window of a 2M run |
|---|---|---|---|---|
| rank-based coverage | 1,331,200 frames | 11,567 | 2.82× | 35% |
| **systematic sweep** | **638,976 frames** | **6,434** | **1.57×** | **68%** |

Not the full 1.0× because 16 workers draw concurrently between publishes, so a
tier can be over-sampled before the next distribution lands. Still roughly
doubles the time spent actually prioritising.

The sweep also removes the tie-break-by-buffer-position artefact: equally
visited levels now receive equal mass rather than being ordered by index.

### Are the curves still rising?

| arm | window | slope (/M frames) | p |
|---|---|---|---|
| curriculum | all 19 | +0.023 | 0.658 |
| curriculum | last 10 | −0.056 | 0.716 |
| uniform | all 19 | +0.074 | 0.191 |
| uniform | last 10 | +0.073 | 0.469 |

Every slope is indistinguishable from zero. At 20 episodes per point the eval
curves are too noisy to separate "still improving" from "plateaued". So the claim
that training had not converged is *plausible and consistent with the maturation
timing*, but it is **not demonstrated by the evaluation data**.

`train/win_rate` sat at 0.459 → 0.468 (curriculum) and 0.445 → 0.480 (uniform),
flat as expected: under self-play the league tracks the learner, pinning the
collected win rate near 0.5 by construction. It is not a progress measure.

## 4. Metric caveats

- **`eval/first_snapshot` is not comparable across arms.** Each arm is scored
  against *its own* 50k-frame snapshot, so the two face different opponents. It
  is a within-arm progress curve only. For the record it read curriculum 0.540 vs
  uniform 0.650 over the last 5 rounds — *opposite* in direction to the random
  metric — which is precisely why it must not be read as cross-arm evidence.
- **The evaluation does not match the deployment objective.** It reports a
  uniform mean over *mirror* matchups on held-out lists, whereas at submission a
  single deck is chosen and faces a varied field: the objective is
  `max_d E_field[win rate | agent plays d]`. Mean-over-mirrors is blind to the
  matchup advantage that deck selection exploits, and would wash out a spikier
  skill profile — plausibly the exact signature of a working curriculum.
- **Held-out decks are unseen *lists*, not unseen archetypes.** 35 of 37
  evaluation archetypes also appear in training, so this measures
  within-archetype generalisation.
- Per eval point, n = 20 gives a binomial SE of ≈0.11 at p = 0.5.

## 5. What would settle it

The cheapest improvement is not more frames but **faster maturation**, so a given
budget buys a longer exploitation window:

- **Raise the archetype threshold to ≥20 lists** → 27 archetypes, 729 levels.
  Maturation cost roughly halves; at 2M frames PLR would prioritise for ~80% of
  the run rather than 35%. Keeps 1,587 of 2,071 decks.
- **Or 4M frames** at the current 1,369 levels (~2.5 h/arm).

Then seeds. Note the scale of the problem: at the observed +0.071 with sd 0.177,
roughly **25 seeds** would be needed to resolve an effect that small. If the true
benefit is a couple of points, it is not worth chasing at this budget — widening
the exploitation window is the higher-leverage change, and a deployment-shaped
metric (§4) is more likely to reveal an effect than more seeds on this one.

Also worth doing before the next run: `scripts/curriculum_ab.py` uses
`capture_output=True`, so per-batch metrics are invisible until an arm finishes.
Tee-ing each arm's output to its log would allow live monitoring without W&B —
during this run, curriculum health had to be verified by re-analysing the state
dumps instead of reading `sampling_fidelity`.
