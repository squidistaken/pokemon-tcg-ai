# The policy could not see its actions — diagnosis and fix

Investigated 2026-08-06, prompted by 5M-frame runs that plateaued far below a
trivial baseline. This documents what was wrong, how it was measured, and what
changed.

**Headline**

- The policy was **structurally incapable** of choosing an action on its merits.
  It could only learn a prior over option-list *indices*.
- After 5M frames it scored **0.71** against a random opponent. A one-line
  heuristic — always take the first legal option — scores **0.82**.
- The critic was **no better than a constant**: `corr(V, outcome) = 0.067`
  on-policy, MSE 1.008 against 1.000 for predicting zero.
- Both had one root cause: **mean-pooling** in `StructuredObsAdapter`.
- Fixed by a per-option scoring head (`PointerHead`) plus richer set pooling.
  Still plain MLPs — no attention involved.

---

## 1. The defect

`StructuredObsAdapter._encode_options` collapsed the option table with a masked
mean, and `LinearPolicyHead` produced all `max_options + 1` logits from the
pooled state vector alone. A mean is permutation-invariant, so the encoding of a
state carries no correspondence between an option and the slot it occupies.

Measured directly — shuffle the real options of a live state and re-encode:

```
n_options at probe state: 7
permutation applied: [3, 5, 0, 2, 4, 6, 1]
max |Δ| in adapter output over ALL features: 5.96e-08
```

Bit-identical. The network's logits are unchanged while the correct action has
moved. No amount of training can fix this: the information required to tell
"attack" from "retreat" is destroyed before the first weight is applied.

The one thing such a policy *can* represent is a prior over slot indices, and
that is exactly what training produced. Sampling the 5M-frame checkpoint:

| statistic | value |
|---|---|
| argmax is slot 0 | **79.6%** of decisions |
| argmax in slots 0–2 | 90.5% of decisions |
| mean policy entropy (legal actions) | 0.55 nats |

### It converged to a hand-codeable heuristic, and then underperformed it

| policy | win rate vs `RandomOpponent` |
|---|---|
| uniform random | 0.530 |
| **`legal[0]` — one line, no learning** | **0.820** |
| trained checkpoint, greedy | 0.713 |
| trained checkpoint, sampled | 0.717 |
| 5M-frame run, eval vs random | 0.625 |

Reproducing the agent by hand as "pick slot 0 with probability *p*, else
uniform" places it *on the curve*, i.e. it is behaviourally indistinguishable
from a positional prior at matching entropy:

| P(slot 0) | entropy | win rate |
|---|---|---|
| 1.00 | 0.00 | 0.793 |
| 0.80 | 0.76 | 0.803 |
| 0.70 | 1.00 | 0.733 |
| 0.50 | 1.36 | 0.623 |

This also explains why the eval curve *decayed* after ~2.6M frames: the entropy
bonus (`entropy_coeff: 0.02`) holds the policy off the deterministic optimum of
its own hypothesis class, and within that class stochasticity is pure loss.

### The critic was broken by the same mechanism

Zones (`hand`, `discard`, `prizes`) were also mean-pooled, reducing a hand to a
card-embedding centroid — which cannot express *which* cards are held, only what
the average one looks like. Measured on-policy over 2409 states:

```
corr(V, final outcome)      : 0.0667
MSE(V, outcome)             : 1.0084   | constant-zero baseline: 1.0000
V range: [-0.582, 0.803], std 0.171
corr in last 20% of episode : 0.2482
```

Worse than predicting zero, and barely above chance even when the game is nearly
decided. The reported `train/loss_critic: 0.066` was the critic fitting its own
bootstrap targets, not predicting wins — a self-consistent but uninformative
value function. Consequences: GAE advantages were close to noise, and the PLR
curriculum, which scores levels by critic residual, was prioritizing on noise.

**The environment was never at fault.** `StructuredObservationEncoder` already
emits `options` as a per-slot table, row `i` describing action `i`, 87 features
across 129 rows. Every bit the policy needed was present and was discarded on
the model side.

---

## 2. The fix

Three changes, all in `src/models/`, all plain MLPs.

### `PointerHead` (`src/models/heads.py`) — the ceiling-mover

One scorer, shared across slots:

```
logit_i = scorer([state_repr, option_repr_i])       # weights shared over i
logit_stop = stop_scorer([state_repr])              # stop has no option row
```

This makes the policy permutation-**equivariant** instead of invariant: reorder
the options and the logits follow. Weight sharing is also what makes it
learnable — every option seen in any slot trains the same parameters, rather
than each index having to learn its own mapping from its own visits.

Verified as the exact inverse of the defect:

```
permutation: [4, 1, 0, 3, 2]
original logits[:n]: [-0.0734, -0.0906, -0.0853, -0.1020, -0.0492]
permuted logits[:n]: [-0.0492, -0.0906, -0.0734, -0.1020, -0.0853]
EQUIVARIANT (logits followed the permutation): True
stop logit unchanged: True
```

Both directions are pinned by tests — `test_pointer_head_is_permutation_equivariant`
and, deliberately, `test_flat_head_is_permutation_invariant`, so a regression
that silently converged the two heads would fail.

### Option tokens (`StructuredObsAdapter`)

`emit_option_tokens` makes `forward` return `(state_vector, option_tokens)` with
tokens shaped `(*batch, n_slots, entity_dim)`. The pooled option digest stays in
the state vector as legitimate "what can I do right now" context.

It is **not** a config key: `build_actor_critic` derives it from the selected
head, so adapter and head can never disagree.

### Set pooling (`zone_pooling: mean_max_sum`)

Zones and the Pokémon board now contribute `mean ‖ max ‖ sum` rather than the
mean alone — the max answers *is card X present*, the capacity-normalized sum
answers *how many*, neither of which a centroid can express. Adapter output
widens 824 → 1848.

### Cost

| | before | after |
|---|---|---|
| adapter output width | 824 | 1848 |
| policy head | 1 linear, `128 → 129` | scorer `192 → 128 → 128 → 1` applied per slot |

The scorer runs once per option slot per forward pass, so its hidden widths cost
more than the trunk's; `num_cells: [128, 128]` is deliberately modest.

---

## 3. Result

`train=fixed_opponent` against `RandomOpponent`, single mirror deck, seed 0,
600k frames per arm — W&B group `pointer-head-validation-20260806`.

Win rates below are **windowed** (differenced from the cumulative counters),
because `train/win_rate` as logged is an integral from frame 0 and understates
the endpoint.

| | windowed win rate vs random |
|---|---|
| uniform random | 0.530 |
| `legal[0]` heuristic | 0.820 |
| flat head, **5M** frames, self-play (eval) | 0.625 |
| **pointer head, 600k frames** | **0.922** |

Learning curve for the pointer arm:

| progress | frames | windowed win rate |
|---|---|---|
| 25% | 147k | 0.777 |
| 50% | 295k | 0.903 |
| 75% | 442k | 0.923 |
| 100% | 606k | 0.922 |

It clears the `legal[0]` ceiling — the thing the flat head could only
approximate — by ~10 points, on **8× less data** than the run that scored 0.625.
The critic improves in step (`loss_critic` 0.24 → 0.09 over the same run),
consistent with the pooling having been the shared constraint.

---

## 4. What this does *not* fix

`PointerHead` over an MLP trunk is a DeepSets-class model: it scores each option
against a pooled state, but cannot model interactions *between* options or
between an option and a specific board slot ("this attack is good *because*
their active is weak to it"). That needs attention, and is the remaining
argument for the Transformer backbone — but the Transformer was never going to
help on its own. Paired with `LinearPolicyHead` it would have stayed
permutation-invariant and moved nothing.

Also unaddressed, and worth separate attention:

- `train/*` metrics accumulate from frame 0 (`Trainer.train`), so `train/win_rate`
  is an integral and can never show improvement. It wants windowing.
- `eval_episodes=40` gives ±7.9pp standard error — the 5M-frame eval "curve"
  (0.625 → 0.85 → 0.625) is largely noise around the ceiling above.
- `max_options=128` against a measured median of 4 legal options (mode 2).
- The engine seeds from `std::random_device`, so `set_seed=true` does **not**
  make runs reproducible; seeded A/B comparisons are noisier than they look.

---

## 5. Compatibility

`submission/runtime.py` mirrors the architecture for the Kaggle sandbox and was
updated in step. It infers what to build from the checkpoint rather than
assuming — `emit_option_tokens` from the presence of `policy_head.scorer.*`,
`zone_pooling` from the saved adapter config — so pre-fix checkpoints keep
loading as the architecture they were trained with.

Parity is verified by loading one training checkpoint into both paths:

```
strict state_dict load: OK
max |train logits - submission logits| over 60 states: 0.0
```

`model/head=linear` is retained as the control for the comparison above.
