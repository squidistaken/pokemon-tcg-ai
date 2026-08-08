# PPO Actor-Critic — Architecture & Implementation Plan

Design for the first learned agent: **PPO + invalid-action masking + self-play**, built as a
shared-trunk actor-critic with a **pluggable backbone** (MLP → Deep Sets / Set Transformer →
temporal transformer) over a **tokenized, deck-aware observation**.

## Context

PPO + invalid-action masking + self-play is the well-trodden baseline recipe from the
[literature review](../research/rl-tcg-literature-review.md) (Vieira et al., ByteRL,
Huang & Ontañón on masking). The RL environment already exists (`torchrlenv` line of work):
`TCGEnv` (a custom `EnvBase` wrapping the `cg` engine), a `Trainer`/`Collector` harness with an
`_update` hook, an `OpponentPool` self-play scaffold, and passing env/rollout tests. Its
[`docs/torchrl_environment.md`](../torchrl_environment.md) explicitly lists the **PPO
actor-critic + card-aware encoder** as the "not here yet" piece — this document specifies it.

Three design decisions frame everything below:

1. **Built against the documented env contract.** The network is designed against the `TCGEnv`
   tensordict contract; wiring it into the live harness is an integration step (see
   [Prerequisites](#prerequisites--integration-notes)), not a rewrite of the env.
2. **The backbone is a literature-grounded menu, not transformer-only.** The proven, dominant
   network in this space is a **flat MLP** (PPO+masking+self-play on an MLP is *the* recurring
   recipe). Permutation-equivariant **Deep Sets / Set Transformer** are the recognized next step;
   **Decision-Transformer**-style temporal attention is the frontier for history; recurrent
   **LSTM** is a deprioritized backburner fallback. Transformers are *emerging, not established* —
   the MLP baseline must exist and be beaten first.
3. **Tokenized, deck-aware observation.** The placeholder 36-dim flat observation is replaced by a
   card-aware tokenized observation that also **conditions on the deck being piloted** — behavior
   is highly contingent on deck archetype.

## The env contract we build against

Per-step TensorDict keys the policy consumes/produces (`src/env/tcg_env.py`):

| Key | Spec | Notes |
|-----|------|-------|
| `observation` | `Unbounded((36,), float32)` | Placeholder hand-crafted summary; **not** a modeling decision. Replaced by the tokenized obs below. |
| `action_mask` | `Binary(n=(max_options + 1), bool)` | Legal option indices `0..95` plus index `96` = synthetic **stop**. |
| `action` (out) | `Categorical((max_options + 1), int64)` | Action `i < 96` picks option `i`; `96` = stop. |
| `reward` | `Unbounded((1,), float32)` | Terminal-only: +1 win / −1 loss / draw. |
| `done`/`terminated`/`truncated` | `Binary(1, bool)` | `truncated` on the engine-selection safety cap. |

The opponent plays inside `_step`, so it is a standard **single-agent** env. Multi-select
(`minCount..maxCount`) selections are decomposed by the env into sequential single picks; the
policy only ever sees one masked `Categorical` per step.

## Module structure
See README.md for the module structure.


## Architecture

### 1. Base actor-critic class (`src/models/actor_critic.py`)

`class ActorCritic(nn.Module)` = one **shared backbone** + a **policy head** + a **value head**.
Forward once through the backbone, fan out to both heads (weight sharing → sample efficiency,
matches standard PPO / ByteRL). Outputs `logits` (shape `(..., (max_options + 1))`) and `state_value` (`(..., 1)`).

### 2. Pluggable backbone (`src/models/backbone.py`)

One `Backbone` ABC → `forward(obs_td, deck_ctx) -> (state_repr, option_repr)`. Heads and
TorchRL assembly are identical across all implementations, so backbones are swappable via
`conf/model/`. Concrete backbones live in their own modules beside the ABC
(`src/models/mlp.py`, `src/models/transformer.py`).

- **`MLPBackbone`** (`src/models/mlp.py`) — the literature's **dominant, proven** network: flattens
  every observation field (including the structured encoder's nested per-option/per-Pokemon/zone
  tables — card and attack IDs go in as raw floats, no embedding lookup) and concatenates them →
  `torchrl.modules.MLP`. **Implemented and trains against the default `structured` encoder today**
  (as well as the legacy flat 36-dim obs, kept for regression testing) — this naive-flatten
  pairing is the baseline every richer, permutation-invariant backbone below must beat (Vieira et
  al.), not a placeholder blocked on Phase 2.
- **`TransformerBackbone`** (`src/models/transformer.py`) — **implemented** (Issue #45). Attention
  over the *features within one observation*, not over time. Each `StructuredObsAdapter`
  group (`globals`, `options`, `pokemon`, the zone tables, …) becomes one token via its own linear
  projection plus a learned per-group type embedding; `nn.TransformerEncoder` attends across the
  ~10 tokens; the readout gives `state_repr`. Deliberately shallow by default (1 layer, 4 heads) —
  the sequence is short and the MLP is still the control to beat.

  The **first run of this backbone underperformed**, and the diagnosis added knobs for each
  candidate cause (`conf/experiment/tf_*.yaml`, one arm each; see the sweep in
  `scripts/run_tf_diagnosis.sh`). Defaults reproduce that first run exactly:

  | knob | default | alternative |
  |---|---|---|
  | `pooling` | `mean` over tokens | `cls` (learned query token) or `attention` |
  | `norm_first` / `final_norm` | `false` — torch's post-LN, which wants an LR warmup the PPO config lacks | `true` — pre-LN, trains without warmup |
  | `token_groups` | `[]` — attention sees only pooled group summaries | e.g. `[pokemon]`, expanding a group into per-entity tokens |
  | `option_tokens` | `false` | `true` — emits `option_repr` for the pointer head |
  | `replace_pooled` | `false` — the pooled `encode_groups()` token for each `token_groups` name is kept alongside its per-entity tokens | `true` — drop that group's pooled token, so the per-entity tokens are the only route it reaches the trunk by |
  | `encoded_option_repr` | `false` — `option_repr` is the cheap pre-attention per-option projection | `true` — read `option_repr` from the encoder's *output* rows instead (needs `option_tokens` and `"options"` in `token_groups`) |

  `token_groups` is the important one. The adapter's `encode_groups()` masked-mean-pools each
  group *before* the trunk sees it, so with the default `[]` the transformer attends over ten
  group averages — the same 824 features the MLP consumes, merely un-concatenated.
  `encode_entity_tokens()` returns the unpooled entities instead, and — since a follow-up
  correctness pass after the first diagnosis — every per-entity token this produces also carries a
  learned **segment embedding** (`StructuredObsAdapter.group_segment_ids`, unconditional, not a
  flag): which seat a Pokémon belongs to, which zone a card sits in, whether an option slot is the
  synthetic stop action. Before that embedding existed, per-entity tokens carried no such identity
  at all — swapping the two players' Pokémon rows left `state_repr` bit-identical (measured
  `0.000e+00`) — so `token_groups: [pokemon]` could not represent "this Pokémon threatens that
  one" even in principle; a card in `my.hand[0]` and the same card in `my.discard[0]` produced
  identical tokens for the same reason. Both are fixed now (seat swap moves `state_repr` by
  `4.857e-02`), but the fix only reaches the *entity-token* path: with the default `token_groups:
  []`, or under `model/backbone=mlp` in every configuration, groups are still masked-mean-pooled
  before anything sees them, and that pooling erases the same identity the segment embedding
  restores — those configurations remain seat/zone-blind. See
  `docs/architecture/transformer-diagnosis-45.md`'s "what changed since" note and
  `conf/experiment/ptr_tf_entities.yaml` for the redesigned test this enabled.

  `token_groups` is opt-in per group because the padded slot counts are large (434 tokens in
  total, `options` alone being `max_options + 1`), not because attention is unaffordable at these
  lengths: measured unbatched CPU forward (1 thread — the path the league opponent forward runs
  on inside every worker, already 46% of throughput per `docs/training-performance.md`) is 1.53 ms
  for the MLP, 1.90 ms for the transformer's default 10 tokens, 2.30 ms at 28 tokens
  (`token_groups: [pokemon]`), and 2.88 ms routing every option through attention too (139
  tokens). Of the default arm's 1.90 ms, `adapter.encode_groups` itself is 1.36 ms and
  `nn.TransformerEncoder` is 0.24 ms — the adapter's per-entity encoders dominate, not attention.
  Routing the full option table through attention is roughly +50% on a ~2 ms forward, not the
  quadratic blowup this section previously argued from complexity alone rather than a
  measurement.
- **`DeepSetsBackbone`** — permutation-**invariant** pooling (shared per-token MLP → sum/mean pool)
  over entity/hand/option tokens. The lightweight permutation-equivariant option named alongside
  Set Transformers in the literature; far cheaper than attention, order-invariant over cards.
- **`SetTransformerBackbone`** — permutation-equivariant self-attention (`nn.TransformerEncoder`)
  over the token set with a **padding mask**. Pool the global/state token → `state_repr`; keep
  per-option tokens → `option_repr`.
- **`TemporalTransformerBackbone`** — attention over a window of *past* states
  (Decision-Transformer-style; the literal "time series" reading). Preferred history-aware
  architecture; highest integration cost (sequence batching), so it comes after the per-state
  encoders.
- **`RecurrentBackbone` (LSTM/GRU)** — *backburner / lowest priority.* Established history-aware
  choice from the drafting literature and a fit for partial observability, but deprioritized below
  the transformer. Kept in the menu (same interface; needs `InitTracker` + recurrent-aware
  collection) as a fallback if the temporal transformer underperforms or is too costly — not a
  planned build step.

### 3. Heads (`src/models/heads.py`)

- **`PointerHead`** — **implemented, and the default** (`model/head=pointer`). One shared MLP scores
  slot `i` from `[state_repr, option_repr_i]`, with a separate state-only branch for the synthetic
  stop action (whose slot is padding). Permutation-equivariant, and the arm with the measured win:
  0.922 against the flat head's 0.825 on a matched task/budget/seed — see
  `docs/architecture/pointer-head.md`, which also covers the `zone_pooling: mean_max_sum` half of
  that fix.
- **`PointerPolicyHead`** — **implemented** (`model/head=pointer_dot`). Scaled dot product between each
  `option_repr` token and a query derived from `state_repr` → one logit per option. The option table
  already carries one row per action slot *including* the synthetic stop at `max_options`, so
  scoring every row yields exactly `(..., (max_options + 1))` logits and no separate stop logit is
  needed — true only in the trivial sense that a logit gets produced for that slot. Until
  `StructuredObsAdapter` grew a per-slot segment embedding, the stop row's *representation* was
  byte-identical to a padded slot's. The stop slot is not an engine option at all: the encoder
  writes rows only for the options the engine offered, so slot `max_options` keeps the same fill
  values (`card_id`/`cats` zero, `scalars` `-1.0`) as every unused slot, and
  `entity_projections["options"]` therefore produced one shared vector for stop and padding alike
  — the pointer head measurably gave them the identical logit (`0.056268`). The head could not
  have learned any identity-based preference for "the stop action" specifically, only whatever the
  state-derived query happened to produce against that shared constant.

  Two distinct fixes, easily conflated. The stop slot is now marked valid *by position* (it is
  structurally always present) and carries its own learned segment embedding, distinct from every
  real option and from padding. Separately, a slot's realness is now `cats[..., 0] != 0` — the
  option type the encoder always writes — rather than `card_id != 0`, which had been silently
  classifying every genuine *card-less* option (YES/NO/NUMBER/RETREAT/END) as padding: with only
  those legal, the pooled `options` vector was exactly all-zero and none of the 129 option tokens
  was valid. That second bug reached the `MLPBackbone` baseline too, not just the transformer.
  Naturally
  handles the variable-length option set; `MaskedCategorical` + `action_mask` zeroes illegal
  indices regardless — that masking was never wrong, only the pre-mask representation was
  impoverished. Cost is linear in the option count — one dot product each — unlike routing the
  129 option tokens through the trunk's self-attention (measured +50% on a ~2 ms forward, not
  prohibitive; see §2).

  This matters more than it looks. `LinearPolicyHead` reads only the pooled `state_repr`, by which
  point option identity has been averaged away twice (the adapter's masked mean over option rows,
  then the trunk's mean over tokens), so it can only learn *positional* preferences — "pick slot
  3" — never "pick the option that KOs". That is a ceiling on the MLP baseline too, and a candidate
  explanation for both arms flattening out near 0.85 against the random opponent.

  Either pointer head needs a backbone emitting `option_repr`; `build_actor_critic` raises at
  construction otherwise. It also sizes the head's `option_dim` from the backbone's
  `option_repr_dim`, so both heads pair with either token source — the adapter's unprojected
  per-entity encodings (`emit_option_tokens`, chosen automatically when the trunk builds none) or
  the trunk's own projection to `embed_dim` (`model.backbone.option_tokens=true`).
- **`ValueHead`** — MLP on `state_repr` → scalar `state_value`.
- *Alternative (not Phase 1):* the Hearthstone ByteRL work factors the action **auto-regressively**
  as `(type, target)` with a per-step mask instead of one flat softmax. The `Backbone`/`ActorCritic`
  interface is head-agnostic, so a factored head can replace the pointer head later without touching
  the trunk — worth it if the flat (max_options + 1)-way head plateaus.

### 4. Card embedding + deck conditioning (`card_embedding.py`, `deck_encoder.py`)

Shared `nn.Embedding(num_cards + 1, d)` with a reserved OOV index (the env doc warns "encoders must
tolerate unknown IDs"). Feeds entity tokens, hand tokens, option tokens **and** the deck encoder.
The embedding is designed so it can later be upgraded to *generalised* card representations
(numeric + text + type features) for unseen cards — the OOV index is the minimal version of that.

**Deck conditioning** — behavior is highly deck-contingent. The piloted deck is a fixed 60-card
list known at construction (`TCGEnv._deck0`/`_deck1`), *not* derivable from the per-step
observation (cards left in deck are hidden). `DeckContextEncoder` takes that card-ID list, embeds
each card and pools the multiset → a **deck-context vector** (Deep-Sets pool, or a small
set-attention pool). It is computed once per episode (the deck is static) and injected into every
backbone: concatenated to the global/state token, prepended as deck tokens the transformer attends
over, or via FiLM modulation. This is what lets one policy pilot different archetypes and lets the
self-play population span the scraped decklists in `archive/deck_scraping/` (D001–D072) rather than
overfitting a single list. For a single-deck first run it degenerates to a constant, so it is safe
to include from day one.

### 5. TorchRL assembly (`src/policies/ppo_actor.py`)

Use `torchrl.modules.ActorValueOperator` for the **shared-trunk** pattern (verify the constructor
against the installed torchrl 0.13.2):

```python
common = TensorDictModule(backbone, in_keys=[<obs token keys>], out_keys=["hidden", "option_repr"])
policy = ProbabilisticActor(
    TensorDictModule(PointerPolicyHead, in_keys=["hidden", "option_repr"], out_keys=["logits"]),
    in_keys={"logits": "logits", "mask": "action_mask"},
    distribution_class=MaskedCategorical,
    out_keys=["action"], return_log_prob=True, spec=action_spec,
)
value = ValueOperator(ValueHead, in_keys=["hidden"], out_keys=["state_value"])
ac = ActorValueOperator(common, policy, value)
# ac.get_policy_operator() / ac.get_value_operator() feed ClipPPOLoss and share the trunk.
```

`build_ppo_actor_critic(cfg, obs_spec, action_spec)` returns this, drop-in compatible with the
existing `RandomMaskedPolicy` contract (`in_keys` include `action_mask`; `out_keys=["action"]`,
int64). The factory builds the `backbone` and `head` modules with `hydra.utils.instantiate` from
`cfg.model.backbone` / `cfg.model.head` (each carries a `_target_`), so **which backbone and head
are used is a pure config choice** — no code change to swap them (see
[Configuration](#configuration-hydra)).

> The shared-trunk `ActorValueOperator` pattern is not yet in
> [`docs/torchrl/02-modules.md`](../torchrl/02-modules.md) — add a KB entry once verified against
> the installed version.

### 6. PPO trainer (`src/training/ppo_trainer.py`)

`PPOTrainer` overrides `Trainer._update(data)`: `GAE(gamma, lmbda)` **once per collected batch** →
shuffled-permutation minibatch loop over `ClipPPOLoss` (`loss_objective + loss_critic +
loss_entropy`, summed generically by the `loss_` key prefix so `DiscoPPOLoss` drops in unchanged) →
grad-clip → `Adam.step()`. Losses / grad-norm are logged in the outer loop.

It absorbs the feature set of a colleague's `TorchRLTrainer` **while keeping the friendly
hyperparameter constructor** (no separate dependency-injection trainer class, no extra inheritance
layer). Adopted features: AMP (`torch.amp`), `torch.compile` (loss + policy),
`target_kl` early stopping, NaN/Inf-guarded minibatches, and LR / entropy annealing.

> **TODO — RPO removed, revisit separately.** An earlier version of this trainer carried an RPO
> (Robust Policy Optimization) perturbation path (`MaskedRPOCategorical` / `RPOTanhNormal`,
> `rpo_alpha`). It has been removed: RPO is originally a continuous-control technique, the discrete
> analogue used here was novel and empirically unvalidated, and it added process-global mutable
> state (`rpo_enabled` toggled around the loss pass) for a benefit nobody had measured. If RPO (or a
> validated discrete equivalent) turns out to be worth having, reintroduce it as its own scoped
> task with a benchmark showing it helps, rather than carrying unvalidated inert code.

**Documented behavioural changes & inferences** (each also flagged inline in code):

- **GAE once per batch, not per epoch.** The previous implementation recomputed GAE every epoch;
  the adopted loop computes it once. A real learning-dynamics change (the more common PPO form).
- **Permutation minibatching.** `randperm` + contiguous slicing uses the final smaller minibatch;
  the previous `ReplayBuffer` + floor-division path silently dropped up to `sub_batch_size - 1`
  frames per epoch.
- **Anneal schedule inferred.** The `lr_anneal` / `ent_anneal` / `ent_warm_frac` flags come from the
  colleague's file, but the schedule lived in an unseen base class; implemented as the conventional
  linear anneal (LR → 0; entropy held for `ent_warm_frac` of training, then → 0).
- **`reward_scaling`** is accepted for parity but has no effect on the current sign-based win/draw
  stats; reserved for future magnitude logging.
- **AMP uses `torch.amp`** (not the deprecated `torch.cuda.amp`), required by the test suite's
  `filterwarnings=error`. `compile_*` defaults **off** (slow/fragile on the CPU dev box).

#### NCL (`ncl_model`) — removed

The colleague's trainer exposed an `ncl_model` parameter for **Natural Continual Learning**: a
Fisher-Information-Matrix estimate that anchors the weights important to previously-learned tasks
and projects/clips gradients to resist **catastrophic forgetting**. It was carried here for a time
as a guarded stub that accepted the parameter and raised `NotImplementedError`. That stub has since
been **removed** — an always-raising parameter bought interface parity at the cost of implying a
capability that never existed.

The rationale is kept because the underlying question is still open, and is now *more* live than
when it was written. Self-play **is** wired into the training entrypoint (`train=ppo_selfplay`), so
the non-stationarity that motivates NCL can now actually manifest; the
[self-play exploitability review](self-play-exploitability-review.md) names catastrophic forgetting
/ cycling (best-responding only to the latest opponent) as the central risk, and FIM
weight-anchoring is a *candidate* mitigation.

It remains deferred because (1) the canonical fix in this project's chosen literature (ByteRL/OSFP)
addresses forgetting at the **opponent-sampling** level — a win-rate-gated diverse mixture — not via
optimizer-side weight regularization, and the current league already keeps a permanent fixed
reference member in that spirit; (2) EWC/NCL-style FIM regularization in deep RL is finicky
(noisy/expensive FIM, can suppress plasticity), and there is no in-repo spec for the intended
contract. Revisit only once forgetting is empirically observed on the `eval/` curve — trying
OSFP-style opponent gating first.

### 7. Self-play (reuse existing scaffold)

Reuse `OpponentPool` + its `on_reset()` snapshot mechanism. Periodically freeze the learner's actor
to a checkpoint dir; snapshot opponents **rescan the checkpoint dir in `on_reset()`** — this is the
documented `ParallelEnv` per-worker-isolation caveat (no shared Python objects across worker
processes). The opponent wraps the same network as a greedy `Observation -> list[int]` callable
running the same tokenizer + masked-argmax + multi-select decode loop (this mirrors the Kaggle
`main.py` inference path). Track **exploitability**, not just win rate — the literature shows
self-play agents are brittle off-distribution.

> For a detailed gap analysis of the *current* self-play implementation against this design (and
> against the mechanisms behind ByteRL's OSFP), see
> [`docs/architecture/self-play-exploitability-review.md`](self-play-exploitability-review.md).

### 8. Tokenized observation encoder (`src/env/observation_encoder.py`, integration)

**Done** — this is `src/env/structured_observation_encoder.py::StructuredObservationEncoder` (the
env default), not a separate `TokenizedObservationEncoder` to still be built. It emits a
structured, **information-set-correct** observation (`observation_spec` keys below, in the
encoder's own naming):

- `entities (E_max, F)` + `entity_mask (E_max,)` — board pokemon for both seats (active + bench):
  card-id / hp / energies / status / owner features (opponent hand hidden, only counts).
- `hand (H_max, F)` + `hand_mask` — the agent's **own visible hand** (the engine exposes `hand`,
  hides the opponent's) — critical for decisions.
- `zones (Fz,)` — scalar summaries per seat: `deckCount`, `discard`, `prize`, `handCount`.
- `options (96, G)` + existing `action_mask` — per-option features for the pointer head.
- `global (Fg,)` — turn / flags / seat scalars.
- `deck (D_max,)` — card-id list of the piloted deck (feeds `DeckContextEncoder`; static per episode).

> **Train/serve skew:** this exact encoder must be shared *verbatim* with the Kaggle `main.py`
> `agent()` inference path.

## Configuration (Hydra)

Backbone and head are **independent, swappable config groups** instantiated via `_target_`, so a
run is fully specified by picking one of each — no code edits to change architecture.

**Top level** — `conf/config.yaml` (extends the existing `defaults:` pattern):

```yaml
defaults:
  - agent: ppo
  - model: default
  - train: ppo_selfplay
  - _self_

seed: 42
```

**Model composition** — `conf/model/default.yaml` selects a backbone + head and holds the shared
dims (card/deck embedding, value head) that both depend on:

```yaml
defaults:
  - backbone: mlp        # <- default backbone; override on the CLI
  - head: linear         # <- default head;     override on the CLI
  - _self_

embed_dim: 128           # shared token/hidden width; backbones read it via ${..embed_dim}
card_embedding:
  num_cards: 20000       # size the table above the current pool; index 0 reserved for OOV
  dim: ${..embed_dim}
deck_encoder:
  pool: mean             # mean | sum | attention
  dim: ${..embed_dim}
value_head:
  num_cells: [256, 256]
```

**Backbones** — `conf/model/backbone/*.yaml`, one `_target_` each (constructor kwargs only):

```yaml
# mlp.yaml — the proven baseline
_target_: src.models.mlp.MLPBackbone
num_cells: [256, 256]
activation: tanh

# transformer.yaml — attention across the observation's feature groups
_target_: src.models.transformer.TransformerBackbone
num_heads: 4
num_layers: 1
ff_dim: 256
dropout: 0.0
activation: gelu

# deepsets.yaml
_target_: src.models.backbone.DeepSetsBackbone
d_model: ${model.embed_dim}
phi_cells: [128, 128]
pool: mean               # mean | sum | max

# set_transformer.yaml
_target_: src.models.backbone.SetTransformerBackbone
d_model: ${model.embed_dim}
n_heads: 4
n_layers: 2
ff_dim: 256
dropout: 0.0

# temporal_transformer.yaml — Phase 3 (needs recurrent-aware collection)
_target_: src.models.backbone.TemporalTransformerBackbone
d_model: ${model.embed_dim}
n_heads: 4
n_layers: 2
context_len: 16

# recurrent.yaml — backburner fallback
_target_: src.models.backbone.RecurrentBackbone
hidden_size: ${model.embed_dim}
rnn: lstm                # lstm | gru
num_layers: 1
```

**Heads** — `conf/model/head/*.yaml`:

```yaml
# linear.yaml — flat (max_options + 1)-way logits; the MLP baseline's head
_target_: src.models.heads.LinearPolicyHead

# pointer.yaml — shared MLP over [state, option_i], separate stop branch (the default)
_target_: src.models.heads.PointerHead
num_cells: [128, 128]
activation: tanh

# pointer_dot.yaml — score per-option tokens against a state query (needs option_repr)
_target_: src.models.heads.PointerPolicyHead

# autoregressive.yaml — factored (type, target) head (later; see Heads §3)
_target_: src.models.heads.AutoRegressivePolicyHead
```

**Agent (PPO)** — `conf/agent/ppo.yaml`:

```yaml
name: ppo
clip_epsilon: 0.2
entropy_coeff: 0.01
gamma: 0.99
lmbda: 0.95
lr: 3.0e-4
num_epochs: 4
sub_batch_size: 256
frames_per_batch: 4096
max_grad_norm: 1.0
```

**Selecting an architecture** is then just group overrides:

```bash
# Phase-1 baseline (defaults)
python -m src.train

# Attention over the observation's feature groups (Issue #45), linear head
python -m src.train agent=ppo model/backbone=transformer

# Phase-2 primary: Set Transformer trunk + pointer head
python -m src.train model/backbone=set_transformer model/head=pointer

# Deep Sets trunk, pointer head, sweep two learning rates
python -m src.train -m model/backbone=deepsets model/head=pointer agent.lr=3e-4,1e-4
```

Valid pairings note: `PointerPolicyHead` needs a backbone that emits per-option tokens
(`option_repr`) — Deep Sets / Set Transformer / temporal. `LinearPolicyHead` works with any
backbone (it reads only `state_repr`), so it is the natural head for `MLPBackbone`. The factory
should assert this compatibility at build time with a clear error.

## Staging

Backbones are ordered by literature maturity + integration cost.

- **Phase 1 — proven baseline, loop green. Done.** `ActorCritic` + `MLPBackbone` (flattening
  every field, including the structured encoder's nested tables) on the *structured* obs (the env
  default) or the legacy flat obs; `PPOTrainer._update`; self-play vs `RandomOpponent` then
  `OpponentPool`. Proves PPO+masking+self-play end-to-end. This MLP-on-structured pairing is the
  reference every later backbone must beat — it is not blocked on Phase 2 (the structured,
  card-aware observation already exists as `StructuredObservationEncoder`; only the tokenizing
  encoder was ever the Phase-1/2 boundary, not the backbone).
- **Phase 2 — permutation-equivariant encoders + deck conditioning.** `DeckContextEncoder` +
  `PointerPolicyHead`, with `DeepSetsBackbone` first (cheap, order-invariant embedding + pooling
  over card/attack IDs, in place of the MLP's naive raw-ID flattening) then `SetTransformerBackbone`.
  The Phase-1 MLP-on-structured-obs run is the honest control this must beat. This is the intended
  main agent.
- **Phase 3 — sequence/history follow-up.** `TemporalTransformerBackbone` (DT-style history window),
  needing recurrent-aware collection (`InitTracker`, sequence batching). `RecurrentBackbone` (LSTM)
  is backburner — a fallback only if the temporal transformer underperforms or proves too costly.
