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
`conf/model/`.

- **`MLPBackbone`** — the literature's **dominant, proven** network: flattens every observation
  field (including the structured encoder's nested per-option/per-Pokemon/zone tables — card and
  attack IDs go in as raw floats, no embedding lookup) and concatenates them →
  `torchrl.modules.MLP`. **Implemented and trains against the default `structured` encoder today**
  (as well as the legacy flat 36-dim obs, kept for regression testing) — this naive-flatten
  pairing is the baseline every richer, permutation-invariant backbone below must beat (Vieira et
  al.), not a placeholder blocked on Phase 2.
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

- **`PointerPolicyHead`** — pointer/attention-style logits: score each `option_repr` token against
  a query derived from `state_repr` (dot-product or per-option MLP) → one logit per option, plus a
  learned **stop** logit → `(..., (max_options + 1))`. Naturally handles the variable-length option set;
  `MaskedCategorical` + `action_mask` zeroes illegal indices. (The MLP baseline uses a plain
  `Linear((max_options + 1))` head instead.)
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

#### NCL (`ncl_model`) — guarded stub, deferred

The colleague's trainer exposed an `ncl_model` parameter for **Natural Continual Learning**: a
Fisher-Information-Matrix estimate that anchors the weights important to previously-learned tasks
and projects/clips gradients to resist **catastrophic forgetting**. It is carried here as a
**guarded stub** — the parameter is accepted for interface parity, but passing a non-None module
raises `NotImplementedError`.

*Why it is not merely academic here:* this project is building toward **self-play against a shifting
`OpponentPool`**, a non-stationary / quasi-continual problem, and the
[self-play exploitability review](self-play-exploitability-review.md) names catastrophic forgetting
/ cycling (best-responding only to the latest opponent) as the central risk. FIM weight-anchoring is
a *candidate* mitigation for that forgetting.

*Why it is nonetheless deferred:* (1) the canonical fix in this project's chosen literature
(ByteRL/OSFP) addresses forgetting at the **opponent-sampling** level — a win-rate-gated diverse
mixture — not via optimizer-side weight regularization; (2) self-play is **not yet wired into the
training entrypoint** (review finding #1), so the forgetting problem cannot manifest today; (3)
EWC/NCL-style FIM regularization in deep RL is finicky (noisy/expensive FIM, can suppress
plasticity), and there is no in-repo spec for the intended contract. Revisit only once self-play is
live and forgetting is empirically observed — trying OSFP-style opponent gating first.

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
_target_: src.models.backbone.MLPBackbone
num_cells: [256, 256]
activation: tanh

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

# pointer.yaml — score per-option tokens against a state query (needs option_repr)
_target_: src.models.heads.PointerPolicyHead
query_dim: ${model.embed_dim}
score: dot               # dot | mlp

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

