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
| `action_mask` | `Binary(n=97, bool)` | Legal option indices `0..95` plus index `96` = synthetic **stop**. |
| `action` (out) | `Categorical(97, int64)` | Action `i < 96` picks option `i`; `96` = stop. |
| `reward` | `Unbounded((1,), float32)` | Terminal-only: +1 win / −1 loss / draw. |
| `done`/`terminated`/`truncated` | `Binary(1, bool)` | `truncated` on the engine-selection safety cap. |

The opponent plays inside `_step`, so it is a standard **single-agent** env. Multi-select
(`minCount..maxCount`) selections are decomposed by the env into sequential single picks; the
policy only ever sees one masked `Categorical` per step.

## Module structure

```
src/
  models/
    card_embedding.py   # nn.Embedding over card IDs, OOV-tolerant (engine IDs can grow mid-competition)
    deck_encoder.py     # DeckContextEncoder: pool the piloted 60-card deck list -> deck-conditioning vector/tokens
    backbone.py         # Backbone ABC + MLP / DeepSets / SetTransformer / TemporalTransformer (+ Recurrent(LSTM), backburner)
    heads.py            # PointerPolicyHead (logits over options+stop), ValueHead (scalar)
    actor_critic.py     # base class assembling trunk + 2 heads; builder for ActorValueOperator
    transformer.py      # set-transformer building blocks (or fold into backbone.py)
  policies/
    ppo_actor.py        # build_ppo_actor_critic(cfg, obs_spec, action_spec) factory
  env/
    observation_encoder.py  # INTEGRATION: add TokenizedObservationEncoder (extends FlatObservationEncoder)
  training/
    ppo_trainer.py      # INTEGRATION: PPOTrainer overriding Trainer._update with GAE+ClipPPOLoss+optim loop
conf/
  config.yaml           # top-level defaults list (agent + model + train)
  agent/{dummy,ppo}.yaml            # PPO hyperparams (clip_epsilon, lr, gamma, lmbda, epochs, ...)
  model/
    default.yaml                    # composes one backbone + one head + shared dims (card/deck embedding)
    backbone/{mlp,deepsets,set_transformer,temporal_transformer,recurrent}.yaml
    head/{linear,pointer,autoregressive}.yaml
  train/ppo_selfplay.yaml           # snapshot interval, pool size/weights, deck list / deck sampling
```

The **backbone** and **head** are independent Hydra config groups, so any backbone can be paired
with any head from the CLI or a sweep (see [Configuration](#configuration-hydra)).

## Architecture

### 1. Base actor-critic class (`src/models/actor_critic.py`)

`class ActorCritic(nn.Module)` = one **shared backbone** + a **policy head** + a **value head**.
Forward once through the backbone, fan out to both heads (weight sharing → sample efficiency,
matches standard PPO / ByteRL). Outputs `logits` (shape `(..., 97)`) and `state_value` (`(..., 1)`).

### 2. Pluggable backbone (`src/models/backbone.py`)

One `Backbone` ABC → `forward(obs_td, deck_ctx) -> (state_repr, option_repr)`. Heads and
TorchRL assembly are identical across all implementations, so backbones are swappable via
`conf/model/`.

- **`MLPBackbone`** — the literature's **dominant, proven** network: concat/flatten features →
  `torchrl.modules.MLP`. Runs on the flat 36-dim obs *and* on the tokenized obs (flattened), so it
  is both the zero-env-change starter **and** the baseline every richer backbone must beat
  (Vieira et al.). Not a throwaway.
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
  learned **stop** logit → `(..., 97)`. Naturally handles the variable-length option set;
  `MaskedCategorical` + `action_mask` zeroes illegal indices. (The MLP baseline uses a plain
  `Linear(97)` head instead.)
- **`ValueHead`** — MLP on `state_repr` → scalar `state_value`.
- *Alternative (not Phase 1):* the Hearthstone ByteRL work factors the action **auto-regressively**
  as `(type, target)` with a per-step mask instead of one flat softmax. The `Backbone`/`ActorCritic`
  interface is head-agnostic, so a factored head can replace the pointer head later without touching
  the trunk — worth it if the flat 97-way head plateaus.

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

### 6. PPO trainer (`src/training/ppo_trainer.py`, integration)

Override `Trainer._update(data)` following [`docs/torchrl/05-ppo-recipe.md`](../torchrl/05-ppo-recipe.md):
`GAE(gamma, lmbda)` → refill `ReplayBuffer(LazyTensorStorage, SamplerWithoutReplacement)` →
minibatch loop over `ClipPPOLoss` (`loss_objective + loss_critic + loss_entropy`) → grad-clip →
`Adam.step()`. Recompute GAE each epoch (value estimates change). Log mean reward / losses /
episode length to W&B in the outer loop.

### 7. Self-play (reuse existing scaffold)

Reuse `OpponentPool` + its `on_reset()` snapshot mechanism. Periodically freeze the learner's actor
to a checkpoint dir; snapshot opponents **rescan the checkpoint dir in `on_reset()`** — this is the
documented `ParallelEnv` per-worker-isolation caveat (no shared Python objects across worker
processes). The opponent wraps the same network as a greedy `Observation -> list[int]` callable
running the same tokenizer + masked-argmax + multi-select decode loop (this mirrors the Kaggle
`main.py` inference path). Track **exploitability**, not just win rate — the literature shows
self-play agents are brittle off-distribution.

### 8. Tokenized observation encoder (`src/env/observation_encoder.py`, integration)

Extend the placeholder `FlatObservationEncoder` into a `TokenizedObservationEncoder` emitting a
structured, **information-set-correct** observation (new `observation_spec` keys):

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
# linear.yaml — flat 97-way logits; the MLP baseline's head
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

- **Phase 1 — proven baseline, loop green.** `ActorCritic` + `MLPBackbone` on the *current* flat
  obs; `PPOTrainer._update`; self-play vs `RandomOpponent` then `OpponentPool`. Proves
  PPO+masking+self-play end-to-end with no env changes. This MLP is the reference every later
  backbone must beat.
- **Phase 2 — deck-aware structured obs + permutation-equivariant encoders.**
  `TokenizedObservationEncoder` + `DeckContextEncoder` + `PointerPolicyHead`, with
  `DeepSetsBackbone` first (cheap, order-invariant) then `SetTransformerBackbone`. Re-run the MLP on
  the tokenized obs as the honest control. This is the intended main agent.
- **Phase 3 — sequence/history follow-up.** `TemporalTransformerBackbone` (DT-style history window),
  needing recurrent-aware collection (`InitTracker`, sequence batching). `RecurrentBackbone` (LSTM)
  is backburner — a fallback only if the temporal transformer underperforms or proves too costly.

## Prerequisites & integration notes

- **Merge/rebase** the model branch onto `torchrlenv` (or vice-versa) so
  `TCGEnv`/`Trainer`/`OpponentPool`/tests are present. This is where `observation_encoder.py` and
  `ppo_trainer.py` edits land.
- No repo-root `CLAUDE.md` currently exists, though [`docs/torchrl/README.md`](../torchrl/README.md)
  references one for the "consult the TorchRL KB first" routine — worth creating.
- `ActorValueOperator` shared-trunk pattern missing from the TorchRL KB — add once verified.

## Verification

- **Unit** — forward a dummy tokenized TensorDict through `ActorCritic`: assert `logits` shape
  `(..., 97)` and `state_value` `(..., 1)`; assert `MaskedCategorical` gives ~0 probability to
  illegal indices; assert gradients reach both heads *and* the shared trunk.
- **Contract** — `check_env_specs(make_env())` (existing), then swap `RandomMaskedPolicy` → the PPO
  actor in a short `env.rollout`; reuse the existing `test_actions_respect_mask` assertion
  (`masks.gather(-1, actions).all()`).
- **Integration smoke** — short PPO run (a few hundred frames) on the MLP baseline via `SerialEnv`:
  assert no NaNs, that loss/return move, and that W&B logs appear.
- **Self-play smoke** — run one snapshot cycle; assert a snapshot opponent loads the checkpoint from
  disk and plays.
