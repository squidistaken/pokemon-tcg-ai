# TorchRL Environment Integration

How the cabt Pokémon TCG engine (Kaggle [pokemon-tcg-ai-battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)) is exposed as a TorchRL environment with parallel collection. The engine is not Gym-shaped: it presents a battle as a sequence of *selections*, each with a variable-length option list, interleaving both players. So the integration is a custom `EnvBase` subclass rather than a `GymWrapper`.

## Component map

```
cg/libcg.so (C engine, ctypes)
    → BattleHandle            per-instance battle pointer      src/env/battle_handle.py
    → TCGEnv(EnvBase)         selection → RL step mapping      src/env/tcg_env.py
        StructuredObservationEncoder  state → structured TensorDict  src/env/structured_observation_encoder.py
        RandomOpponent / OpponentPool  plays the non-agent seat    src/env/random_opponent.py, opponent_pool.py
    → TransformedEnv(ActionMask)  keeps the action spec mask in sync
    → ParallelEnv / SerialEnv     built from picklable factories    src/training/env_factory.py
    → Collector(policy)           batched collection
    → Trainer                     collector lifecycle + _update hook  src/training/trainer.py
```

Entry point: `src/train.py` (Hydra; config groups `conf/env/`, `conf/collector/`). Currently collection runs with `RandomMaskedPolicy` (`src/policies/random_masked_policy.py`); the PPO actor replaces it later.

## Engine access

`BattleHandle` wraps `lib.BattleStart` / `lib.Select` / `lib.GetBattleData` and holds its own `battle_ptr`. The organizers' `cg.game` module keeps a single global pointer, limiting a process to one battle; holding the pointer per instance supports many interleaved battles per process (verified empirically), which `SerialEnv` needs. `GetBattleData`'s `selectPlayer` field tells whose selection it is (verified equal to `obs.current.yourIndex`).

## Action space

Every step the engine offers `select.option`, a list whose length *and meaning* change per step. The spec is `Categorical(max_options + 1)` with `max_options = 96`:

- action `i < max_options` picks option `i` of the current selection,
- the last index is a synthetic **stop** action (see multi-select below).

The observation carries an `action_mask` bool tensor marking the legal indices. The `ActionMask` transform keeps the action *spec's* mask in sync (so `env.rand_step()` and `check_env_specs` are valid); actual collection policies read `action_mask` from the data directly, which is robust across `ParallelEnv` process boundaries. Option lists longer than `max_options` are truncated with a one-time warning (deck searches can reach ~60 options; 42 was the max observed under random play).

## Multi-select decomposition

Some selections require `minCount` to `maxCount` option indices in one engine call (~3% of selections, `maxCount ≤ 3`). Since a Categorical policy picks one index at a time, `TCGEnv` accumulates: each RL step picks one option (already-picked ones masked out), stop becomes legal once `minCount` is reached, and the accumulated list is submitted when the agent stops or hits `maxCount`. Accumulation sub-steps are ordinary zero-reward transitions; the agent never sees multi-select as a special case. At Kaggle inference time the same loop runs *inside* `agent()` without engine stepping, since the state does not change between picks.

## Two players, one agent

`TCGEnv._step` submits the agent's selection, then plays the opponent's selections internally (`_advance_to_agent`) until it is the agent's seat again or the battle ends. Control alternates irregularly (opponents select mid-turn); this loop absorbs all of it, so TorchRL sees a standard single-agent env. The agent's seat is randomized every reset, so it experiences going first and second.

## Opponent handling and self-play

The opponent is any callable `Observation -> list[int]`, injected into `TCGEnv` at construction (via `make_env_factories(cfg, opponent_factory=...)`). It operates on the raw engine observation, entirely outside TorchRL: its selections consume zero collector frames and produce no training data, only the agent's selections become transitions. It answers multi-select in one call (no decomposition needed on its side).

Current state: `train.py` uses the default `RandomOpponent`. `OpponentPool` is the self-play container: a callable opponent holding weighted members; the env calls its `on_reset()` hook at every episode start, at which point the pool draws the member that plays the whole episode, and `add()` registers new members.

Planned self-play flow (not built yet, needs the agent network): periodically freeze the learner into a checkpoint and have pool members face it. One caveat is already known: with `ParallelEnv`, each worker process constructs its **own** pool instance via the env factory, so calling `add()` on a pool in the main process reaches no worker. Snapshot distribution must therefore go through a channel workers can see, the intended design is a snapshot opponent that rescans a checkpoint directory in `on_reset()` (cheap, once per episode), with rebuilding the collector between opponent generations as the fallback. With `SerialEnv` (single process) `add()` works directly.

## Observation

The `StructuredObservationEncoder` (`src/env/structured_observation_encoder.py`)
translates the engine `Observation` dataclass into a nested `TensorDict` of
fixed-shape tensors (padded, masked, no modelling decisions). The encoder must
be shared verbatim with the Kaggle `main.py` inference path to avoid train/serve
skew.

```text
Engine Observation --> StructuredObservationEncoder --> TensorDict
```

### Conventions

- **ID `0` = none / padding / face-down / unknown.** Real card/attack IDs start at 1.
- **Categoricals store `enum + 1`**, reserving 0 for absent (embedding indices, not quantities).
- **`-1.0` marks absent floats** where `0` is a valid value (indices, counts).
- **Agent-relative**: owner = 1 (agent), 2 (opponent). Agent rows in `pokemon` precede opponent's.
- **Boolean masks**: `True` = occupied (including face-down cards with ID 0).
- **Floats are raw**: normalization is model-side.
- **Truncation**: zones exceeding their cap are silently truncated (one-time warning). Caps are constructor parameters (`bench_cap=8`, `hand_cap=30`, `discard_cap=60`, `prize_cap=6`, `energy_cap=16`, etc.).

### Schema

Each step tensordict carries `action_mask` alongside `observation`:

| Key | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `action_mask` | `(97,)` | bool | Legal actions: indices 0-95 pick option i, 96 = stop |
| `observation` | nested | -- | The tree below |

`97 = max_options + 1` with `max_options=96` (TCGEnv default).

```text
observation
├── globals              float32 (41,)           turn, counts, flags, per-player block
├── select_cats          int64   (2,)            SelectType+1, SelectContext+1
├── context_card_ids     int64   (2,)            [contextCard.id, effect.id]
├── stadium_id           int64   (1,)            stadium card ID (0 = none)
│
├── options              nested (97,)            one row per action
│   ├── card_id          int64   (97,)           resolved card ID
│   ├── target_id        int64   (97,)           target Pokemon card ID
│   ├── attack_id        int64   (97,)           attack ID
│   ├── owner            int64   (97,)           0/1/2 (n/a / agent / opponent)
│   ├── cats             int64   (97, 4)         [type, area, inPlayArea, specialCondition] +1
│   └── scalars          float32 (97, 6)         [number, count, index, toolIdx, energyIdx, inPlayIdx]
│
├── pokemon              nested (18 = 2 x (1+8))
│   ├── card_id          int64   (18,)           Pokemon card ID
│   ├── tool_id          int64   (18,)           tool card ID
│   ├── energy_card_ids  int64   (18, 16)        attached energy card IDs
│   ├── pre_evolution_ids int64  (18, 2)         evolved-from card IDs
│   ├── features         float32 (18, 20)        hp, hp%, status, energy counts
│   └── mask             bool    (18,)           occupied slots
│
├── my
│   ├── hand_ids         int64   (30,)           hand card IDs
│   ├── hand_mask        bool    (30,)
│   ├── discard_ids      int64   (60,)
│   ├── discard_mask     bool    (60,)
│   ├── prize_ids        int64   (6,)
│   └── prize_mask       bool    (6,)
│
├── opp (only discard + prize -- hand is hidden)
│   ├── discard_ids      int64   (60,)
│   ├── discard_mask     bool    (60,)
│   ├── prize_ids        int64   (6,)
│   └── prize_mask       bool    (6,)
│
└── select_deck / looking  (optional zones, mostly empty)
    ├── ids              int64   (60,) / (60,)
    └── mask             bool    (60,) / (60,)
```

`globals` layout (41 floats):

```text
[0]  turn                    [9]  minCount
[1]  turnActionCount         [10] maxCount
[2]  agent seat (0/1)        [11] n_options
[3]  firstPlayer flag        [12] chosen_count
     (1/0/-1)                [13] remainDamageCounter
[4]  supporterPlayed         [14] remainEnergyCost
[5]  stadiumPlayed           [15] has_search (select_deck populated)
[6]  energyAttached          [16] has_looking
[7]  retreated
[8]  has_select (0 = terminal)

[17-28] agent player block  (same layout as 29-40 for opponent)
    [17] deckCount  [18] handCount  [19] prize count  [20] bench count
    [21] benchMax   [22] has_active  [23] active_facedown
    [24-28] poisoned, burned, asleep, paralyzed, confused

[29-40] opponent player block
```

`pokemon` row layout (agent active = row 0, agent bench = rows 1-8,
opponent active = row 9, opponent bench = rows 10-17):

```text
features columns (20):
[0] hp              [5]  tool count
[1] maxHp           [6]  energy card count
[2] hp / maxHp      [7]  pre-evolution count
[3] appearThisTurn  [8-19] provided energy per EnergyType (12 cols)
[4] is_active
```

### Option reference resolution

The engine describes options as zone references (player, area, index), not
concrete card IDs. The encoder dereferences them so the model side never
has to:

```text
Engine option:                    Encoder output:
────────────────────────────────────────────────────
CARD / PLAY / ABILITY             card_id = card at zone
  (player=X, area=Y, index=Z)      if hidden zone (opp hand) -> 0

TOOL_CARD / ENERGY_CARD           card_id = attached card
  (player=X, area=Y, index=Z)      target_id = carrier Pokemon card ID
  + toolIndex / energyIndex

ATTACH / EVOLVE                   card_id = played card
  (target in-play Pokemon)         target_id = target Pokemon card ID

ENERGY (selection)                 card_id = attached energy card
  + index + energyIndex            target_id = carrier Pokemon card ID

ATTACK                             attack_id filled from option
  (player=X, area=Y, index=Z)

SKILL                              card_id direct (no zone deref)

YES/NO/NUMBER/RETREAT/END          All zeros (card_id=0, attack_id=0)
  (plain choices)
```

Rows beyond the offered option count -- including row 96 (stop) -- are all
zeros or `-1.0`.

### How to consume

This is a suggestion, one way to approach it.

The integer fields fall into three categories by vocabulary size:

| Category | Fields | Vocabulary |
| --- | --- | --- |
| Card IDs | `options.card_id`, `target_id`, `pokemon.card_id`/`tool_id`/`energy_card_ids`/`pre_evolution_ids`, `my.*_ids`, `opp.*_ids`, `context_card_ids`, `stadium_id`, `select_deck.ids`, `looking.ids` | ~1,300 (shared `nn.Embedding` is natural) |
| Attack IDs | `options.attack_id` | ~1,600 (separate embedding) |
| Small categoricals | `options.owner` (0/1/2), `options.cats` (4 columns, each ~5-50 values), `select_cats` (2 columns, each ~5-50 values) | Can embed, one-hot, or feed as small integers |

The 97 rows in `options` align one-to-one with `action_mask`: row `i`
carries every feature for action `i`. Row 96 (the stop action) is all
zeros / `-1.0` and is scored the same as any other row -- the mask
determines legality. A typical actor produces 97 logits via a shared
function `score(row_i, global_state) -> logit_i`.

The observation has nested structure with heterogeneous shapes:
`globals` is a flat vector, `pokemon` is a tabular set of 18 rows with
per-row features and masks, and `my`/`opp` zones are variable-length
sets of card IDs with masks. How to fuse them into `global_state`
(pooling, attention, flattening, recurrence over timesteps) is a
model-side decision. The game is partially observable (opponent's hand,
deck order, face-down prizes), so memory across timesteps is the
standard remedy.

### What is not in the observation

- **`Observation.logs`** (event list). Most is recoverable from state diffs.
  The raw `Observation` is reachable on `TCGEnv._pending` if needed.
- **Deck order / opponent hand beliefs.** The observation is a single
  information-set draw.

### Fixtures

`tests/fixtures/observations.pt` has one captured tensordict per named
case, for building model code without the engine:

| Case | Meaning |
| --- | --- |
| `setup` | Before turn 1 |
| `main_select` | Playing a card from hand |
| `card_select` | Choosing a card (e.g. for effect) |
| `multi_select_partial` | Mid multi-select accumulation |
| `deck_search` | Searching/ordering the deck |
| `yes_no` | Binary choice |
| `attack_option` | MAIN selection with attacks |
| `energy_select` | Energy attachment target |
| `terminal` | Game over |

`tests/fixtures/card_tables.pt` holds static `CardDatabase` tables.

```python
import torch
samples = torch.load("tests/fixtures/observations.pt", weights_only=False)
tables = torch.load("tests/fixtures/card_tables.pt", weights_only=False)
obs = samples["attack_option"]                  # TensorDict
my_hand = obs["observation", "my", "hand_ids"]  # card IDs in agent's hand
```

Regenerate with `uv run python scripts/generate_obs_fixtures.py`. The
compatibility gate is `tests/test_structured_observation.py`.

### Static card database

`CardDatabase` (`src/env/card_database.py`) provides card-ID-indexed
static tables:

| Table | Shape | Content |
| --- | --- | --- |
| `card_features` | `(1268, 12)` float32 | hp, retreat, stage flags, ex/tera/aceSpec, attack count |
| `card_cats` | `(1268, 4)` int64 | cardType, energyType, weakness, resistance (+1, 0 absent) |
| `card_attack_ids` | `(1268, 2)` int64 | attack IDs on each card |
| `attack_features` | `(1557, 14)` float32 | damage, cost total, cost per energy type |

Row index = card/attack ID (row 0 = null). The competition may add cards
over time, so the observed max ID is not final.

## Reward and termination

- Reward is terminal-only: +1 win, −1 loss, `reward_draw` (default 0) on a draw, read from `current.result`.
- A safety cap (`max_engine_selections`, default 5000) yields `truncated=True` if a game never ends.

## Vectorization and collection

`make_env_factories(cfg, opponent_factory=None)` builds one picklable factory per worker (module-level function + `functools.partial`, per-worker seeds). `Trainer` builds `ParallelEnv` (default: 8 fork workers) or `SerialEnv` from them, runs the `Collector`, tracks episode stats, and calls `_update(data)` after every batch, a no-op in the base class, to be overridden by the PPO trainer with the GAE/loss/optimizer loop.

A **frame**, in TorchRL terminology, is one environment transition: one tensordict of `(observation, action, reward, done, ...)` produced by a single step call. It is not a full engine selection turn, and it is not a game frame in the video sense. Here, one frame = one agent-side `Categorical` pick in `TCGEnv._step`, including multi-select sub-steps. The opponent's moves inside `_advance_to_agent` never reach the collector and consume zero frames, only the agent's own decisions count.

`frames_per_batch` (`conf/collector/default.yaml`, default 2048) is how many frames the `Collector` gathers before handing a batch to `Trainer._update`. `total_frames` (default 16384) is the total collection budget for the run, so training runs for `total_frames / frames_per_batch` batches. `fps`, reported by `Trainer.train()`, is therefore collection **throughput**, frames (agent decisions) per second, not a rendering rate. See [Throughput](#throughput) for how the vectorization compares to a naive single environment.

## Throughput

How much does the TorchRL vectorization actually buy over the naive baseline, a single `TCGEnv` with no batching, stepped one transition at a time (`construct env → loop rand_step`, the shortest thing you'd write by hand)? Measured on an AMD Ryzen 7 7800X3D (8 cores / 16 threads), random policy against `RandomOpponent`, example-deck mirror match, 65,536 frames at `frames_per_batch=2048`, `fps` as reported by `Trainer.train()`:

| Configuration | fps | Speedup |
| --- | --- | --- |
| naive single env (no vectorization) | 587 | 1.00× |
| `SerialEnv(8)`, one process | 821 | 1.40× |
| `ParallelEnv(8, fork)` | 1,948 | 3.32× |
| `ParallelEnv(16, fork)` | 2,394 | 4.08× |

These numbers reflect the [observation schema](#observation): card IDs for embedding lookup, per-option features aligned with the action mask, and padded zone tables. The encoder stages per-element writes in preallocated NumPy buffers and converts once per field (~124 µs/step).

Reading the numbers:

- **`SerialEnv(8)` gains only ~40 %.** Batching 8 envs in one process still steps them in a sequential Python loop under one GIL; the gain is amortized per-batch tensordict/collector overhead, not real parallelism.
- **`ParallelEnv(8, fork)` gives ~3.3×**, not 8×. `ParallelEnv` steps *synchronously*: every step the main process gathers observations from all workers, runs the policy centrally, and scatters actions back. With a policy this cheap (random) and a C-engine step at ~0.9 ms, the per-step IPC round-trip plus the straggler effect (variable engine work per step gates the batch on the slowest worker) dominate the compute being parallelized. Oversubscribing to 16 workers still helps (4.08×) by keeping the pipeline fuller than the physical core count.
- **This measures the plumbing ceiling, not training throughput.** With a free policy the run is IPC/straggler-bound. Multiprocessing overhead pays off more once the policy is a real neural net: batched inference in the main process amortizes the round-trip while workers do more per step. An async collector is the next lever if IPC stays the bottleneck.

Reproduce with `uv run python scripts/bench_throughput.py` (from the repo root). It builds the config directly and drives `Trainer`, bypassing the Hydra CLI, and prints the table above. Because the engine RNG is not seedable (see [Known properties](#known-properties-and-limitations)), exact fps varies a few percent per run and is hardware-dependent; the ratios are stable.

## Known properties and limitations

- **Engine RNG is not seedable** (`std::random_device` internally, no seed export): `set_seed: true` covers Python/torch only; episodes stay stochastic.
- Random agent vs `RandomOpponent` wins ~0.48–0.55, not exactly 0.50: the two random policies have different multi-select size distributions (sequential picks + stop vs uniform count). Keep in mind when reading baselines.
- The observation is information-set-correct out of the box (opponent hand/facedown cards are hidden by the engine).
- Enums and dataclass fields may gain members during the competition; encoders must tolerate unknown IDs.
- The Collector emits a `FutureWarning` about an `InitTracker` transform; only relevant for recurrent policies, safe to ignore.

## Not here yet

PPO actor-critic (`ProbabilisticActor` + `MaskedCategorical`, `in_keys={"logits": "logits", "mask": "action_mask"}`), the neural encoder over the structured observation (embeddings + option scoring; the [observation section](#observation)), snapshot opponents for the self-play pool, evaluation loop, and the export path into the Kaggle `main.py` `agent()` function. The engine's `search_begin`/`search_step` determinized-search API is a separate, later opportunity.

Tests: `tests/test_tcg_env.py` (spec check, full episodes, mask legality, concurrent handles), `tests/test_structured_observation.py` (observation contract: zone/state consistency, option-mask alignment, fixture validity, card tables), `tests/test_trainer.py` (collection, opponent pool).
