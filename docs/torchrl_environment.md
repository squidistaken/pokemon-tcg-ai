# TorchRL Environment Integration

How the cabt Pokémon TCG engine (Kaggle [pokemon-tcg-ai-battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)) is exposed as a TorchRL environment with parallel collection. The engine is not Gym-shaped: it presents a battle as a sequence of *selections*, each with a variable-length option list, interleaving both players. So the integration is a custom `EnvBase` subclass rather than a `GymWrapper`.

## Component map

```
cg/libcg.so (C engine, ctypes)
    → BattleHandle            per-instance battle pointer      src/env/battle_handle.py
    → TCGEnv(EnvBase)         selection → RL step mapping      src/env/tcg_env.py
        FlatObservationEncoder  state → flat tensor (placeholder)  src/env/observation_encoder.py
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

`TCGEnv._step` submits the agent's selection, then plays the opponent's selections internally (`_advance_to_agent`) until it is the agent's seat again or the battle ends. Control alternates irregularly (opponents select mid-turn); this loop absorbs all of it, so TorchRL sees a standard single-agent env. The agent's seat is randomized every reset.

The opponent is injected at construction. `OpponentPool` is a callable opponent whose member is drawn per episode via the env's `on_reset` hook, the self-play mechanism: frozen policy snapshots get `add()`-ed to the pool as training progresses (snapshot opponents require the agent network and are not built yet).

## Observation, reward, termination

- `FlatObservationEncoder` produces 36 floats (turn/flags, selection metadata, per-player board summary). It is an explicit **placeholder** so the pipeline is tensor-complete; the real encoder (card-ID embeddings + per-option features for a pointer-style actor) comes with the agent. It must then be shared verbatim with the Kaggle `main.py` inference path to avoid train/serve skew.
- Reward is terminal-only: +1 win, −1 loss, `reward_draw` (default 0) on a draw, read from `current.result`.
- A safety cap (`max_engine_selections`, default 5000) yields `truncated=True` if a game never ends.

## Vectorization and collection

`make_env_factories(cfg, opponent_factory=None)` builds one picklable factory per worker (module-level function + `functools.partial`, per-worker seeds). `Trainer` builds `ParallelEnv` (default: 8 fork workers) or `SerialEnv` from them, runs the `Collector`, tracks episode stats, and calls `_update(data)` after every batch, a no-op in the base class, to be overridden by the PPO trainer with the GAE/loss/optimizer loop. Throughput: ~2,200 fps with the random policy.

## Known properties and limitations

- **Engine RNG is not seedable** (`std::random_device` internally, no seed export): `deterministic: true` covers Python/torch only; episodes stay stochastic.
- Random agent vs `RandomOpponent` wins ~0.48–0.55, not exactly 0.50: the two random policies have different multi-select size distributions (sequential picks + stop vs uniform count). Keep in mind when reading baselines.
- The observation is information-set-correct out of the box (opponent hand/facedown cards are hidden by the engine).
- Enums and dataclass fields may gain members during the competition; encoders must tolerate unknown IDs.
- The Collector emits a `FutureWarning` about an `InitTracker` transform; only relevant for recurrent policies, safe to ignore.

## Not here yet

PPO actor-critic (`ProbabilisticActor` + `MaskedCategorical`, `in_keys={"logits": "logits", "mask": "action_mask"}`), card-aware encoder + option featurization, snapshot opponents for the self-play pool, evaluation loop, and the export path into the Kaggle `main.py` `agent()` function. The engine's `search_begin`/`search_step` determinized-search API is a separate, later opportunity.

Tests: `tests/test_tcg_env.py` (spec check, full episodes, mask legality, concurrent handles), `tests/test_trainer.py` (collection, opponent pool).
