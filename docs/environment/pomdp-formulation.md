## Full POMDP/MDP Formulation and TorchRL Structure

---

### 1. What kind of problem is this?

**Partially Observable Markov Decision Process (POMDP).** The agent cannot see the full game state:

- **Hidden information**: Opponent's hand, deck order, face-down prize cards, face-down active Pokémon, future draws.
- **What the agent sees**: Exactly what a human player would see — their own cards, their own board, opponent's discard pile and prize count, and the opponent's active Pokémon (if not face-down).
- **No belief aggregation** is done by the encoder. If you want memory across timesteps (e.g. an RNN or transformer with history), that is a model-side decision, explicitly flagged in the docs as the standard remedy for partial observability.

---

### 2. The core problem the environment solves

The Pokémon TCG engine (the C library behind `cg/`) is not RL-shaped. It works like this:

1. **Engine says**: "Here is a list of things you can pick. Choose between 1 and 3 of them."
2. **You call**: `lib.Select(battle_ptr, [indices], count)` with **all** chosen indices at once.
3. **Engine responds**: New state.

But an RL policy picks **one action at a time**. The environment bridges this gap.

---

### 3. Action space and the stop action

```python
self.action_spec = Categorical(n_actions, dtype=torch.int64)
# n_actions = max_options + 1 = 96 + 1 = 97
```

| Action index | Meaning |
|---|---|
| `0 .. 95` | Pick option index `i` from the current `select.option[]` list |
| `96` | **Stop** — I'm done picking; submit whatever I've accumulated |

The **action is a single integer** (`0`, `5`, `96`, etc.) — not a vector, not a boolean mask. Just one number per step.

The option list length varies per step (e.g. 5 attack targets vs 60 deck cards), but the action space is always 97 because we **pad** to `max_options=96`. Action indices beyond the actual option count are forbidden by the mask.

---

### 4. The action mask (separate from the action)

The `action_mask` is a **boolean input tensor** that tells the policy which integers are legal to pick. It is not the action itself.

```python
# In the environment:
self.observation_spec = Composite(
    observation=self._encoder.spec(),    # the nested state tree
    action_mask=Binary(n=n_actions, dtype=torch.bool),  # (97,) bool
)
self.action_spec = Categorical(n_actions, dtype=torch.int64)  # single int64
```

A step's tensordict contains:

```
data["action"]         = tensor([3])          # int64, shape (1,) — just the index
data["action_mask"]    = tensor([True, True, True, True, False, ...])  # bool, shape (97,)
```

The policy produces 97 raw logits. The `MaskedCategorical` distribution (used via `ProbabilisticActor`) zeroes out any logit where `action_mask == False`, then samples one integer index. That sampled integer is the action.

---

### 5. Normal single-pick step (the common case)

Most selections are `minCount=1, maxCount=1` — pick exactly one thing.

```
Engine: select.type=ATTACK, select.option=[<attack1>, <attack2>, <attack3>]
        minCount=1, maxCount=1

Environment builds:
  action_mask = [True, True, True, False, False, ..., False, False]
                ↑ opt 0   ↑ opt 1   ↑ opt 2   options 3..95   ↑ stop(96)

Agent picks: action = 1

Inside _step:
  chosen.append(1)          → chosen = [1]
  len(chosen) >= maxCount   → True (1 >= 1)
  Submit handle.select([1])
  Engine processes the attack, returns new observation
  Agent sees the result
```

**Single-pick → submit immediately.** Since `len(chosen) == maxCount`, the environment calls the engine right away.

---

### 6. Multi-select step (decomposed into sequential RL steps)

~3% of selections require multiple picks: e.g. "Choose 2 of your Bench Pokémon to discard" (`minCount=2, maxCount=2`). The environment breaks this into **multiple RL steps**, each producing one transition in the collected data.

```
Engine: select.context=DISCARD, minCount=2, maxCount=2
        option=[<bench1>, <bench2>, <bench3>, <bench4>, <bench5>]

── Step 1 ──
  chosen = []
  action_mask = [True, True, True, True, True, False, ..., False]
                ↑ bench 0-4                                 ↑ stop(96) = NOT legal
  Agent picks: action = 3  (bench4)

  chosen = [3]
  len(chosen)=1 < maxCount=2  →  NO submit
  reward = 0.0
  Return to agent for another step

── Step 2 ──
  action_mask = [True, True, True, False, True, False, ..., False]
                ↑ bench0   ↑ bench1  ↑ bench2  ↑ bench3=MASKED  ↑ bench4
                (index 3 already picked, so it's masked out)
                stop(96) = STILL not legal (need minCount=2, have only 1)
  Agent picks: action = 1  (bench2)

  chosen = [3, 1]
  len(chosen)=2 == maxCount  →  SUBMIT handle.select([3, 1])
  Engine processes, returns new observation
  reward = 0.0 (still intermediate, not terminal)
```

---

### 7. The stop action in detail

Some selections have `minCount < maxCount` — e.g. "Choose up to 3 cards" (`minCount=1, maxCount=3`). The **stop action** lets the agent decide when to finish.

```
Engine: select.context=TO_BENCH, minCount=1, maxCount=3
        option=[<bench1>, <bench2>, <bench3>, <bench4>]

── Step 1 ──
  action_mask = [True, True, True, True, False, ..., False, False]
                ↑ stop(96) = NOT legal yet (need ≥ minCount=1 first)
  Agent picks: action = 0
  chosen = [0]
  reward = 0.0

── Step 2 ──
  action_mask = [True, True, True, False, ..., False, True]
                ↑ bench0=MASKED  bench1  bench2  bench3=out   ↑ stop(96)=NOW LEGAL
  Agent can either:
    - pick another bench (actions 1, 2, or 3), or
    - pick stop (action 96)

  Agent picks: action = 96  (STOP)
  SUBMIT handle.select([0])
```

**Key rule**: The stop action becomes legal when `len(chosen) >= minCount`. It stays legal for all subsequent steps. At `maxCount` the environment auto-submits and stop is never needed.

---

### 8. What happens on a terminal observation

When the game is over or truncated, the environment produces a special mask to keep the spec valid:

```python
if select is None or self._game_over(pending) or self._truncate_flag:
    mask[self._stop_index] = True   # only stop is legal
    return mask
```

The policy samples `action=96`, and `_step` processes it normally — but the game is already over, so the same terminal reward is returned. This satisfies TorchRL's spec enforcement (every step must have at least one legal action).

---

### 9. Two players, one agent

The opponent is played **inside** `_step`, not by the RL loop:

```python
def _advance_to_agent(self, observation):
    while not game_over and not truncated:
        if select.maxCount == 0:
            observation = self._engine_select([])       # auto-submit empty
            continue
        if self._handle.select_player == self._agent_seat:
            break                                        # agent's turn → return
        observation = self._engine_select(self._opponent(observation))  # opponent plays
    return observation
```

- The opponent is any `Callable[[Observation], list[int]]` — e.g. `RandomOpponent` or `OpponentPool`.
- The opponent answers a **full** selection in one call (no RL decomposition for them).
- Opponent moves **never appear in the collected data**. They consume zero frames.
- The agent's seat is randomized every `reset()` (50/50 going first).

---

### 10. Reward structure

```
reward = 0.0           ← every intermediate step (including multi-select sub-steps)
reward = +1.0          ← agent won
reward = -1.0          ← agent lost
reward = reward_draw   ← draw (default 0.0)
```

Reward is **terminal-only**. The agent only gets a signal at the end of the game.

A safety cap (`max_engine_selections=5000`) can truncate infinitely long games:

```
terminated = game_over
truncated  = hit cap AND not naturally over
done       = terminated OR truncated
```

---

### 11. Full data flow through TorchRL collection

When using `ParallelEnv` + `Collector` (the standard training path), each collected batch looks like:

```
data                                    shape = (B, T)   B = num_workers, T = frames_per_batch // B
├── "observation"   (nested)            (B, T, ...)
├── "action_mask"   bool                (B, T, 97)
├── "action"        int64               (B, T, 1)         ← single integer per frame
├── "sample_logits" float32             (B, T, 97)        from ProbabilisticActor
├── "action_log_prob" float32           (B, T, 1)         from ProbabilisticActor
├── "done"          bool                (B, T, 1)
├── "terminated"    bool                (B, T, 1)
├── "truncated"     bool                (B, T, 1)
└── "next"
    ├── "observation"   (nested)        (B, T, ...)
    ├── "action_mask"   bool            (B, T, 97)
    ├── "reward"        float32         (B, T, 1)         ← 0 or terminal ±1
    ├── "done"          bool            (B, T, 1)
    ├── "terminated"    bool            (B, T, 1)
    └── "truncated"     bool            (B, T, 1)
```

The PPO trainer reads:
- `data["action"]` — what the agent chose
- `data["action_log_prob"]` — log-prob under the old policy
- `data["next", "reward"]` — for GAE advantage estimation
- `data["next", "done"]` — to know which transitions are terminal

---

### 12. Summary of key design decisions

| Aspect | Decision |
|---|---|
| Formulation | POMDP (hidden opponent hand, deck, face-down cards) |
| Agent type | Single-agent (opponent played inside `_step`) |
| Action | Single `int64` index — picks one option or triggers stop |
| Action mask | Boolean `(97,)` tensor — input to policy, tells which indices are legal |
| Multi-select | Decomposed into sequential zero-reward steps with mask tracking |
| Action-to-option alignment | Row `i` of `observation.options` table = description of action `i` |
| Observation | Nested `TensorDict` with padded/masked tables, raw card IDs |
| Reward | Terminal-only: +1 win / −1 loss / 0 draw |
| Seat randomization | Random at every `reset()` |
| Opponent | `Callable[[Observation], list[int]]` outside TorchRL, zero frames |
| Multi-select for opponent | Full selection in one call (no decomposition) |
| Encoding paradigm | No modelling in encoder — the model embeds and normalizes |

---

### 13. The observation — exact structure

Produced by `StructuredObservationEncoder.encode()` (`src/env/structured_observation_encoder.py`). It maps the engine's `Observation` dataclass to a nested `TensorDict` with batch size `()`. Every table is padded to a fixed capacity so the shape is identical every step, with a boolean mask marking the slots that hold real data. Everything is **agent-relative**: "my"/"agent" always means the learning agent, regardless of which seat it drew at `reset()`.

#### 13.1 The full tree

Shapes below use the default capacities (see [13.9](#139-capacities-and-overflow) for where each number comes from):

```
observation
├── globals              (41,)      float32   scalar game/selection/player context
├── select_cats          (2,)       int64     [select.type + 1, select.context + 1]; [0, 0] on terminal
├── context_card_ids     (2,)       int64     [contextCard.id, effect.id]; 0 when absent
├── stadium_id           (1,)       int64     stadium card ID, 0 if no stadium in play
├── options                                   97 rows; row i describes action i
│   ├── card_id          (97,)      int64     resolved card the option refers to
│   ├── target_id        (97,)      int64     in-play Pokémon the option acts on / carrier
│   ├── attack_id        (97,)      int64     attack ID for ATTACK options
│   ├── owner            (97,)      int64     0=none, 1=agent, 2=opponent
│   ├── cats             (97, 4)    int64     type / area / inPlayArea / specialConditionType
│   └── scalars          (97, 6)    float32   number / count / index / toolIndex / energyIndex / inPlayIndex
├── pokemon                                   18 rows: both boards
│   ├── card_id          (18,)      int64     0 = empty slot or face-down
│   ├── tool_id          (18,)      int64     first attached tool's card ID (only the first)
│   ├── energy_card_ids  (18, 16)   int64     attached energy card IDs, padded with 0
│   ├── pre_evolution_ids (18, 2)   int64     cards underneath (evolution history), padded with 0
│   ├── features         (18, 20)   float32   HP / flags / attachment counts / energy histogram
│   └── mask             (18,)      bool      True = slot occupied (also True for face-down active)
├── my                                        agent's zones
│   ├── hand_ids         (30,)      int64
│   ├── hand_mask        (30,)      bool
│   ├── discard_ids      (60,)      int64
│   ├── discard_mask     (60,)      bool
│   ├── prize_ids        (6,)       int64     0 = face-down prize (mask still True)
│   └── prize_mask       (6,)       bool
├── opp                                       opponent's public zones (no hand — hidden)
│   ├── discard_ids      (60,)      int64
│   ├── discard_mask     (60,)      bool
│   ├── prize_ids        (6,)       int64     0 = face-down prize (mask still True)
│   └── prize_mask       (6,)       bool
├── select_deck                               deck-search reveal (select.deck), empty when no search
│   ├── ids              (60,)      int64
│   └── mask             (60,)      bool
└── looking                                   "looking" reveal (state.looking), empty when not looking
    ├── ids              (60,)      int64
    └── mask             (60,)      bool
```

**How to read any field** — the same sentinel values are used everywhere:

| Value | Means | Applies to |
|---|---|---|
| `0` | none / padding / face-down / unknown | every int64 ID field (card, tool, attack, energy) |
| `enum + 1` | categorical as embedding index; `0` = absent | `select_cats`, `options.cats` |
| `-1.0` | absent scalar (because `0` is a valid value there) | `options.scalars` only |
| `0.0` | padding | every other float field |
| `1` / `2` / `0` | agent / opponent / no owner reference | `options.owner` |
| mask `True` | slot is occupied — possibly by a face-down card with ID `0` | every `*_mask` |

All floats are raw and unnormalized. Embedding, normalization and aggregation are model-side.

#### 13.2 `globals` — 41 float32, index by index

**Game block (0–7):**

| Index | Value | Notes |
|---|---|---|
| 0 | `state.turn` | Raw turn number |
| 1 | `state.turnActionCount` | Actions taken this turn |
| 2 | `agent_seat` | 0.0 or 1.0 — which engine seat the agent occupies |
| 3 | first-player flag | `-1.0` if not yet decided; else `1.0` if the agent goes first, `0.0` otherwise |
| 4 | `state.supporterPlayed` | Once-per-turn flag |
| 5 | `state.stadiumPlayed` | Once-per-turn flag |
| 6 | `state.energyAttached` | Once-per-turn flag |
| 7 | `state.retreated` | Once-per-turn flag |

**Selection block (8–16).** All nine entries are `0.0` on a terminal observation; index 8 doubles as the "selection present" flag:

| Index | Value | Notes |
|---|---|---|
| 8 | `1.0` | Selection present |
| 9 | `select.minCount` | Minimum picks required |
| 10 | `select.maxCount` | Maximum picks allowed |
| 11 | `len(select.option)` | Actual (unpadded) option count |
| 12 | `already_chosen_option_count` | Picks accumulated so far in an ongoing multi-select |
| 13 | `select.remainDamageCounter` | Remaining damage counters to distribute |
| 14 | `select.remainEnergyCost` | Remaining energy cost to pay |
| 15 | deck-search flag | `1.0` if `select.deck is not None` |
| 16 | looking flag | `1.0` if `state.looking is not None` |

**Player blocks (17–40)** — same 12 entries for each player, agent first:

| Agent idx | Opp idx | Value | Notes |
|---|---|---|---|
| 17 | 29 | `deckCount` | Cards left in deck |
| 18 | 30 | `handCount` | Hand size — the *only* opponent-hand information anywhere |
| 19 | 31 | `len(prize)` | Prizes remaining |
| 20 | 32 | `len(bench)` | Benched Pokémon |
| 21 | 33 | `benchMax` | Current bench capacity |
| 22 | 34 | has-active flag | `1.0` if the active slot is occupied |
| 23 | 35 | active-face-down flag | `1.0` if occupied but the Pokémon's identity is hidden |
| 24 | 36 | `poisoned` | Special condition on the active |
| 25 | 37 | `burned` | Special condition |
| 26 | 38 | `asleep` | Special condition |
| 27 | 39 | `paralyzed` | Special condition |
| 28 | 40 | `confused` | Special condition |

#### 13.3 `select_cats`, `context_card_ids`, `stadium_id`

- `select_cats = [int(select.type) + 1, int(select.context) + 1]` — embedding indices for the engine's `SelectType`/`SelectContext` enums; `[0, 0]` on terminal.
- `context_card_ids = [contextCard.id, effect.id]` — which card/effect *caused* the current selection (e.g. the trainer card being resolved); `0` for each absent field.
- `stadium_id = [state.stadium[0].id]`, or `[0]` if no stadium is in play.

#### 13.4 `options` — per-action feature table (97 rows)

Row `i` describes action `i` exactly; this alignment with the action mask is what lets the model score each row into a per-action logit. Rows past the actual option count — including row 96, the stop action — stay at their fill values (IDs and `cats` at `0`, `scalars` at `-1.0`, `owner` at `0`).

**`cats` (4 int64 columns)**, each `enum + 1` with `0` = absent:

| Column | Source | Notes |
|---|---|---|
| 0 | `option.type` | `OptionType`; always present for a real option, so never 0 there |
| 1 | `option.area` | `AreaType`: HAND / DISCARD / ACTIVE / BENCH / PRIZE / STADIUM / DECK / LOOKING |
| 2 | `option.inPlayArea` | Target board area for ATTACH/EVOLVE |
| 3 | `option.specialConditionType` | |

**`scalars` (6 float32 columns)**, raw values with `-1.0` = absent:

| Column | Source | Notes |
|---|---|---|
| 0 | `option.number` | |
| 1 | `option.count` | |
| 2 | `option.index` | Position within the referenced area |
| 3 | `option.toolIndex` | |
| 4 | `option.energyIndex` | |
| 5 | `option.inPlayIndex` | Position within `inPlayArea` |

**`owner`**: `1` if `option.playerIndex == agent_seat`, `2` if it is the opponent's seat, `0` if the option carries no player reference.

**`card_id` / `target_id` / `attack_id`**: the engine's options reference cards indirectly as (player, area, index) triples; `OptionReferenceResolver` dereferences them against the current state so the model never has to. Any unresolvable, face-down or hidden reference yields `0`. Resolution per `OptionType`:

| OptionType | `card_id` | `target_id` | `attack_id` |
|---|---|---|---|
| `CARD` | card at (owner, `area`, `index`) | 0 | 0 |
| `TOOL_CARD` | `pokemon.tools[toolIndex].id` | carrier Pokémon's ID | 0 |
| `ENERGY_CARD`, `ENERGY` | `pokemon.energyCards[energyIndex].id` | carrier Pokémon's ID | 0 |
| `PLAY`, `ABILITY`, `DISCARD` | card at (owner, `area` or HAND, `index`) | 0 | 0 |
| `ATTACH`, `EVOLVE` | played card at (owner, `area`, `index`) | Pokémon at (owner, `inPlayArea`, `inPlayIndex`) | 0 |
| `ATTACK` | 0 | 0 | `option.attackId` |
| `SKILL` | `option.cardId` | 0 | 0 |
| anything else | 0 | 0 | 0 |

So `target_id` means: the in-play Pokémon this option acts on — the attach/evolve target, or the carrier of a selected tool/energy. References with `area = DECK` dereference through the deck-search list `select.deck`; `area = LOOKING` through `state.looking`.

#### 13.5 `pokemon` — board table (18 rows)

| Rows | Content |
|---|---|
| 0 | Agent's active |
| 1–8 | Agent's bench slots 0–7 |
| 9 | Opponent's active |
| 10–17 | Opponent's bench slots 0–7 |

`mask[row] = True` means the slot is physically occupied. A **face-down active** is masked `True` but has `card_id = 0` and all features zero except the is-active flag — the agent knows something is there, not what. `tool_id` carries only the **first** attached tool (`pokemon.tools[0]`); further tools show up only in the tool-count feature.

**`features` (20 float32 columns):**

| Column | Value | Notes |
|---|---|---|
| 0 | `hp` | Current HP |
| 1 | `maxHp` | |
| 2 | `hp / maxHp` | `0.0` if `maxHp == 0` |
| 3 | `appearThisTurn` | Came into play this turn — e.g. can't evolve yet |
| 4 | is-active flag | `1.0` for rows 0 and 9 when occupied; set even for a face-down active |
| 5 | `len(tools)` | Attached tool count |
| 6 | `len(energyCards)` | Attached energy *card* count |
| 7 | `len(preEvolution)` | Evolution stage depth |
| 8–19 | provided-energy histogram | Column `8 + int(energy_type)` counts provided energy **units** of that `EnergyType` (12 types). Differs from column 6 because one card can provide multiple or differently-typed units |

#### 13.6 `my` / `opp` — zone tables

All follow the same pattern: IDs written front-aligned into a `0`-filled table, mask `True` for exactly the occupied slots.

- `my.hand_ids`: the agent's full hand, ordered as the engine reports it.
- `my.discard_ids` / `opp.discard_ids`: discard piles, fully visible for both players.
- `my.prize_ids` / `opp.prize_ids`: **face-down prizes occupy a slot with mask `True` but ID `0`** — the mask counts prizes remaining, the IDs reveal identity only if a card effect turned them face-up.
- `opp` has **no hand table**. The opponent's hand is hidden; the only signal is `handCount` in the globals (index 30).

#### 13.7 `select_deck` and `looking` — reveal tables

- `select_deck`: the card list of a deck search (`select.deck`) — what the agent is currently allowed to search through. All-zero mask when the current selection is not a deck search. Options with `area = DECK` index into this list.
- `looking`: cards currently revealed by a "look at" effect (`state.looking`), e.g. peeking at the top of the deck. `None` entries encode as ID `0`. All-zero mask when nothing is being looked at. Options with `area = LOOKING` index into this list.

#### 13.8 Terminal observation

When the game is over (`select is None`), the observation is still fully populated from the final state — boards, zones, globals game and player blocks — but everything selection-derived is at its absent value: globals indices 8–16 all `0.0`, `select_cats = [0, 0]`, `context_card_ids = [0, 0]`, the entire `options` table at fill values, and `select_deck` empty.

#### 13.9 Capacities and overflow

Every cap is an `__init__` parameter of the encoder; the defaults trace to engine constants where one exists:

| Parameter | Default | Origin |
|---|---|---|
| `max_options` | 96 | Empirical headroom (max seen under random play: 42; full deck search ≈ 60). Option table has `max_options + 1 = 97` rows so row `i` matches action `i` |
| `bench_cap` | 8 | Engine constant `BENCH_SIZE_MAX` (in-game default is 5; card effects can raise it) |
| `hand_cap` | 30 | Headroom; no engine limit on hand size |
| `discard_cap` | 60 | `DECK_SIZE` — a discard pile can never exceed a full deck |
| `prize_cap` | 6 | `PRIZE_SIZE` — always exactly 6 by rule |
| `deck_cap` | 60 | `DECK_SIZE` — largest possible deck-search reveal |
| `looking_cap` | 60 | `DECK_SIZE` |
| `energy_cap` | 16 | Headroom; no engine limit on attachments |
| `evolution_cap` | 2 | Basic → Stage 1 → Stage 2 = at most 2 pre-evolutions |

Derived: `pokemon_rows = 2 * (1 + bench_cap) = 18`.

**Overflow policy**: a zone longer than its cap is truncated with a one-time warning — *except* the option list, which raises (`ValueError`), because a truncated option table would desynchronize from the action mask. An option referencing an index outside a visible zone also raises (`IndexError`) rather than silently encoding "unknown".