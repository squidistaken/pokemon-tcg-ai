# The Pokémon TCG Battle, Mechanics and Formalism

What the cabt engine actually simulates, and a formal partially observable
stochastic game (POSG) description of it, tied to the `cg.api` dataclasses
and to the concrete spaces `TCGEnv` exposes (see `docs/torchrl_environment.md`
for the TorchRL integration itself). Read this before touching the
observation encoder or the action-space decomposition, they are encodings
of the objects defined here.

## Objective and win conditions

A game is a race to take all six of your **Prize** cards (`PlayerState.prize`,
length 6 at the start). You take one Prize each time you knock out an
opposing Pokémon (two Prizes for a `CardData.ex`/`megaEx` Pokémon). The
engine recognizes four ways a game ends, exposed as `LogType.RESULT.reason`
and reflected in `State.result`:

1. `reason=1`: a player has taken all of their Prize cards.
2. `reason=2`: a player cannot draw at the start of their turn (deck-out),
   because `PlayerState.deckCount` reached 0.
3. `reason=3`: a player has no Pokémon left in any zone after a knockout.
4. `reason=4`: a card effect ends the game directly.

`State.result` is `-1` while the battle is ongoing, and the winner's seat
index (or `2` for a draw) once one of these fires.

## Board and zones

Each `PlayerState` is exactly six zones:

| Zone | Field | Notes |
| --- | --- | --- |
| Active | `active: list[Pokemon \| None]` | length 0 or 1; the only Pokémon that can attack or be attacked |
| Bench | `bench: list[Pokemon]` | up to `benchMax` reserves |
| Hand | `hand: list[Card] \| None` | `None` for the opponent (hidden) |
| Deck | `deckCount: int` | count only; identity of remaining cards is not exposed |
| Discard | `discard: list[Card]` | face-up, fully observed for both players |
| Prize | `prize: list[Card \| None]` | `None` entries are cards you have not looked at |

The count fields (`deckCount`, `handCount`, `len(bench)`, `len(prize)`) are
always public, even when the contents behind them are not. This distinction
matters for the information-partition below.

## Setup

Each player has a 60-card deck (`load_deck` enforces this), draws 7, and
must have at least one Basic Pokémon in that hand or mulligan (reveal,
reshuffle, redraw; the opponent draws one extra card per mulligan). One
Basic goes to Active, the rest of the opening hand's Basics may fill the
Bench, and six cards are set aside face-down as Prizes. A coin flip decides
`State.firstPlayer`; the player going first cannot attack on their first
turn.

## A turn, and what a "selection" is

The engine does not expose turns as a unit. It exposes **selections**
(`Observation.select`), and a turn is a variable-length sequence of them
ending in an `OptionType.END` choice. Within a turn a player can, in almost
any order: play Basics to the Bench, evolve, play Trainer cards, use
Abilities, retreat, then attack. Two actions are capped at one per turn and
tracked as booleans on `State`: `energyAttached` (one manual Energy
attachment) and `supporterPlayed` (one Supporter). `State.turnActionCount`
counts selections taken this turn; `State.turn` counts turns, with turn 1
belonging to the first player.

## Attacking, damage, knockouts

The Active attacks the opponent's Active. `Attack.energies` is the cost,
`Attack.damage` the base damage; damage accumulates on the defender and
never heals on its own. `CardData.weakness` roughly doubles incoming damage
of that type, `CardData.resistance` reduces it. A Pokémon with accumulated
damage `>= maxHp` is knocked out: everything attached to it goes to the
discard, and the attacker takes a Prize (two for `ex`, three for `megaEx`).
Attack → knockout → Prize is the core loop the whole game funnels into.

## Evolution and status

Evolution is a fixed chain, Basic → Stage 1 → Stage 2, tracked by
`Pokemon.preEvolution` / `CardData.evolvesFrom`; a Pokémon cannot evolve the
turn it entered play, and evolving preserves damage but clears special
conditions. The five special conditions
(`SpecialConditionType`: `POISON`, `BURN`, `SLEEP`, `PARALYZE`, `CONFUSE`,
mirrored as booleans on `PlayerState`) only ever apply to the Active:
Poison and Burn deal chip damage between turns, Sleep and Paralysis block
attacking/retreating, Confusion risks self-damage on attack. Most clear when
that Pokémon leaves Active, which is what makes retreat (an Energy cost,
once per turn, `State.retreated`) both an escape hatch and a tempo cost.

## Card types and Scarlet & Violet-era mechanics

Every card is `CardType.POKEMON`, `ENERGY` (`BASIC_ENERGY`/`SPECIAL_ENERGY`),
or a Trainer (`ITEM`, unlimited per turn; `SUPPORTER`, one per turn;
`STADIUM`, one in play, affects both players; `TOOL`, attaches to a
Pokémon). `CardData` carries the current-era flags: `ex`/`megaEx` (extra
Prizes on knockout, described above), `tera` (untargetable and
damage-reduced while benched), `aceSpec` (at most one per deck). These flags
are what defines deck archetypes and the matchup structure an opponent
model has to generalize across.

---

## Formal game description

### Two-player partially observable stochastic game (POSG)

Ignoring the initial deck-selection/mulligan phase, a battle is

$$
\mathcal{G} = \big\langle\, \mathcal{I},\ \mathcal{S},\ \{\mathcal{A}_i\}_{i\in\mathcal{I}},\ \iota,\ p,\ \{r_i\}_{i\in\mathcal{I}},\ \{\mathcal{Z}_i\}_{i\in\mathcal{I}},\ \{o_i\}_{i\in\mathcal{I}},\ \rho_0 \,\big\rangle
$$

- $\mathcal{I} = \{0, 1\}$: the two seats, `State.yourIndex`.
- $\mathcal{S}$: the full engine state: both decks (contents and order),
  both hands, all in-play Pokémon (HP, attached energies/tools, evolution
  stack), both discards, both Prize piles, special conditions, turn
  counters, and the engine's internal RNG state. $\mathcal{S}$ is a
  structured, variable-length record, not a fixed-dimensional vector space;
  see "State space size" below for its component-wise size and why it is
  not enumerable.
- $\iota: \mathcal{S} \to \mathcal{I}$: the acting player at state $s$: one
  selection is made at a time, never simultaneously. `BattleHandle.select_player`
  is verified equal to `State.yourIndex` at every selection (see
  `docs/torchrl_environment.md`, "Engine access"). $\iota$ changes
  irregularly within a turn (e.g. the defender may be asked to assign
  damage or respond to an effect), not just at turn boundaries.
- $\mathcal{A}_i(s)$: player $i$'s legal moves at a state with $\iota(s)=i$,
  detailed in "Action space" below: state-dependent and heterogeneous, not
  a fixed symbol set. $|\mathcal{A}_i(s)|$ varies per state, from $1$ (a
  forced choice) up to the size of the largest option list the engine can
  produce (empirically ~60, for a full-deck search).
- $p(s' \mid s, a)$: the transition function. Stochastic: deck
  shuffles at setup and on shuffle effects, the first-player coin flip, and
  per-selection coin flips for Sleep/Confusion/certain effects
  (`LogType.COIN`). The engine's RNG is `std::random_device`-seeded and not
  exposed for seeding (`docs/torchrl_environment.md`, "Known properties").
- $r_i(s)$: terminal-only reward, defined below.
- $\mathcal{Z}_i$: player $i$'s observation space, elements $z$, formally
  the projection of $\mathcal{S}$ onto the information visible to seat $i$;
  see "Observation space" below for its component-wise size, and
  "Information partition" for exactly which parts of $\mathcal{S}$ it
  redacts.
- $o_i(z \mid s)$: the observation function, the probability of seat $i$
  seeing $z \in \mathcal{Z}_i$ when the true state is $s$.
- $\rho_0$: the initial-state distribution induced by both players' deck
  lists, the shuffle, the mulligan procedure, and the coin flip.

### State space size

$\mathcal{S}$ is finite but not remotely enumerable, which is precisely why
this problem needs function approximation rather than tabular RL. It has no
fixed dimensionality in the vector-space sense (its zones are variable-length
lists), so the relevant question is its component-wise size rather than a
coordinate count. Per player:

| Component | Size / range |
| --- | --- |
| Deck order | permutation of up to 60 slots |
| Card identity (any card-typed field) | one of $1{,}267$ `cardId` values (`all_card_data()`) |
| Hand, discard | up to 60 cards each |
| Prize pile | 6 slots, each revealed or face-down |
| Active + Bench | 1 + up to `benchMax` (typically 5) Pokémon slots |
| Per-Pokémon HP | integer in $[0, \texttt{maxHp}]$, $\texttt{maxHp} \in [30, 380]$ across the card pool |
| Per-Pokémon attached Energy | multiset over 12 `EnergyType` values |
| Special conditions (Active only) | 5 independent bits |
| Turn/phase counters | `turn`, `turnActionCount` (unbounded ints), `firstPlayer` $\in \{-1,0,1\}$, 4 per-turn-use booleans |

The dominant factor is deck order: each player's 60-card deck contributes up
to $60!$ orderings of its undrawn suffix, already far beyond any tabular
representation before accounting for the card-identity, board-state, and
counter components layered on top. This is also why $\mathcal{A}_i(s)$ below
and $\mathcal{Z}_i$ above are structured objects rather than fixed-length
vectors: there is no natural fixed dimensionality to assign to $\mathcal{S}$
itself, only to specific encodings of it (see "Concrete spaces in `TCGEnv`").

### Action space

At a state with $\iota(s) = i$, the engine offers `select.option`, a list of
`Option` records: heterogeneous structs whose shape depends on
`select.type` (`SelectType`, 11 values: `MAIN`, `CARD`, `ENERGY`, `ATTACK`,
`EVOLVE`, `COUNT`, `YES_NO`, ...). An index into this list is only meaningful
relative to the state it was drawn from: index 3 might mean "play this
Trainer card" at one selection and "retreat to this Bench slot" at the next.
This is why the action space is *positional*, not *semantic*: a policy
must interpret an action through the paired observation, not through a
fixed action identity (unlike, say, "move up" in a gridworld).

When `select.maxCount > 1` (~3% of selections), the legal move is not a
single index but a **subset**: $S \subseteq \{0, \dots, n(s)-1\}$ with
$|S| \in [\texttt{minCount}(s), \texttt{maxCount}(s)]$, submitted to the
engine in one call: a combinatorial action space of size
$\sum_{k=\texttt{minCount}}^{\texttt{maxCount}} \binom{n(s)}{k}$ at that state
(observed `maxCount` $\le 3$ across the card pool, so this stays small even
though $n(s)$ does not).

$n(s) = |\texttt{select.option}|$ has no fixed value: it was observed up to
$60$ (a full-deck search) under random play, with $42$ the largest seen in
practice. `TCGEnv` pads this to the fixed-size $\mathcal{A}^{\text{RL}}$
described in "Concrete spaces in `TCGEnv`" below, since a `Categorical`
policy needs a fixed action dimensionality; states with $n(s)$ below the pad
size are handled by the action mask, and states with $n(s)$ above it are
truncated with a warning (see `docs/torchrl_environment.md`, "Action space").

### Observation space

At a state with $\iota(s) = i$, $z = o_i(s)$ is exactly the
engine's `Observation` object: three fields, none fixed-size.

| Field | Size / range |
| --- | --- |
| `select` | current selection context: `type` (`SelectType`, 11 values), `context` (`SelectContext`, 49 values), `minCount`/`maxCount` (small ints, `maxCount` $\le 3$ when $>1$), `option` (the state-dependent list from "Action space", up to $n(s)$), `deck` (present only during a deck search, up to `deckCount` cards) |
| `logs` | variable-length list of `Log` entries (`LogType`, 24 values) since the previous selection; can span an entire skipped opponent turn |
| `current` | the full `State`: turn/phase counters plus `players: list[PlayerState]` for **both** seats, using the zones from "Board and zones" above |

Critically, $z$ is not a redaction of only the acting player's own board:
both `PlayerState`s are included, with the same hidden/public split as
"Information partition" below, own `hand` visible, opponent's `hand` is
`None`, both `deckCount`/`handCount`/`len(bench)`/`len(prize)` always
visible, both `active`/`bench` Pokémon fully identified (board state is
always public, see "Board and zones"). Concretely,
$\mathcal{Z}_i = \mathcal{S}$ with exactly the six components in
"Information partition" removed: $z$ has the same variable-length,
structured shape as $\mathcal{S}$ (deck order, hand and board contents,
etc.), it is strictly smaller only in the specific fields it drops, not in
overall shape. `TCGEnv` encodes this whole object into fixed-shape padded
and masked tensors without discarding information (apart from `logs`); see
"Concrete spaces in `TCGEnv`" and `docs/torchrl_environment.md` for that
encoding.

### Reward and termination

Reward is terminal-only, read from `State.result`:

$$
r_0(s_T) = \begin{cases}
+1 & \text{result}(s_T) = 0 \\
-1 & \text{result}(s_T) = 1 \\
r_{\text{draw}} & \text{result}(s_T) = 2
\end{cases}
\qquad r_1(s_T) = -r_0(s_T)
$$

exactly zero-sum when $r_{\text{draw}} = 0$ (the current default,
`TCGEnv(reward_draw=0.0)`). All non-terminal transitions have
$r_i(s) = 0$.

### Information partition and belief state

The imperfect information is a small, bounded, *named* set of unknowns.
Conveniently, the engine tells you exactly what they are, because
`cg.api.search_begin` requires you to supply a prediction for each one in
order to run a determinized search:

| Hidden component | `search_begin` argument | Public counterpart |
| --- | --- | --- |
| Your deck's order | `your_deck` | `deckCount` (composition is known, since you built the deck) |
| Opponent's deck (contents and order) | `opponent_deck` | `deckCount` |
| Your face-down Prizes | `your_prize` | `len(prize)` |
| Opponent's face-down Prizes | `opponent_prize` | `len(prize)` |
| Opponent's hand contents | `opponent_hand` | `handCount` |
| Opponent's face-down Active (rare) | `opponent_active` | presence of an active slot |

Both players' in-play Pokémon (Active and Bench, with HP, energy, tools),
both discard piles, the Stadium, and turn counters are all fully observed
by both seats: the game hides *card identity* in specific zones, never
*board state*. Player $i$'s belief over the true state factorizes
(approximately, consistent with the public discard/play history) along
exactly these six unknowns:

$$
b_i(s) \;=\; b_i(\text{deck}_i) \cdot b_i(\text{deck}_{1-i}) \cdot b_i(\text{prize}_i) \cdot b_i(\text{prize}_{1-i}) \cdot b_i(\text{hand}_{1-i}) \cdot b_i(\text{active}_{1-i})
$$

where $b_i(\text{deck}_i)$ is a belief over orderings of a *known* multiset
(pure card-counting), while $b_i(\text{deck}_{1-i})$ and
$b_i(\text{hand}_{1-i})$ are beliefs over unknown card identities as well.

## Reduction to a Partially Observable Markov Decision Process (POMDP) for training

`TCGEnv` fixes an opponent policy $\pi_{1-i}$ (`RandomOpponent` or an
`OpponentPool` member, `docs/torchrl_environment.md`, "Opponent handling
and self-play") and plays it internally
(`TCGEnv._advance_to_agent`). From the agent's seat $i$, the two-player POSG
above collapses to a single-agent POMDP:

$$
\mathcal{M}_i^{\pi_{1-i}} = \big\langle\, \mathcal{S},\ \mathcal{A}_i,\ p^{\pi_{1-i}},\ r_i,\ \mathcal{Z}_i,\ o_i,\ \rho_0 \,\big\rangle
$$

with

$$
p^{\pi_{1-i}}(s' \mid s, a) \;=\; \sum_{\substack{\tau:\ s \xrightarrow{a} \cdots \xrightarrow{} s' \\ \iota = 1-i \text{ or chance along } \tau}} p(\tau) \, \pi_{1-i}(\tau)
$$

i.e. the kernel marginalizes over every opponent action and chance event
between the agent's selection and its next one: exactly what
`_advance_to_agent`'s loop computes by simulation rather than by summation.
Because $\pi_{1-i}$ is periodically swapped (self-play pool draws a new
member `on_reset()`), $p^{\pi_{1-i}}$ is stationary *within* an
episode but not across episodes: the standard self-play framing of a
non-stationary opponent as a distribution over stationary POMDPs.

### Concrete spaces in `TCGEnv`

The action space is padded to a fixed size and the combinatorial
multi-select is decomposed into a sequential factorization:

$$
\mathcal{A}^{\text{RL}} = \{0, \dots, \texttt{max\_options}-1\} \cup \{\text{stop}\},
\qquad |\mathcal{A}^{\text{RL}}| = \texttt{max\_options} + 1 = 97
$$

with `max_options=96` (`TCGEnv.__init__` default). A legal subset $S$ is
recovered by picking its elements one at a time (already-picked indices
masked out) then emitting `stop` once $|S| \ge \texttt{minCount}(s)$:

$$
\pi(S \mid s) \;=\; \pi(\text{stop} \mid s, S) \prod_{a \in S} \pi(a \mid s, S_{<a})
$$

implemented by `TCGEnv._step` / `_build_mask` (`tcg_env.py`). The
observation space is the action mask together with
`StructuredObservationEncoder`'s output: a nested TensorDict of padded,
masked, fixed-shape tensors that preserves the information set rather than
summarizing it — raw card IDs per zone (hand, boards, discards, prizes,
deck-search results), a per-Pokemon feature table, scalar game/selection
context, and a per-option table whose row $i$ describes exactly action $i$
of $\mathcal{A}^{\text{RL}}$ (the hook for a pointer-style actor). The full
schema and its conventions are the contract in `docs/torchrl_environment.md`.
The only dropped field is `logs`; its non-recoverable content (revealed
hidden cards, coin results) is deferred to a belief-tracking feature. Note
$z$ remains a single draw $z \sim o_i(\cdot \mid s)$: a policy that
conditions solely on the current $z$ (no recurrence, no running belief over
$b_i$) is Markov in $\mathcal{Z}_i$ but not in the true information state: a
gap that matters most for the deck-order and hand-tracking components of
$b_i$, and a candidate place to spend model capacity in the agent.

The true horizon is unbounded by the rules (a very long deck-out race is
legal); `TCGEnv` adds a `max_engine_selections` safety cap (default 5000)
that truncates rather than terminates the episode: an artifact of the
training setup, not part of $\mathcal{G}$ itself.
