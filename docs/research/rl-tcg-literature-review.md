# RL for Two-Player Trading Card Games — Literature Review (2021–2026)

> **Scope.** Reinforcement-learning approaches for two-player TCGs/CCGs with deck
> construction and hidden information (Legends of Code and Magic, Hearthstone, Magic: The
> Gathering, and related). Priority lens: **attention / transformer** architectures for
> state and action representation; secondary coverage of deep RL, self-play, PPO/DQN, MCTS
> hybrids, large/variable discrete action spaces, imperfect information, and action masking.
>
> **Method & confidence.** Produced by the deep-research harness: 5 search angles → 18
> sources fetched → 81 claims extracted → top 25 claims put through 3-vote adversarial
> verification (needs 2/3 to survive). **All 25 survived (25 confirmed, 0 refuted.)** Papers
> in **Part 1** are backed by verified, quoted claims. Papers in **Part 2** were surfaced by
> search but their specific claims were *not* in the independently-verified top-25 — treat
> those summaries as unverified leads. Publication years/venues are best-effort from source
> metadata; confirm before citing formally.

---

## TL;DR — the landscape

- **The dominant public benchmark is Legends of Code and Magic (LOCM)** — a deliberately
  small, open CCG built as an AI testbed, and the basis of the multi-year *Strategy Card
  Game AI Competition* (IEEE CEC/CoG). Most rigorous RL-for-TCG work you can actually
  reproduce is on LOCM. Pokémon TCG has **essentially no published RL literature** — you are
  working in a gap.
- **The recurring recipe is PPO + invalid action masking + self-play**, with the game framed
  as an MDP and a **flat MLP** as the network. This is the well-trodden baseline (Vieira et
  al.; the LOCM competition entries).
- **The strongest agents (ByteDance's "ByteRL")** go beyond naive self-play: an **end-to-end
  policy over both deck-building and battle**, trained with **Optimistic Smooth Fictitious
  Play** to approximate a Nash equilibrium. It won the COG2022 competition and beat a top-10
  Hearthstone streamer — but was later shown to be **exploitable** in restricted deck pools.
- **Attention/transformers are still emergent here, not established.** The clearest
  transformer result is **DTCard** (Decision Transformer). Two other attention-relevant
  papers (transformer-as-policy for variable action spaces; action-representation for
  large action spaces) are adjacent but were not verified in this run. Notably, the
  best-known LOCM battle paper explicitly lists **Set Transformers / Deep Sets as future
  work** — i.e., permutation-equivariant attention is seen as the obvious next step but
  wasn't yet done.
- **Two structural challenges dominate every paper:** (1) **large + variable discrete action
  spaces** (handled via action masking and/or auto-regressive/entity-factored actions), and
  (2) **deck-building as a separate combinatorial problem** from in-game play.

---

# Part 1 — Papers with verified claims

## 1. DTCard: A Framework for Decision Transformers in Card Games
- **Venue/year:** *Applied Sciences* (MDPI), 2026 · **Link:** <https://doi.org/10.3390/app16073117>
- **Core method:** Casts card-game play as sequence modeling using a **Decision Transformer**
  rather than value/policy iteration.
- **How attention is used (priority):** ✅ **Central.** Uses the **self-attention mechanism of
  the Decision Transformer over a fixed episodic context window** to capture long-range
  temporal dependencies and **implicitly track hidden information without recurrent memory or
  explicit search trees.** This is the most on-point "attention for TCG state representation"
  result found.
- **Takeaway:** The strongest signal that transformer/sequence-modeling approaches transfer to
  hidden-information card games. Best starting point for your attention-first interest.
  > *"DTCard leverages the self-attention mechanism of Decision Transformers over a fixed
  > episodic context window to capture long-range temporal dependencies and implicitly track
  > hidden information without requiring recurrent memory or explicit search trees."*

## 2. Mastering Strategy Card Game (Legends of Code and Magic) via End-to-End Policy and Optimistic Smooth Fictitious Play  ("ByteRL")
- **Authors/year:** Xi, Zhang, Xiao et al. (ByteDance), 2023 · **Link:** <https://arxiv.org/abs/2303.04096>
- **Game:** Legends of Code and Magic (two-stage: draft + battle).
- **Core method:** A **single end-to-end policy** spanning both the deck-building/draft stage
  and the battle stage, trained with an **Optimistic Smooth Fictitious Play (OSFP)** algorithm
  to approximate the **Nash equilibrium** of the two-player game.
- **Attention:** Not the focus of the verified claims.
- **Results:** **Won double championships at the COG2022 competition** — the reference SOTA for
  LOCM.
  > *"We … propose an end-to-end policy … We also propose an optimistic smooth fictitious play
  > algorithm to find the Nash Equilibrium for the two-player game."* / *"Our approach wins
  > double championships of COG2022 competition."*

## 3. Mastering Strategy Card Game (Hearthstone) with Improved Techniques
- **Authors/year:** ByteDance group, 2023 · **Link:** <https://arxiv.org/pdf/2303.05197>
- **Game:** **Hearthstone** — full game, both deck building **and** battle.
- **Core method:** Applies the end-to-end policy + **Optimistic Smooth Fictitious Play**
  (self-play against a mixture of historical models) to the much larger Hearthstone.
- **Large/variable action space:** Handled via **auto-regressive decomposition** — a card play
  is a `(type, target)` tuple — together with a **0/1 action mask** that changes each timestep.
- **Results:** The trained models **defeated a top-10-ranked (China region) Hearthstone
  streamer in all Best-of-5 tournaments** of full games (deck building + battle).
  > *"we employ a 0/1 action mask to indicate the available actions that varies at each time
  > step."* / *"Our models defeat the human player in all Best-of-5 tournaments of full games."*

## 4. Learning to Beat ByteRL: Exploitability of Collectible Card Game Agents
- **Year/venue:** 2024 (ALA'24 workshop) · **Links:** <https://arxiv.org/abs/2404.16689> · <https://arxiv.org/html/2404.16689v1>
- **Game:** Legends of Code and Magic 1.5.
- **Core method:** A two-stage attack on the black-box SOTA agent: **behavior cloning → PPO
  fine-tuning** against ByteRL, i.e. techniques that work even without white-box access.
- **Results:** An RL-fine-tuned agent reached an **80.1% win rate against ByteRL when restricted
  to 256 deck pools** — showing a strong self-play/Nash-seeking agent is **highly exploitable
  within restricted deck distributions.**
- **Takeaway:** A caution for anyone training via self-play — approximate-Nash agents can be
  brittle off their training distribution. Robustness/exploitability should be measured, not
  assumed.
  > *"its play in LOCM 1.5 is 'highly exploitable' within restricted deck pools."*

## 5. Causal Reinforcement Learning for Complex Card Games: A Magic: The Gathering Benchmark (MTG-Causal-RL)
- **Year:** 2026 preprint · **Link:** <https://arxiv.org/pdf/2605.06066> *(very recent; verify metadata)*
- **Game:** **Magic: The Gathering** (Standard 2025 archetypes).
- **Core method:** Introduces **MTG-Causal-RL**, a **Gymnasium benchmark**: a
  **3,077-dimensional partial observation**, a **478-action masked discrete action space**,
  five competitive Standard archetypes, three reward schemes, and a hand-specified
  **Structural Causal Model** over strategic variables. Compares **masked PPO** vs. a causal
  agent **CGFA-PPO**.
- **Attention / action space:** Masked discrete action space (478 actions); attention not the
  focus.
- **Results:** Both agents beat uniform-random on all five decks, but **neither dominates**:
  CGFA-PPO wins on Azorius Control (25.6% vs 20.6%) and Mono-Red Aggro (70.6% vs 67.8%); plain
  PPO wins on Boros Convoke, Dimir Midrange, and Domain Ramp.
- **Takeaway:** The first structured **MTG RL benchmark** with masked actions + partial
  observability — a useful reference design for building your own Pokémon TCG environment/spec.

## 6. Exploring Deep Reinforcement Learning for Battling in Collectible Card Games
- **Authors/year:** R. Vieira, A. R. Tavares, L. Chaimowicz — **SBGames 2022** (also IEEE CoG) · **Links:** <https://homepages.dcc.ufmg.br/~ronaldo.vieira/assets/pdf/sbgames-2022.pdf> · [ResearchGate](https://www.researchgate.net/publication/365831654_Exploring_Deep_Reinforcement_Learning_for_Battling_in_Collectible_Card_Games)
- **Game:** LOCM 1.2 — **battle phase**.
- **Core method:** Battle framed as an MDP; a **PPO variant** trains an **MLP** via **self-play**;
  outputs a single in-game action.
- **Large action space:** **Invalid action masking** over 145 actions — logits of invalid actions
  set to −∞ before softmax (→ zero probability). Reported as **critical for convergence**.
- **Results:** Modest — ~**51% win rate vs. max-attack (MA)** and ~**36% vs. one-step-lookahead
  (OSL)**, **well below tree-search SOTA** (which beats MA ≈100%).
- **Takeaway:** The canonical "plain deep-RL baseline" for TCG battle, and honest about its
  ceiling: flat-MLP PPO alone does not reach search-based performance.
  > *"before the softmax activation, all logits that refer to invalid actions are set to
  > -infinity. As a result, they have zero probability."*

## 7. Towards Sample-Efficient Deep Reinforcement Learning in Collectible Card Games
- **Venue/year:** *Entertainment Computing* (Elsevier), 2023 · **Link:** <https://www.sciencedirect.com/science/article/abs/pii/S1875952123000496>
- **Game:** LOCM 1.5 — battle phase (extended/updated line of work from #6).
- **Core method:** Battle as MDP; **PPO variant on an MLP** (no tree search); one action per step.
- **Attention (priority, notable):** ❌ Not used — **but explicitly proposes permutation-
  equivariant attention (Deep Sets, Set Transformers) as future work** to make the agent
  invariant to card order and reduce learning effort. This is direct evidence that attention
  is the recognized "next step" for TCG state encoders.
  > *"Deep Sets …, Set Transformers …, or other permutation-equivariant architectures … may be
  > a good fit."*

## 8. Exploring Reinforcement Learning Approaches for Drafting in Collectible Card Games
- **Authors/venue:** R. Vieira, L. Chaimowicz, A. R. Tavares — *Entertainment Computing*, 2022 · **Link:** <https://www.sciencedirect.com/science/article/abs/pii/S1875952122000490>
- **Game:** LOCM — **arena/drafting (deck-building) phase**.
- **Core method:** Drafting framed as an MDP; **three DRL approaches trained via self-play**,
  differing in how they use previously-drafted-card history:
  1. **History** — history-aware **MLP** exploiting card synergies;
  2. **LSTM** — retains information about past picks;
  3. **Immediate** — history-oblivious MLP.
- **Takeaway:** The reference treatment of **deck construction as a sequential RL problem** — and
  a concrete study of how much modeling pick-history matters (relevant if you ever learn deck
  selection rather than fixing a 60-card list).
  > *"three DRL approaches trained in self-play that differ on how to handle information from
  > previously drafted cards."*

## 9. Learning With Generalised Card Representations for Magic: The Gathering
- **Authors/year:** Bertram, Fürnkranz, Müller — 2024 · **Links:** <https://arxiv.org/abs/2407.05879> · <https://arxiv.org/html/2407.05879v1>
- **Game:** Magic: The Gathering — **deck-building / draft choices**.
- **Core method (note: not RL):** **Contextual Preference Ranking** with a **Siamese neural
  network** trained on human draft-preference triplets. Card/deck encoders are **fully-connected
  + convolutional layers** (sentence-transformers used *only* for text preprocessing) — **no
  attention/transformer in the model itself.** Builds *generalised* card representations from
  numeric, nominal, text, image, and meta-usage features so the model handles **unseen/newly-
  released cards.**
- **Results:** Predicts ~**55% of human draft choices on completely unseen cards** (vs ~22%
  random) — evidence the representation captures card quality/strategy rather than memorizing a
  fixed card set.
- **Takeaway:** Most relevant for the **card-embedding** problem — how to represent thousands of
  cards (incl. new ones) so a policy generalizes. Directly analogous to encoding the Pokémon TCG
  card pool.

## 10. A Closer Look at Invalid Action Masking in Policy Gradient Algorithms
- **Authors/year:** Huang & Ontañón, 2020 (later FLAIRS) · **Link:** <https://arxiv.org/abs/2006.14171>
- **Core method / result:** The **theoretical + empirical reference for invalid action masking.**
  Shows masking yields a **valid policy gradient** (masked logits still form a valid
  distribution whose gradient is well-defined), and that its **benefit grows as the proportion
  of invalid actions increases** — i.e. increasingly essential for large discrete action spaces.
- **Takeaway:** The citation to justify action masking in your Pokémon TCG env (where legal
  options vary per step). *Note: 2020 — just outside the strict 5-year window but foundational
  and universally cited by the TCG papers above.*

## 11. Summarizing Strategy Card Game AI Competition
- **Authors/venue:** Kowalski & Miernik — IEEE CoG, 2023 · **Link:** <https://arxiv.org/abs/2305.11814>
- **What it is:** A **5-year retrospective** of the Strategy Card Game AI Competition built on
  LOCM (through 2022). Confirms LOCM was **designed as a research testbed** and surveys the field
  across **game-tree search, neural networks, evaluation functions, and CCG deck-building.**
- **Takeaway:** Best single entry point to the LOCM ecosystem and the range of methods that have
  been tried (search vs. learning). Read this first for orientation.

---

# Part 2 — Additional relevant papers (surfaced, NOT independently verified in this run)

> These were returned by the searches but their specific claims did not make the verified
> top-25. Summaries are from search snippets — **verify before relying on specifics.**

- **Transformers as Policies for Variable Action Environments** — Bamford & Ovalle, 2023 ·
  <https://arxiv.org/abs/2301.03679>. Uses a **transformer encoder as the policy network** with
  **self-attention over game entities** to handle **variable-size, entity-structured action
  spaces**, trained with PPO. Not TCG-specific (Griddly), but the **most directly relevant
  "attention-for-variable-actions" architecture** for your priority lens.
- **Towards Modern Card Games with Large-Scale Action Spaces Through Action Representation** —
  2022 · <https://arxiv.org/pdf/2206.12700>. Tackles the **large-scale action-space** problem via
  **learned action representations** (embedding actions rather than enumerating them) — relevant
  to Pokémon TCG's large/variable option set.
- **Two-Step Reinforcement Learning for Multistage Strategy Card Game** — 2023 ·
  <https://arxiv.org/html/2311.17305v1>. Decomposes the multistage (draft + battle) LOCM problem
  into a **two-step RL** procedure.
- **Policy-Based Reinforcement Learning Approach in Imperfect Information Card Game** —
  *Applied Sciences*, 2025 · <https://www.mdpi.com/2076-3417/15/4/2121>. Policy-gradient RL under
  **imperfect information** — relevant to hidden-hand handling.
- **Drafting in Collectible Card Games via Reinforcement Learning** — Vieira, Tavares,
  Chaimowicz, SBGames **2020** · <https://www.sbgames.org/proceedings2020/ComputacaoFull/209690.pdf>.
  The **predecessor** to #8 (just outside the 5-year window); establishes the drafting-as-MDP
  framing.

---

# Relevance to this project (Pokémon TCG RL)

There is **no published RL work on the Pokémon TCG specifically** — the transferable knowledge
comes from LOCM, Hearthstone, and MTG. Concrete carry-overs:

1. **Action masking is non-negotiable.** Every strong agent uses it; #10 gives the theory, #6
   found it critical for convergence. Matches your planned `action_mask` + max-option
   `Categorical` spec. (See `docs/torchrl/01-custom-env.md`.)
2. **Factor the action, don't enumerate it.** Hearthstone's `(type, target)` **auto-regressive**
   decomposition (#3) and learned **action representations** (#2, Part 2) are the scalable ways
   to handle large/variable option sets — worth considering over a single flat softmax.
3. **Attention is the recognized frontier, still under-explored.** DTCard (#1) shows Decision-
   Transformer self-attention works on hidden-info card games; #7 explicitly names **Set
   Transformers / Deep Sets** as the fix for card-order invariance; #Part-2 #1 shows transformer
   policies for variable action spaces. A **permutation-equivariant (set-attention) encoder over
   your board/hand entities** is the well-motivated architecture to try.
4. **Self-play reaches SOTA but is exploitable.** ByteRL (#2, #3) + its exploitation (#4) argue
   for **fictitious-play / population-based self-play** over naive self-play, and for measuring
   exploitability.
5. **Card representation generalization matters.** #9's generalised embeddings (numeric + text +
   image + meta) predicting choices for **unseen cards** is the template for encoding a large,
   evolving Pokémon card pool.
6. **Baseline expectation:** plain PPO+MLP+masking is a *starting* baseline that historically
   **underperforms search** (#6). Budget for a stronger method (population self-play, attention
   encoder, or search hybrid) to be competitive.

---

## Sources (verification status)

| # | Paper | Link | Verified claims |
|---|-------|------|:---:|
| 1 | DTCard: Decision Transformers in Card Games | https://doi.org/10.3390/app16073117 | ✅ |
| 2 | Mastering SCG (LOCM) — End-to-End + OSFP (ByteRL) | https://arxiv.org/abs/2303.04096 | ✅ |
| 3 | Mastering SCG (Hearthstone) — Improved Techniques | https://arxiv.org/pdf/2303.05197 | ✅ |
| 4 | Learning to Beat ByteRL — Exploitability | https://arxiv.org/abs/2404.16689 | ✅ |
| 5 | Causal RL for Card Games — MTG-Causal-RL | https://arxiv.org/pdf/2605.06066 | ✅ |
| 6 | Exploring DRL for Battling in CCGs | https://homepages.dcc.ufmg.br/~ronaldo.vieira/assets/pdf/sbgames-2022.pdf | ✅ |
| 7 | Towards Sample-Efficient DRL in CCGs | https://www.sciencedirect.com/science/article/abs/pii/S1875952123000496 | ✅ |
| 8 | Exploring RL Approaches for Drafting in CCGs | https://www.sciencedirect.com/science/article/abs/pii/S1875952122000490 | ✅ |
| 9 | Generalised Card Representations for MTG | https://arxiv.org/abs/2407.05879 | ✅ |
| 10 | A Closer Look at Invalid Action Masking | https://arxiv.org/abs/2006.14171 | ✅ |
| 11 | Summarizing Strategy Card Game AI Competition | https://arxiv.org/abs/2305.11814 | ✅ |
| P2 | Transformers as Policies for Variable Action Envs | https://arxiv.org/abs/2301.03679 | — |
| P2 | Large-Scale Action Spaces via Action Representation | https://arxiv.org/pdf/2206.12700 | — |
| P2 | Two-Step RL for Multistage Strategy Card Game | https://arxiv.org/html/2311.17305v1 | — |
| P2 | Policy-Based RL in Imperfect Information Card Game | https://www.mdpi.com/2076-3417/15/4/2121 | — |
| P2 | Drafting in CCGs via RL (2020, predecessor) | https://www.sbgames.org/proceedings2020/ComputacaoFull/209690.pdf | — |

*Generated by the deep-research workflow (5 angles · 18 sources · 81 claims → 25 verified,
3-vote adversarial). The workflow's automatic synthesis step was interrupted by a session
token limit; this write-up was synthesized manually from the verified claim set.*
