# Synthesis and Checkup

Personal notes on each paper from the old `rl-tcg-literature-review.md`. Each section has a one-line
recap for context plus a **Notes** block for my [matthijs] own take.

## Index (papers that were found from deep research style search)

| # | Paper | Link |
|---|-------|------|
| 1 | DTCard: Decision Transformers in Card Games | https://doi.org/10.3390/app16073117 |
| 2 | Mastering SCG (LOCM) — End-to-End + OSFP (ByteRL) | https://arxiv.org/abs/2303.04096 |
| 3 | Mastering SCG (Hearthstone) — Improved Techniques | https://arxiv.org/pdf/2303.05197 |
| 4 | Learning to Beat ByteRL — Exploitability | https://arxiv.org/abs/2404.16689 |
| 5 | Causal RL for Card Games — MTG-Causal-RL | https://arxiv.org/pdf/2605.06066 |
| 6 | Exploring DRL for Battling in CCGs | https://homepages.dcc.ufmg.br/~ronaldo.vieira/assets/pdf/sbgames-2022.pdf |
| 7 | Towards Sample-Efficient DRL in CCGs | https://www.sciencedirect.com/science/article/abs/pii/S1875952123000496 |
| 8 | Exploring RL Approaches for Drafting in CCGs | https://www.sciencedirect.com/science/article/abs/pii/S1875952122000490 |
| 9 | Generalised Card Representations for MTG | https://arxiv.org/abs/2407.05879 |
| 10 | A Closer Look at Invalid Action Masking | https://arxiv.org/abs/2006.14171 |
| 11 | Summarizing Strategy Card Game AI Competition | https://arxiv.org/abs/2305.11814 |
| P2-1 | Transformers as Policies for Variable Action Envs | https://arxiv.org/abs/2301.03679 |
| P2-2 | Large-Scale Action Spaces via Action Representation | https://arxiv.org/pdf/2206.12700 |
| P2-3 | Two-Step RL for Multistage Strategy Card Game | https://arxiv.org/html/2311.17305v1 |
| P2-4 | Policy-Based RL in Imperfect Information Card Game | https://www.mdpi.com/2076-3417/15/4/2121 |
| P2-5 | Drafting in CCGs via RL (2020, predecessor) | https://www.sbgames.org/proceedings2020/ComputacaoFull/209690.pdf |

---

## 1. DTCard: A Framework for Decision Transformers in Card Games
**Link:** https://doi.org/10.3390/app16073117

[Clanker Recap]: Decision Transformer for card games — self-attention over a fixed context window,
implicitly tracks hidden info without recurrence. Offline imitation, not RL (no value
function, no rollouts), so the training paradigm doesn't fit a self-play PPO loop even
though the attention architecture is the clearest "attention works for hidden-info TCG
state" result in the review.

**Relevant/useful:** not at all, because its decision transformers, offline imitation style RL, not applicable. 

**Notes:**: The environments they used are too specific of a card game for us to really get some ideas on how to encode the TCG observations.

---

## 2. Mastering Strategy Card Game (LOCM) via End-to-End Policy and Optimistic Smooth Fictitious Play ("ByteRL")
**Link:** https://arxiv.org/abs/2303.04096

Clanker Recap: Single end-to-end policy over draft + battle, trained with Optimistic Smooth
Fictitious Play (OSFP) to approximate a Nash equilibrium. Won COG2022. The reference SOTA
for LOCM and the source of the OSFP mechanism (opponent-pool mixture + win-rate-gated
checkpoint additions) discussed in the synthesis section.

**Relevant/useful:** yes, as it covers a sophisticated self-play algorithm for Nash Equilibriums in two-player games. Pokemon TCG is such a game. They use online self-play PPO-style RL which we also do.

**Notes:**: we should look more into their self-play approach, and how they handle card embeddings (e.g. shared). They seem to be using an LSTM. Also has some information about how they handle (shared) card embeddings. This one we should read in detail.

---

## 3. Mastering Strategy Card Game (Hearthstone) with Improved Techniques
**Link:** https://arxiv.org/pdf/2303.05197

Recap: Same end-to-end + OSFP approach scaled to full Hearthstone. Large action space
handled via auto-regressive `(type, target)` decomposition + a 0/1 action mask that changes
per timestep. Beat a top-10-ranked human streamer in Bo5.

**Relevant/useful:** Same as above (2)
**Notes:** Same as above (2), but none of these seem to involve novel model archtiecture involving self-attention laters, we could probably come up with something ourselves even if the literature is sparse.

---

## 4. Learning to Beat ByteRL: Exploitability of Collectible Card Game Agents
**Link:** https://arxiv.org/abs/2404.16689

Clanker Recap: Behavior-clone-then-PPO-fine-tune attack against ByteRL, no white-box access needed.
Reached 80.1% win rate when restricted to 256 deck pools; exploitability falls
monotonically as the deck pool widens (0.904 at 32 decks → 0.542 at 1024). Shows
approximate-Nash self-play agents can be brittle off their training distribution —
motivates building an exploitability eval into training rather than trusting a single
win-rate number.

**Relevant/useful:** mixed

**Notes:**: Does not seem to have that many novel contributions over the two bytedance papers listed above. Methodology is quite flowed. Clanker summary because I can not be bothered to write this out myself in detail:

  Two problems: first, their "attacker" gets to use ByteRL's own drafted deck instead of facing ByteRL's real
  draft+battle policy, so it's only attacking half the agent under favorable conditions, not a fair best response.
    Second, the win-rate results only hold on tiny fixed deck pools (32–1024 decks); as the pool grows toward the real,
    near-infinite deck space, the win rate collapses back to parity, meaning the agent overfit to a shrunk version of the  game rather than proving ByteRL is actually exploitable.

---

## 5. Causal Reinforcement Learning for Complex Card Games (MTG-Causal-RL)
**Link:** https://arxiv.org/pdf/2605.06066

Recap: Gymnasium benchmark for MTG — 3,077-dim partial observation, 478-action masked
discrete action space, five Standard archetypes, hand-specified Structural Causal Model.
Compares masked PPO vs. causal agent CGFA-PPO; neither dominates across all five decks. Useful
as a reference env/benchmark design (masked actions + partial observability) even if the
causal-RL angle itself isn't a priority.

**Relevant/useful:** Not useful because it is an environment/benchmark

**Notes:** Also the model they constructed does not seem to be useful for us.

---

## 6. Exploring Deep Reinforcement Learning for Battling in Collectible Card Games
**Link:** https://homepages.dcc.ufmg.br/~ronaldo.vieira/assets/pdf/sbgames-2022.pdf

Recap: LOCM 1.2 battle phase as an MDP; PPO variant on a flat MLP, self-play, invalid action
masking over 145 actions (logits set to −∞). ~51% win rate vs. max-attack, ~36% vs.
one-step-lookahead — well below tree-search SOTA. The canonical "plain deep-RL baseline,"
honest that flat-MLP PPO alone doesn't reach search-based performance.

**Relevant/useful:**: Not really.

**Notes:**:  Not a serious state of the art paper, ignore.

---

## 7. Towards Sample-Efficient Deep Reinforcement Learning in Collectible Card Games
**Link:** https://www.sciencedirect.com/science/article/abs/pii/S1875952123000496

Recap: Follow-up to #6 on LOCM 1.5; PPO on an MLP, no attention used, but explicitly names
permutation-equivariant attention (Deep Sets, Set Transformers) as future work for card-order
invariance. Direct evidence that attention is the recognized-but-unexplored next step for TCG
state encoders.

**Relevant/useful:** Mixed

**Notes:**: Negative results, their method performs worse than the baseline.

---

## 8. Exploring Reinforcement Learning Approaches for Drafting in Collectible Card Games
**Link:** https://www.sciencedirect.com/science/article/abs/pii/S1875952122000490

Recap: Drafting/deck-building phase of LOCM as an MDP. Compares three self-play DRL
approaches differing in how they use pick-history (history-aware MLP, LSTM, history-oblivious
MLP). The reference treatment of deck construction as sequential RL — relevant only if deck
selection is ever learned rather than fixed.


**Relevant/useful:**: Not really.

**Notes:**:  Not a serious state of the art paper, ignore.

---

## 9. Learning With Generalised Card Representations for Magic: The Gathering
**Link:** https://arxiv.org/abs/2407.05879

Recap: Not RL — Contextual Preference Ranking with a Siamese network trained on human
draft-preference triplets. Builds generalised card representations (numeric, nominal, text,
image, meta-usage features) so the model handles unseen/newly-released cards; predicts ~55%
of human draft choices on completely unseen cards vs ~22% random. Most relevant to the
card-embedding problem — directly analogous to encoding a large, evolving Pokémon card pool.

**Relevant/useful:**  NO

**Notes:** Not RL, so skip. Okay, maaaybe we could look into their card representation approach but probably more useful to invest that time somewhere else.

---

## 10. A Closer Look at Invalid Action Masking in Policy Gradient Algorithms
**Link:** https://arxiv.org/abs/2006.14171

Recap: Theoretical + empirical reference for invalid action masking — masked logits still
form a valid distribution with a well-defined gradient, and the benefit grows as the
proportion of invalid actions increases. The citation to justify action masking in the
Pokémon TCG env. Just outside the 5-year window but foundational and universally cited.

**Relevant/useful:** Not really.

**Notes:** We are already doing proper action masking in the code to ensure we are performing legal actions. Nice to check but meh.

---

## 11. Summarizing Strategy Card Game AI Competition
**Link:** https://arxiv.org/abs/2305.11814

Recap: 5-year retrospective of the Strategy Card Game AI Competition (LOCM, through 2022).
Surveys game-tree search, neural networks, evaluation functions, and CCG deck-building
approaches. Best single entry point to the LOCM ecosystem — good orientation read, not a
technique in itself.

**Relevant/useful:** No

**Notes:** No real novel contributions. D-tier survey paper.

---

### P2-1. Transformers as Policies for Variable Action Environments
**Link:** https://arxiv.org/abs/2301.03679

Recap: Transformer encoder as the policy network with self-attention over game entities to
handle variable-size, entity-structured action spaces, trained with PPO. Not TCG-specific
(Griddly), but the most directly relevant "attention-for-variable-actions" architecture found.

**Relevant/useful:** Not really.

**Notes:** Does not really contain anything that is directly transferrable to the competition.

---

### P2-2. Towards Modern Card Games with Large-Scale Action Spaces Through Action Representation
**Link:** https://arxiv.org/pdf/2206.12700

Recap: Tackles large-scale action-space problems via learned action representations
(embedding actions rather than enumerating them) — relevant to a large/variable Pokémon TCG
option set.

**Relevant/useful:** No

**Notes:**: Not the same class of RL algorithm we are useful, same as above in the sense that it is not directly transferable.

---

### P2-3. Two-Step Reinforcement Learning for Multistage Strategy Card Game
**Link:** https://arxiv.org/html/2311.17305v1

Recap: Decomposes the multistage (draft + battle) LOCM problem into a two-step RL procedure.

**Relevant/useful:** No

**Notes:** Just no. Also the lord of the game card game is apparently cooperative game. We are dealing with a zero-sum game.

---

### P2-4. Policy-Based Reinforcement Learning Approach in Imperfect Information Card Game
**Link:** https://www.mdpi.com/2076-3417/15/4/2121

Recap: Policy-gradient RL under imperfect information — relevant to hidden-hand handling.

**Relevant/useful:** No

**Notes:** Simple game, not serious state of the art, we are not gaining anything from reading this.

---

### P2-5. Drafting in Collectible Card Games via Reinforcement Learning (2020, predecessor)
**Link:** https://www.sbgames.org/proceedings2020/ComputacaoFull/209690.pdf

Recap: Predecessor to #8 (just outside the 5-year window); establishes the drafting-as-MDP
framing.

**Relevant/useful:** Not really.

**Notes:** The way decks are constructed is not applicable to the Pokemon trading card game, no "drafting phase".

# Conclusions

We basically should read through in more detail all the ByteDance papers in this list. The rest are either not useful at all or not that relevant.
*  [Mastering SCG (LOCM) — End-to-End + OSFP (ByteRL)](https://arxiv.org/abs/2303.04096)
* [Mastering SCG (Hearthstone) — Improved Techniques](https://arxiv.org/pdf/2303.05197)

From a quick search I did not find any newer papers from them. Regarding the model architecture I think we can come up with something novel ourselves.

For the final policy we produce we have to be careful it generalizes properly and cannot get easily exploited.


# [Relevant Parts from the old literature review] RL for Two-Player Trading Card Games — Literature Review (2021–2026)

## TL;DR — the landscape

- **The dominant public benchmark is Legends of Code and Magic (LOCM)** — a deliberately
  small, open CCG built as an AI testbed, and the basis of the multi-year *Strategy Card
  Game AI Competition* (IEEE CEC/CoG). Most rigorous RL-for-TCG work you can actually
  reproduce is on LOCM. Pokémon TCG has **essentially no published RL literature** — you are
  working in a gap.
- **The recurring recipe is PPO + invalid action masking + self-play**, with the game framed
  as a (Partially Observable) MDP and a **flat MLP** as the network. This is the well-trodden baseline (Vieira et
  al.; the LOCM competition entries).
- **The strongest agents (ByteDance's "ByteRL")** go beyond naive self-play: an **end-to-end
  policy over both deck-building and battle**, trained with **Optimistic Smooth Fictitious
  Play** to approximate a Nash equilibrium. It won the COG2022 competition and beat a top-10
  Hearthstone streamer.
- **Attention/transformers are still emergent here, not established.**
- **Two structural challenges dominate every paper:** (1) **large + variable discrete action
  spaces** (handled via action masking and/or auto-regressive/entity-factored actions), and
  (2) **deck-building as a separate combinatorial problem** from in-game play.

---

## Synthesis: SOTA Mechanisms and Gaps

### Why ByteRL wins: two specific mechanisms, not "self-play in general"

**1. OSFP trains against a mixture of historical policies, not just the latest checkpoint.**
Vanilla fictitious play (and naive self-play, #6/#8) has a known failure mode: the sequence
of trained policies cycles around the Nash equilibrium instead of converging to it, because
each iteration only best-responds to the single most recent opponent — the policy chases
whatever it just lost to and forgets older counters. ByteRL's OSFP fixes this by mixing in an
*optimistic* prediction of the opponent's next move (the current payoff term is counted twice
in the update), which gives **last-iterate convergence** — the actual final policy converges,
not just a running average of past ones. The opponent pool itself is also curated, not
exhaustive: new checkpoints are added only when they clear a win-rate threshold **ξ = 0.7**
against the existing pool.
> *(paraphrased from the OSFP formulation and pool-update rule in the paper's Section on
> algorithm design)*

**2. The end-to-end network lets battle reward backpropagate into deck-building.**
ByteRL uses **one function approximator for both stages**: `π_θ(·|s) = δ·π_θ_CB(·|s) +
(1−δ)·π_θ_BT(·|s)`, where δ just indicates which stage you're in. The deck-card embeddings are
**held fixed during battle** but **updated recursively during drafting** — so the gradient
from a battle *outcome* flows back through those same embeddings and directly shapes which
cards get drafted. This is the concrete difference from the split-phase approach in #6-8:
there, drafting is optimized against a proxy (synergy/curve heuristics baked into the reward
or hand-crafted state), not against what the battle policy can actually convert into wins.

**Net:** ByteRL isn't strong because "self-play" is strong, It's strong because it (a) fixes self-play's
cycling/forgetting problem with a curated historical-opponent mixture, and (b) removes the
draft/battle objective mismatch by sharing gradients between them.

---

# Paper Summary

## Mastering Strategy Card Game (Legends of Code and Magic) via End-to-End Policy and Optimistic Smooth Fictitious Play  ("ByteRL")
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

## Mastering Strategy Card Game (Hearthstone) with Improved Techniques
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

