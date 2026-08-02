# Card-swap strategy handoff

## Current architecture

`CardSwapper` is the strategy interface. Implementations must provide
`resolve(RawCard)` and return ordered `CardSwap` candidates containing complete
provenance. The deck resolver applies the normal competition-card lookup first and
only calls the selected strategy for unresolved or ambiguous card lines.

Two implementations currently exist:

- `HeuristicCardSwapper` ranks compatible same-name competition printings from
  gameplay profiles.
- `MappingCardSwapper` loads reviewed versioned JSON fragments. The tracked rule
  set is deliberately empty until discovery and human review are complete.

The production Slurm scraper passes `--card-swap-strategy all`. A source deck is fetched once
and independently processed into:

- `decks/mapping-resolved/`, with its own `manifest.json`;
- `decks/heuristic-resolved/`, with its own `manifest.json`.

The empty mapping strategy still retains decks that resolve normally without a
fallback. It must never copy results from the heuristic strategy.

Training and evaluation use the same top-level Hydra choice:

```bash
deck_corpus=heuristic-resolved  # default
deck_corpus=mapping-resolved
```

`conf/env/multideck.yaml` resolves this under `paths.data_dir`, so a Slurm scratch
override such as `paths.data_dir=/scratch/$USER/pokemon-tcg-ai/decks` continues to
work for either corpus.

## Discovery and mapping work still to do

Run `slurm-conf/discover_cards.sh` first. Its 12-hour CPU job concurrently scans up
to 5,000 Limitless decks and 200 Bulbapedia pages, writing only the resumable
`seen_cards.jsonl.gz` inventory. Download that file before starting mapping work.

Shard the deterministic inventory across proposer agents, then have independent
reviewer agents challenge the proposed targets and unresolved dispositions. Add
only Stef- or Teun-approved rules under `card_mappings/`. The loader already fails
closed on stale targets, unapproved active rules, subtype violations, invalid
evolution-family rules, and ordinary-card-to-ACE-SPEC mappings.

Add table-driven resolution cases for approved and intentionally unresolved inputs.
Do not change `scraper/EN_Card_Data.csv`.

## Experiment contract

The two folders are separate experimental datasets, not layers to merge. Compare
retained-deck yield, invalid-deck count, unresolved cards, substitution frequency,
and downstream train/eval results for the same fetch window. A mapping result must
never silently fall back to the heuristic result; otherwise the comparison no
longer isolates the strategy.

After approval, `slurm-conf/scrape_all.sh` refetches the full Limitless date window
and complete Bulbapedia category. It writes both corpora plus
`decks/mapping-gaps.jsonl.gz` for the next review cycle.
