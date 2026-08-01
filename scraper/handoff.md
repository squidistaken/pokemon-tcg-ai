# Card-swap strategy handoff

## Current architecture

`CardSwapper` is the strategy interface. Implementations must provide
`resolve(RawCard)` and return ordered `CardSwap` candidates containing complete
provenance. The deck resolver applies the normal competition-card lookup first and
only calls the selected strategy for unresolved or ambiguous card lines.

Two implementations currently exist:

- `HeuristicCardSwapper` ranks compatible same-name competition printings from
  gameplay profiles.
- `MappingCardSwapper` is deliberately empty and therefore resolves nothing yet.

The Slurm scraper passes `--card-swap-strategy all`. A source deck is fetched once
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

## Mapping work still to do

The mapper needs a large, reviewable data source and must not be filled ad hoc in
Python. Before adding rules, decide and document a versioned schema that can key a
source by normalized name plus optional set and collection number, express ordered
competition Card ID candidates, and retain rationale and review metadata.

Implement `MappingCardSwapper.resolve()` against that schema and validate every
configured target against `CardIndex` at startup. Stale, ambiguous, non-competition,
or ordinary-card-to-ACE-SPEC targets must fail closed. Preserve the existing
deck-level copy-count and ACE SPEC guards and the manifest-v3 substitution fields.

Add table-driven tests covering accepted and unresolved inputs, Unicode names,
set/number-specific rules, ordered fallbacks, stale targets, copy caps, ACE SPEC
conflicts, exact-card precedence, and manifest provenance. Do not change
`scraper/EN_Card_Data.csv`.

## Experiment contract

The two folders are separate experimental datasets, not layers to merge. Compare
retained-deck yield, invalid-deck count, unresolved cards, substitution frequency,
and downstream train/eval results for the same fetch window. A mapping result must
never silently fall back to the heuristic result; otherwise the comparison no
longer isolates the strategy.
