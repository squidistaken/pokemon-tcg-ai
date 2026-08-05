# Card swapper

`CardSwapper` defines the strategy interface for resolving an incoming printing
to a competition Card ID without changing `EN_Card_Data.csv`.
`HeuristicCardSwapper` implements same-name gameplay-profile matching.
`MappingCardSwapper` independently loads versioned rules from
`card_mappings/`. Neither strategy falls back to the other. If the selected
strategy has no defensible resolution, the card remains unresolved and the deck
is dropped.

`--minimum-mapping-confidence` controls the lowest eligible target tier. The
production Slurm job sets it to `1`, making all stored targets eligible before the
normal safety guards run.

The optional `rejected_by_review_agent.json` fragment is excluded by default and
has the same compact rule schema. `--use-rejected-mappings` includes it for an
explicit comparison run; production does not pass that flag.

## Resolution order

1. Exact competition set and collection number.
2. A unique competition printing with the same normalized name.
3. The selected strategy: either the closest compatible same-name gameplay
   profile or an approved mapping rule.
4. Unresolved.

The third step is used whenever ordinary lookup remains unresolved or ambiguous.
When source metadata is needed, the printing is fetched from Limitless by set and
collection number. Its HTML is cached under ignored `outputs/card_swap_cache/`; it
is never written to a deck file or added to the training data.

## Gameplay profiles

Competition profiles are aggregated from every CSV row for a Card ID. The source
and competition profiles contain:

- card subtype and rule marker;
- evolution stage and previous stage;
- HP, type, weakness, resistance, and retreat cost;
- attacks, abilities, energy costs, damage, and effect text.

Candidates must have the same normalized name, subtype, rule marker, type, and
evolution relationship. Compatible candidates are ranked by the remaining fields.
The selected candidate must pass its profile-category minimum; equally scored legal
candidates are ordered by Card ID so the result is deterministic. There are no
card-name-specific conditions or target Card IDs in the implementation.

### Similarity configuration

`SimilarityConfig` names every scoring weight and acceptance threshold and can be
passed to `HeuristicCardSwapper`. Text similarity is 60% token overlap and 40%
character sequence similarity by default:

```python
HeuristicCardSwapper(index, config=SimilarityConfig(text_sequence_weight=0.40))
```

Token overlap is primary because it tolerates harmless word-order and formatting
differences between Limitless and the CSV. Sequence similarity still contributes so
effects using similar vocabulary in a different instruction structure do not look
identical. For sequence weights `0.0 / 0.4 / 1.0`:

| Text pair | Token only | Default blend | Sequence only |
| --- | ---: | ---: | ---: |
| Identical search effect | 1.000 | 1.000 | 1.000 |
| Same search action, reordered wording | 0.800 | 0.693 | 0.532 |
| Energy movement using a different source zone | 0.750 | 0.715 | 0.661 |

The last pair shows why text similarity is not sufficient alone: its default score
is still below the `0.90` non-Pokémon acceptance threshold. Pokémon use a lower
overall threshold because their final score also includes hard subtype, rule, type,
and evolution gates plus HP, retreat, attacks, costs, damage, weakness, and
resistance. They must also reach the generic `0.40` move-similarity minimum, which
prevents matching cards that share species and stats but perform unrelated actions.
For example, the live PBL 46 Drilbur scored `0.5868` overall against competition ID
81 but only `0.3885` for its moves: searching for Basic Pokémon is not close enough
to discarding Fighting Energy. OBF 27 Charmeleon reaches `0.4580` for its moves
against ASC 21 and remains accepted. These defaults are explicit heuristics rather
than trained values; configuration makes controlled calibration possible without
card-specific rules.

OBF 27 Charmeleon is a regression fixture, not a special case in the code. Its
profile selects ASC 21 Charmeleon (`927`) from the competition variants. Missing
cards with no competition printing under the same name remain unresolved.

## Deck safety and provenance

The resolver preserves each source line's copy count and rejects an assignment that
would exceed the four-copy or ACE SPEC limits. The normal validator then checks the
complete post-resolution deck before it can be written.

Every selected variant is stored on its manifest observation under `substitutions`,
including source set/number, copy count, target ID/name, and either heuristic
similarity or the explicit mapper's 1–5 confidence. Different observations of the
same final deck therefore keep their own source-printing provenance.

Team Rocket's Energy is already an exact competition card, so exact-card precedence
resolves it directly and records no substitution.

## Mapping rules

Mapping rules are JSON fragments loaded in stable path order. Only exact
name/set/number rules are supported. Every expected target name and Card ID is
validated against the competition CSV, and ordered fallback lists are preserved.

Cross-species Pokémon mappings require an evolution-family identifier and the same
stage. The deck-level assignment backtracks until existing source evolution links
remain coherent after mapping. Cross-subtype Energy mappings require an explicit
rule flag. Ordinary cards cannot target ACE SPEC cards, and the normal copy-count
and ACE SPEC guards still run after candidate selection.

The 6-hour discovery job provides the research input in
`outputs/card_discovery/seen_cards.jsonl.gz`. Only mappings accepted by the
proposer/reviewer research pass are stored in the tracked production mapping.
