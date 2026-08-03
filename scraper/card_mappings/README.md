# Card mappings

JSON fragments in this directory are loaded in stable path order by
`MappingCardSwapper`. Every rule present is eligible for use. Targets are
competition Card IDs and are validated against `EN_Card_Data.csv` at startup.

The discovery job writes `outputs/card_discovery/seen_cards.jsonl.gz`. That
inventory is the input for proposer/reviewer agent waves; accepted rules are stored
here. Mapping never falls back to the heuristic swapper, and uncertain printings
remain unresolved.

Each schema-v2 rule contains `rule_id`, exact source name/set/number, gameplay
subtype, and ordered targets. Every target contains a competition
`card_id`, `expected_name`, integer `mapping_confidence` from 1 through 5, and its
own rationale. Name-level fallbacks are not supported, so unseen printings fail
closed. Cross-species Pokémon rules also provide `family_id` and evolution metadata.
Cross-subtype Energy rules must set `allow_cross_subtype` explicitly. `Tool` and
`Pokémon Tool` are treated as the same subtype.

`--minimum-mapping-confidence` selects the lowest eligible target tier. Production
sets it to `1`, so all stored targets are eligible; the normal competition and
deck-level guards remain authoritative.

`rejected_by_review_agent.json` is retained for comparison runs but skipped by
default. `--use-rejected-mappings` opts into that entire file. Production does not
pass the flag.
