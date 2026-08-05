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

## Shard 13–16 rerun

Shards 1–12 were originally assigned to separate proposer identities. Shards
13–16 were the first four-shard proposer batch and produced only five proposals
from 251 records; all five were accepted by independent reviewers. Later batched
shards 17–24 recovered to 45 proposals with 37 accepted, so batching alone was not
the problem, but the isolated 13–16 trough indicated unusually low recall.

For that reason only shards 13–16 were rerun with one fresh proposer per shard and
fresh independent reviewers. The rerun used the same locked inputs,
competition catalog, confidence definitions, and safety constraints. Original
artifacts remain unchanged, and only newly accepted exact-printing rules that do
not duplicate an existing source identity or rule ID may be merged.

The rerun produced 66 proposals: 58 were accepted and eight were rejected. Five
accepted rules rediscovered existing mappings with the same targets, so the merge
added 53 accepted rules rather than duplicating them. Atomic-family validation
reduced the rejected output to seven stored rules. The resulting production files
contain 133 default rules and 51 opt-in rejected rules, with no duplicate source
identities or rule IDs.
