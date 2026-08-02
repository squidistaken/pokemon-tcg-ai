# Reviewed card mappings

JSON fragments in this directory are loaded in stable path order by
`MappingCardSwapper`. A rule becomes active only after its `review.status` is
`approved` and it names the human reviewer. Targets are competition Card IDs and
are validated against `EN_Card_Data.csv` at startup.

The discovery job writes `outputs/card_discovery/seen_cards.jsonl.gz`. That
inventory is the input for proposer/reviewer agent waves; accepted rules are added
here only after Stef or Teun signs off. Mapping never falls back to the heuristic
swapper, and uncertain printings remain unresolved.

Each schema-v1 rule contains `rule_id`, source identity and gameplay subtype,
ordered `{card_id, expected_name}` targets, `active`, `review`, and `rationale`.
Exact rules provide both `source_set` and `source_number`; omitting both creates a
name fallback. Cross-species Pokémon rules also provide `family_id` and evolution
metadata. Cross-subtype Energy rules must set `allow_cross_subtype` explicitly.
