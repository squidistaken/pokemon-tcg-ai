# StructuredObsAdapter

How `src/models/structured_obs_adapter.py` turns the structured observation into
backbone input. The class docstring carries the summary; this is the detail.

## Feature construction, shared by both paths

Three kinds of input, handled differently:

```
card IDs (context_card_ids, stadium_id, every card in a zone)
    ID -> nn.Embedding(8, padding_idx=0) + static features(12)
       + card_categories(4 fields, Embed(4) each) + pooled attack_repr(22) -> [58]
    [58] -> Linear(58 -> entity_dim) -> [entity_dim]

table rows (options, pokemon)
    concat all row fields -> [87] or [101] -> Linear(-> entity_dim) -> [entity_dim]
    (one token per row; padding rows included, filtered later)

scalars (globals, select_cats)
    globals [41] / fixed scales, select_cats [2] -> Embed(4) -> [8]
```

## Token path: `encode_entity_tokens`, one token per entity

Requested groups return their individual entities instead of a pooled summary,
taken from the same per-entity encoders as the pooled path. This costs no extra
parameters and needs no separate construction. `pool` stays True; it selects the
flat encoding, not this.

Padding is not filtered out, because the slot count has to stay fixed across a
batch. Each group returns its full padded table plus a boolean validity mask.
For options the mask is "the encoder wrote an option type into the slot"
(`cats[..., 0] != 0`) plus the always-present stop slot, see `_option_validity`;
pokemon and zones carry their own masks. Consumers pass the mask to attention as
`src_key_padding_mask`.

Every group with an entity axis also has a per-slot segment id
(`group_segment_ids`). It distinguishes identity that the per-entity encoders
erase: which seat a Pokemon belongs to, which zone a card sits in, whether an
option slot is real or the synthetic stop action. A card in `my.hand` and the
same card in `my.discard` encode to the same vector without it, since they share
an ID, static features and projection. A backbone that expands a group into
per-entity tokens adds a learned embedding indexed by this id to restore the
identity.

```
group             slots (max_options=128)   typical real   dims
context_card_ids                        2              2   [2, 64]
stadium_id                              1              1   [1, 64]
options                   max_options + 1 = 129        4*   [129, 64]
pokemon                  2 x (1 + bench_cap) = 18       4   [18, 64]
my           hand 60 + discard 60 + prize 6 = 126      13   [126, 64]
opp                    discard 60 + prize 6 = 66       10   [66, 64]
select_deck                    deck_cap = 60            3   [60, 64]
looking                     looking_cap = 60            1   [60, 64]
                                      462             ~38

* 3 real options plus the always-valid synthetic stop slot.
```

Note the gap between padded and real counts. Attention is quadratic in the
padded count, so groups are requested individually rather than all at once.
`globals` and `select_cats` have no entity axis and are rejected: they are
already one token each through `encode_groups`.

Face-down cards (prizes) have ID 0. The learned embedding and static features
are zero there, leaving only the fixed "absent category" vector from
`_card_repr`'s categorical block, which is the same constant for every unknown
card, so no identity leaks. The slot is still present and valid: the model knows
how many prizes remain even though it cannot see them.

## MLP path (`pool=True`), one flat vector

Same per-entity encoding. Each group is then collapsed to a single fixed vector
by masked set pooling, which skips padding. Zones also get a fill fraction
(cards present / zone capacity). Under the default `zone_pooling="mean_max_sum"`
each set contributes `3 x 64` (mean, element-wise max, capacity-normalized sum)
instead of the mean alone: a centroid cannot say whether a specific card is
present, only what the average card looks like, which is not enough to read a
hand or a board. The `options` group keeps its plain mean, because it is only
context here and the per-slot detail a pointer head needs lives in the token
path.

```
group              dims (mean_max_sum)   (legacy mean)
globals                    41                  41
select_cats                 8                   8
context_card_ids          128                 128
stadium_id                 64                  64
options                    64                  64   (pooled digest only)
pokemon                   192                  64
my (3 zones)              579                 195
opp (2 zones)             386                 130
select_deck (1)           193                  65
looking (1)               193                  65
             torch.cat -> 1848                 824   ->  MLP
```

## Seat split (`pokemon_seat_split=True`)

`pokemon` is the one group holding both players' rows, so the single pool above
is seat-blind: swapping the two boards leaves the group vector, and every
backbone's `state_repr` built from it, bit-identical. `pokemon_seat_split` pools
the two halves of the row axis separately and concatenates them (192 -> 384 in
the table, total 1848 -> 2040). That is the only way the pooled path can tell
the seats apart. The token path already can, through `group_segment_ids`.

## Option tokens (`emit_option_tokens=True`)

The pooled `options` entry is permutation-invariant, so on its own it tells the
policy nothing about which action sits in which slot. With `emit_option_tokens`,
`forward` also returns the per-slot table for a pointer head to score against:
the same rows `encode_entity_tokens` yields, handed up unprojected for a trunk
that builds no tokens of its own. See
[`pointer-head.md`](pointer-head.md) for why that matters.
