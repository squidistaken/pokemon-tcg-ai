# Deck Scraper

An independent module that builds a corpus of legal 60-card decks for the
Pokémon TCG AI Battle Challenge. It scrapes decklists from the internet
(LimitlessTCG, Bulbapedia) or imports pasted text, maps every card to our
internal `Card ID` (from `EN_Card_Data.csv`), validates the deck against the
engine's rules, and saves each as a deck CSV in `decks/`.

## Install

```bash
uv sync
```

## Usage

```bash
# Recent standard tournament decks from LimitlessTCG (public API, no key needed)
python -m scraper --source limitless --format standard --limit 20 --verbose

# Build a corpus at volume: walk the tournament index back through time.
# --limit is the page size, --max-pages how many pages (0 = until exhausted),
# --per-tournament 0 takes every published list instead of just the top 8.
python -m scraper --source limitless --limit 50 --max-pages 0 \
    --per-tournament 0 --since 2026-01-01 --max-decks 15000 --verbose

# Wiki decklists (best effort; older-set cards get dropped as unavailable)
python -m scraper --source bulbapedia --pages "Abyss (TCG),Aurora Blast (TCG)"

# Import a decklist you pasted into a file
python -m scraper --source text --input mylist.txt --name my-deck

# Run every network source concurrently (text still requires --source text --input)
python -m scraper --source all --limit 20 --bulbapedia-max-pages 20

# Fetch card research data only; no deck CSVs or strategy manifests are written
python -m scraper.discovery --source all --max-decks 15000 \
    --bulbapedia-max-pages 600

# Fetch once and write isolated mapping/heuristic corpora with separate manifests
python -m scraper --source all --card-swap-strategy all --out decks
```

### How much a run can find

Three things bound the yield, and they multiply:

| Bound | Flag | Note |
|---|---|---|
| Tournaments per page | `--limit` | The API does not cap this; 500 works. |
| Pages walked | `--max-pages` | Newest-first; paging reaches back years. `0` = until exhausted. |
| Decks per tournament | `--per-tournament` | Best finish first. `0` = every published list. |

Bulbapedia has its own `--bulbapedia-max-pages` bound. Set it to `0` to follow
MediaWiki continuation until the configured category is exhausted.

Every player who publishes a list is available — a 220-player event exposes 220
decklists — so the default `--per-tournament 8` takes about **10%** of what's there.
Raising it trades corpus size against quality: the tail of a large event is 1-4 and
0-6 finishes, much weaker training signal than the top tables.

The real ceiling, though, is the **card pool**. `EN_Card_Data.csv` holds 19 sets, and
a deck referencing anything outside them is dropped whole (never truncated). Current
Standard leans on `CRI` and `PBL`, which aren't in the pool, so recent tournaments
resolve at roughly **0-33%**; walking further back instead hits `PAR`, `MEW` and
`OBF`, which are also absent. The pool is a subset with gaps rather than a recency
window, so no date range avoids this — expect to fetch ~3x the decks you keep, and
bound runs with `--max-decks` rather than assuming a page count. Requests are
rate-limited to 1/s, so a deep walk is measured in hours.

Bulbapedia hits that ceiling head-on and resolves at **0%**: its `Deck archetypes`
category is a historical archive rather than a current-meta feed. A full walk of the
category fetched 111 lists and dropped all 111, blocked by cards from Base Set, Neo
Genesis, Great Encounters, Legends Awakened, Boundaries Crossed, Phantom Forces and
similar pre-Scarlet & Violet sets. Unlike the `CRI`/`PBL` drops on the Limitless side,
these are not a mapping gap to close — a rule can only redirect a name onto a card that
exists in the pool, and Base Set `Professor Oak` has no `SVI`-onward counterpart. So
Bulbapedia contributes no decks to the corpus and only permanent noise to the rejection
ranking; use `--source limitless` when that noise is in the way.

Text-import format (one card per line; set/number optional):

```
Pokémon: 6
2 Kyogre MEG 45
4 Snover MEG 24
Trainer: 19
4 Lillie's Determination (MEG-119)
Energy: 35
35 Water Energy
```

## Output

- `decks/<slug>.csv` — a bare **60-line list of Card IDs**, one per line, no
  header. This matches `main.read_deck_csv()` and the engine loader exactly, so
  a generated deck can be dropped in as `deck.csv` and submitted unchanged.
- `decks/manifest.json` — the corpus's provenance record (see below).
- `decks/mapping-gaps-<run-id>.jsonl.gz` — a fresh, run-specific report of source
  printings not covered by the reviewed mapper. Production never appends to a
  previous gap inventory.

Decks are **deduplicated** by card multiset, and any deck that references a card
outside our card database (unavailable expansion) is **dropped** with a logged
reason — never silently truncated.

## The manifest (`manifest.py`)

`manifest.json` has one entry per deck file, keyed by slug, and retains **every
occurrence** of that deck — each time a player brought the same 60-card multiset to
an event. `observation_count` is a popularity signal: it separates a one-off brew
from a list fourteen players independently piloted across five tournaments. (Schema
v1 kept only the first occurrence and silently discarded the rest, so the two were
indistinguishable.)

```json
{
  "schema_version": 2,
  "hash_algo": "sha1-sorted-card-ids-12",
  "decks": {
    "gardevoir-ex": {
      "file": "gardevoir-ex/gardevoir-ex.csv",
      "id_hash": "0f3a91c2bb47",
      "archetype": "Gardevoir ex",
      "format": "standard",
      "observation_count": 2,
      "first_seen": "2026-05-10",
      "last_seen": "2026-06-14",
      "warnings": [],
      "observations": [
        {
          "source": "limitless",
          "archetype": "Gardevoir ex",
          "format": "standard",
          "event": "Chicago Regional",
          "event_date": "2026-05-10",
          "scraped_date": "2026-05-11",
          "record": "8-1-0",
          "placing": 3,
          "url": "https://play.limitlesstcg.com/tournament/abc/standings",
          "external_ids": {"tournament_id": "abc", "placing": "3"},
          "merged_from": null
        }
      ]
    }
  }
}
```

Per-entry rules:

- `observation_count` is always `len(observations)`, recomputed on save, so the
  stored count cannot drift from the list it summarises.
- `first_seen` / `last_seen` span the observations by **event** date (falling back
  to the scrape date when a source doesn't report one).
- `archetype` / `format` at deck level are the label the file was *created* under —
  the slug derives from it, so it can't change. Per-occurrence values live in each
  observation, since sources name the same archetype differently.
- `merged_from` is set when `--prune` folded a near-duplicate's occurrence in here;
  it names the deck it was originally observed with.

**Re-scrapes are idempotent.** Every occurrence has a stable identity built from the
source's own IDs — `(tournament_id, placing)` for Limitless, page + table index for
Bulbapedia, the input path for text imports — so re-running the same scrape
re-observes what's already on file and changes nothing. The scrape date is
deliberately *not* part of that identity; if it were, the same standing would look
new every calendar day and the count would just track how often the scraper ran.

A run reports the three outcomes separately: `written (new)` for a new deck file,
`re-observed` for a new occurrence of a deck already in the corpus, and
`already recorded` for an occurrence that was already on file.

The manifest is **accumulated state a fresh scrape cannot reconstruct**, so it is
written atomically (temp file + `fsync` + rename), and a manifest that fails to
parse is a **fatal error** rather than being silently replaced with an empty one.
Every deck CSV re-hashes to its `id_hash`, so a damaged manifest can be rebuilt
from the corpus — overwriting it cannot be undone.

### The hash contract

`id_hash` (`writer.deck_hash`) is the dedup key, and it is **card-order agnostic**
but **card-copy-count sensitive**:

- reorderings of the same multiset hash alike, so the same list scraped from two
  sources that group cards differently collapses onto one entry;
- a different number of copies of any card is a different deck — `4/2/1` and `3/3/1`
  splits of the same three cards do *not* collide (a `set`/`frozenset` would).

It is a sorted, comma-joined digest of the integer Card IDs, truncated to 12 hex
chars. Empty decks and string IDs are rejected rather than hashed: string IDs sort
lexicographically (`"10" < "9"`), which would give one deck two different hashes.
Because 12 hex chars is 48 bits, a hash match is confirmed against the deck CSV on
disk before two decks are treated as one, so a collision can't silently fuse two
decks' provenance. `tests/test_scraper_manifest.py` pins all of this down, including
the save → reload → re-hash round trip against the manifest.

## Card resolution (`card_index.py`)

Scraped names are matched to our `Card ID`s (the competition's legal card pool)
in this order:

1. **Energy aliases** — "Water Energy" / "Basic Water Energy" → `Basic {W} Energy`.
2. **Exact** normalized match. Normalization folds accents, unifies apostrophes,
   and splits glued markers so `LugiaEX` (our CSV) == `Lugia ex` (sources).
3. **Guarded fuzzy fallback** for name variants/truncations in our CSV — e.g.
   `Telepathic Psychic Energy` → `Telepath Psychic Energy`,
   `Rocky Fighting Energy` → `Rock Fighting Energy`. Three guards prevent wrong
   matches:
   - the scraped **set must be one we have** (a card from a set we lack is
     genuinely absent, not a typo);
   - **power markers must be identical** — `ex`/`mega`/`v`/… — so `Greninja ex`
     never collapses into `Greninja` and `Mega Greninja ex` never into `Greninja ex`;
   - **similarity ≥ 0.90**.

   Every fuzzy substitution is logged (`fuzzy: 'X' -> 'Y' (score)`) so it's auditable.

Collection numbers are **not** trusted for cross-source identity — Limitless and
our CSV disagree on POR numbering (e.g. POR 87/88 are swapped) — so name is the
primary key. Cards that aren't in our CSV at all (`Poké Ball`, anything in the
newer `CRI` set, etc.) are outside the competition's card pool and their decks
are dropped.

## Deck legality (enforced in `validator.py`)

Mirrors the native engine (`ptcg_engine/.../Api.h`): exactly 60 cards; every ID
must exist in `EN_Card_Data.csv`; ≤4 copies per card name (Basic Energy exempt);
≥1 Basic Pokémon; ≤1 ACE SPEC card. `validate_deck()` returns a
`ValidationResult` — decks with any `errors` are dropped.

Optionally (`--warn-impossible-evolutions`, or `warn_impossible_evolutions=True`),
it also flags evolutions with no copy of their immediate previous stage in the
deck (e.g. a Vaporeon with no Eevee) — legal per the engine, since evolution
lines aren't enforced, but that copy can never evolve. These are non-fatal
`warnings`: they're logged and recorded in the manifest's `warnings` field, not
dropped.

## Layout

| File | Role |
|------|------|
| `card_index.py` | Loads/indexes `EN_Card_Data.csv`; name→ID resolution, energy aliases, per-card flags |
| `resolver.py` | `RawDeck` → expanded list of Card IDs (reprint fallback) |
| `validator.py` | The 5 legality rules |
| `manifest.py` | Manifest schema (deck entries + observations), occurrence identity, atomic I/O |
| `writer.py` | Writes deck CSVs, dedups by multiset (`deck_hash`), records occurrences |
| `pipeline.py` | resolve → validate → record, with a run summary |
| `http.py` | Shared requests session (UA, retries, rate limit) |
| `sources/` | `limitless.py`, `bulbapedia.py`, `textimport.py` (+ `base.py`, registry) |
| `__main__.py` | CLI |
| `analysis/` | Corpus metrics (similarity/diversity/metagame), plots, and near-dup pruning |

## Analysing the corpus (`analysis/`)

Once decks are scraped, `python -m scraper.analysis` reads a deck directory and
reports pairwise similarity (set / weighted Jaccard / card-semantic), clustering
agreement against the manifest's archetype labels, corpus diversity, deck
structure, and a metagame summary, then writes a plot suite and a captured
report to `outputs/deck_analysis/`.

```bash
# Full report over decks/ (writes plots + a text report to outputs/deck_analysis/)
python -m scraper.analysis

# Collapse near-duplicate lists (weighted-Jaccard >= threshold) to one each.
# This DELETES the redundant deck files (and their manifest entries), but first
# folds their observations into the surviving list, tagged `merged_from`, so the
# popularity signal isn't destroyed along with the files.
python -m scraper.analysis --prune --dupe-threshold 0.9
```

The metric functions are also importable (`from scraper.analysis import
weighted_jaccard_matrix, ...`). The analysis needs the engine's `CardDatabase`
(`src.env`); if it can't load, the card-semantic/structure/metagame sections are
skipped and the Jaccard/diversity sections still run.

## Adding a source

Subclass `sources.base.DeckSource`, implement `iter_decks(**kwargs)` to yield
`RawDeck`s, and register it in `sources/__init__.py`'s `SOURCES` dict.

## Notes

- LimitlessTCG public tournament data needs no key. Set `LIMITLESS_API_KEY` to
  access private/organizer data (sent as the `X-Access-Key` header).
- Be polite: the scraper rate-limits and identifies itself via User-Agent.
```
