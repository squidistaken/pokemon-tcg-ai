# Deck Scraper

An independent module that builds a corpus of legal 60-card decks for the
Pokémon TCG AI Battle Challenge. It scrapes decklists from the internet
(LimitlessTCG, Bulbapedia) or imports pasted text, maps every card to our
internal `Card ID` (from `EN_Card_Data.csv`), validates the deck against the
engine's rules, and saves each as a deck CSV in `decks/`.

It does **not** import from `cg/` or `main.py` and has no effect on the running
agent — it's a dev-time tool for generating decks.

## Install

```bash
uv sync        # pulls in requests, beautifulsoup4, lxml
```

## Usage

```bash
# Recent standard tournament decks from LimitlessTCG (public API, no key needed)
python -m scraper --source limitless --format standard --limit 20 --verbose

# Wiki decklists (best effort; older-set cards get dropped as unavailable)
python -m scraper --source bulbapedia --pages "Abyss (TCG),Aurora Blast (TCG)"

# Import a decklist you pasted into a file
python -m scraper --source text --input mylist.txt --name my-deck

# Run every source
python -m scraper --source all --limit 20

# See what would happen without writing anything
python -m scraper --source limitless --dry-run --verbose
```

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
- `decks/manifest.json` — metadata sidecar keyed by slug: `source`, `url`,
  `archetype`, `format`, `record`, `date`, and `id_hash` (used for dedup).

Decks are **deduplicated** by card multiset, and any deck that references a card
outside our card database (unavailable expansion) is **dropped** with a logged
reason — never silently truncated.

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
≥1 Basic Pokémon; ≤1 ACE SPEC card.

## Layout

| File | Role |
|------|------|
| `card_index.py` | Loads/indexes `EN_Card_Data.csv`; name→ID resolution, energy aliases, per-card flags |
| `resolver.py` | `RawDeck` → expanded list of Card IDs (reprint fallback) |
| `validator.py` | The 5 legality rules |
| `writer.py` | Writes deck CSV + manifest, dedups by multiset |
| `pipeline.py` | resolve → validate → write, with a run summary |
| `http.py` | Shared requests session (UA, retries, rate limit) |
| `sources/` | `limitless.py`, `bulbapedia.py`, `textimport.py` (+ `base.py`, registry) |
| `__main__.py` | CLI |

## Adding a source

Subclass `sources.base.DeckSource`, implement `iter_decks(**kwargs)` to yield
`RawDeck`s, and register it in `sources/__init__.py`'s `SOURCES` dict.

## Notes

- LimitlessTCG public tournament data needs no key. Set `LIMITLESS_API_KEY` to
  access private/organizer data (sent as the `X-Access-Key` header).
- Be polite: the scraper rate-limits and identifies itself via User-Agent.
```
