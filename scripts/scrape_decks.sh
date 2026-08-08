#!/bin/bash
# Scrape the full deck archive from Limitless (+ Bulbapedia) into ./decks.
#
#   ./scripts/scrape_decks.sh             # complete archive, verbose, logged
#   ./scripts/scrape_decks.sh --dry-run   # resolve and validate only, write nothing
#
# Runs both card-swap strategies in one pass, writing isolated corpora to
# decks/mapping-resolved/ and decks/heuristic-resolved/ so the two can be
# compared without cross-contamination.
#
# This deliberately takes everything rather than the window that maximises deck
# yield. The engine card pool covers SVI through MEG/ASC but not CRI or PBL, so
# decks outside roughly 2026-01..2026-07 mostly drop -- measured resolve rates
# run 12/12 around 2026-03 and 0/12 by mid-2025. Those drops are the point: an
# unresolvable deck still names the cards that need mapping rules, which is what
# the next round of rules gets written from. Scraping narrow would throw that
# signal away.
#
# Two outputs carry the rejection signal:
#   decks/mapping-gaps-*.jsonl.gz   structured, deduplicated, with per-card
#                                   occurrence counts and example decks
#   logs/rejected-cards-*.txt       the ranked plain-text summary this script
#                                   derives from the verbose log
#
# SCRAPER_LIMIT is tournaments per page and the API does not cap it. The whole
# Limitless archive is 15708 tournaments, 12464 of them standard, so 15000 takes
# the standard archive in a single page and the walk ends on page two. Raising it
# further returns the same rows -- paging at 5000 sums to the identical 12464 --
# so the archive size, not the limit, is the binding constraint. SCRAPER_SINCE is
# left unset for the same reason. Corpus size comes free from
# SCRAPER_PER_TOURNAMENT instead: the source issues exactly one standings request
# per tournament no matter how many decks it takes from the response.
#
# Nothing here is capped. SCRAPER_FORMAT is empty so every format is taken, all
# 15708 tournaments rather than the 12464 standard ones; --max-decks is never
# passed, so decks are unlimited; SCRAPER_PER_TOURNAMENT takes every published
# list; SCRAPER_MAX_PAGES and BULBAPEDIA_MAX_PAGES walk until exhausted.
#
# The cost of taking every format is that CUSTOM, GLC, EXPANDED, BASEFOSSIL and
# BASENEO are largely pre-SV, and their unresolvable cards can never receive
# mapping rules -- they are not in the engine's pool and never will be. They
# still land in the rejection ranking, so read it knowing the mappable CRI/PBL
# entries sit among permanently unmappable ones. To narrow to the format whose
# rejections are all actionable:
#
#   SCRAPER_FORMAT=standard ./scripts/scrape_decks.sh
#
# Expect an overnight run. The floor is one request per tournament at the 1 req/s
# limiter (~3.5h); cold-cache card-profile lookups over the full archive push the
# realistic figure higher. Safe to interrupt -- the manifest is written on every
# new deck and the gap report every 100 decks.
#
# Every knob is an environment variable override:
#
#   SCRAPER_SINCE=2026-01-01 ./scripts/scrape_decks.sh   # narrow to the yield window
#   SCRAPER_OUT=decks-experiment ./scripts/scrape_decks.sh
#
# Exit codes: 0 clean, 1 finished but a source reported errors (expected --
# Bulbapedia's archetype pages predate the card pool), 2 refused to start.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCRAPER_OUT="${SCRAPER_OUT:-decks}"
SCRAPER_SOURCE="${SCRAPER_SOURCE:-all}"
SCRAPER_STRATEGY="${SCRAPER_STRATEGY:-all}"
MAPPING_MIN_CONFIDENCE="${MAPPING_MIN_CONFIDENCE:-1}"
SCRAPER_LIMIT="${SCRAPER_LIMIT:-15000}"
SCRAPER_FORMAT="${SCRAPER_FORMAT-}"
SCRAPER_MAX_PAGES="${SCRAPER_MAX_PAGES:-0}"
SCRAPER_PER_TOURNAMENT="${SCRAPER_PER_TOURNAMENT:-0}"
SCRAPER_SINCE="${SCRAPER_SINCE:-}"
BULBAPEDIA_CATEGORY="${BULBAPEDIA_CATEGORY:-Deck archetypes}"
BULBAPEDIA_MAX_PAGES="${BULBAPEDIA_MAX_PAGES:-0}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required to run the scraper." >&2
  exit 2
fi

# The corpus is a deduplicated whole: a scrape layered onto an older one silently
# mixes two card pools and two mapping-rule versions in a single manifest. Wipe
# first, keeping the committed example.csv that the tests and the submission
# default both depend on.
STALE="$(find "$SCRAPER_OUT" -mindepth 1 -not -name 'example.csv' -print -quit 2>/dev/null || true)"
if [ -n "$STALE" ]; then
  echo "'$SCRAPER_OUT' already holds deck data; scraping on top of it would append to the existing manifest."
  read -r -p "Delete it and start clean? [y/N] " response
  if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 2
  fi
  find "$SCRAPER_OUT" -mindepth 1 -not -name 'example.csv' -delete
  echo "Cleared '$SCRAPER_OUT'."
fi

mkdir -p "$SCRAPER_OUT" logs
STAMP="$(date +%Y%m%dT%H%M%S)"
LOG="logs/scrape-${STAMP}.log"
REJECTS="logs/rejected-cards-${STAMP}.txt"

# An unset SCRAPER_SINCE must not become `--since ""`. It is also the safer
# default for a full archive: the walk stops at the first tournament older than
# `since`, and the index carries a few entries with missing dates that read as
# 1970, so a date bound can truncate the walk early.
SINCE_ARGS=()
if [ -n "$SCRAPER_SINCE" ]; then
  SINCE_ARGS=(--since "$SCRAPER_SINCE")
fi

echo "Scraping the ${SCRAPER_FORMAT:-all-format} archive into '$SCRAPER_OUT'${SCRAPER_SINCE:+ since $SCRAPER_SINCE}; logging to ${LOG}."
echo "Expect an overnight run. Safe to interrupt: Ctrl-C leaves a usable corpus."

# The scraper exits 1 when any source errored even though the run completed and
# wrote decks, so the pipeline status is captured rather than allowed to trip
# set -e. tee keeps the run visible in tmux while still recording the log.
set +e
uv run --frozen --no-sync python -m scraper \
  --source "$SCRAPER_SOURCE" \
  --card-swap-strategy "$SCRAPER_STRATEGY" \
  --minimum-mapping-confidence "$MAPPING_MIN_CONFIDENCE" \
  --limit "$SCRAPER_LIMIT" \
  --format "$SCRAPER_FORMAT" \
  --max-pages "$SCRAPER_MAX_PAGES" \
  --per-tournament "$SCRAPER_PER_TOURNAMENT" \
  "${SINCE_ARGS[@]}" \
  --category "$BULBAPEDIA_CATEGORY" \
  --bulbapedia-max-pages "$BULBAPEDIA_MAX_PAGES" \
  --out "$SCRAPER_OUT" \
  --verbose \
  "$@" 2>&1 | tee "$LOG"
STATUS="${PIPESTATUS[0]}"
set -e

# Rank the cards that cost the most decks, so the next round of mapping rules can
# start with whatever appears at the top. Each deck is resolved once per strategy,
# so the counts are inflated by a constant factor and only the ranking is meaningful.
grep -h 'unresolved ->' "$LOG" 2>/dev/null \
  | sed 's/.*unresolved -> //' \
  | tr ';' '\n' \
  | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' \
  | grep -v '^$' \
  | sort | uniq -c | sort -rn >"$REJECTS" || true

DECKS="$(find "$SCRAPER_OUT" -name '*.csv' ! -name 'example.csv' | wc -l | tr -d ' ')"
echo
echo "Wrote ${DECKS} deck CSVs across all strategies."
echo "Full log:        ${LOG}"
echo "Rejected cards:  ${REJECTS} (ranked; feed the top entries into scraper/card_mappings/)"
echo "Structured gaps: $(find "$SCRAPER_OUT" -name 'mapping-gaps-*.jsonl.gz' -printf '%p\n' 2>/dev/null | tail -1)"

if [ -s "$REJECTS" ]; then
  echo
  echo "Top 15 cards blocking decks:"
  head -15 "$REJECTS"
fi

if [ "$STATUS" -eq 1 ]; then
  echo
  echo "Note: a source reported errors (grep the log for 'error'); the decks that were written are still valid."
fi
exit "$STATUS"
