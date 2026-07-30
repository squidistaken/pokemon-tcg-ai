#!/bin/bash
# Pull the standard training deck corpus from its GitHub Release into ./decks.
#
#   ./scripts/fetch_decks.sh            # newest decks-* release (the standard)
#   ./scripts/fetch_decks.sh decks-v2   # a specific version, for reproducibility
#
# The corpus is versioned as GitHub Releases tagged `decks-*`, each carrying a
# stable `decks.tar.gz` plus a `decks.sha256` checksum. Nothing about the version
# is committed to the repo, so publishing a new release never edits a tracked
# file -- no merge conflicts when people cut releases. Idempotent: a corpus
# already at the release's sha is left untouched.
#
# Requires the GitHub CLI (`gh auth login`).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: the GitHub CLI (gh) is required. Install it and run 'gh auth login'." >&2
  exit 1
fi

TAG="${1:-}"
if [ -z "$TAG" ]; then
  # Highest-numbered decks-vN release (ignores any code releases). Ordering by
  # the version number is immune to a release being edited/re-dated later.
  TAG="$(gh release list --limit 100 --json tagName \
    --jq '[.[] | select(.tagName | test("^decks-v[0-9]+$"))] | max_by(.tagName | ltrimstr("decks-v") | tonumber) | .tagName // empty')"
  if [ -z "$TAG" ]; then
    echo "ERROR: no decks-* release found. Publish one: ./scripts/update_decks_release.sh v1" >&2
    exit 1
  fi
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# The checksum file is tiny; fetch it first to decide whether downloading the
# full tarball is even necessary.
gh release download "$TAG" --pattern 'decks.sha256' --dir "$TMP"
WANT_SHA="$(cut -d' ' -f1 <"$TMP/decks.sha256")"

MARKER="decks/.release-sha256"
if [ -f "$MARKER" ] && [ "$(cat "$MARKER" 2>/dev/null)" = "$WANT_SHA" ]; then
  echo "Deck corpus already at ${TAG} (${WANT_SHA:0:12}…); nothing to do."
  exit 0
fi

echo "Downloading corpus for ${TAG}…"
gh release download "$TAG" --pattern 'decks.tar.gz' --dir "$TMP"

echo "Verifying checksum…"
echo "${WANT_SHA}  ${TMP}/decks.tar.gz" | sha256sum -c -

echo "Unpacking into ./decks…"
tar -xzf "${TMP}/decks.tar.gz" -C "$ROOT"
echo "$WANT_SHA" >"$MARKER"

echo "Installed ${TAG}: $(find decks -name '*.csv' ! -name 'example.csv' | wc -l) decks."
