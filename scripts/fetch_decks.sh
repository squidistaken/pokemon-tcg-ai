#!/bin/bash
# Pull the standard training deck corpus from its GitHub Release.
#
#   ./scripts/fetch_decks.sh                  # newest release -> ./decks
#   ./scripts/fetch_decks.sh decks-v2         # pinned release -> ./decks
#   ./scripts/fetch_decks.sh --root /scratch/$USER/slopemon
#                                             # newest -> ROOT/decks
#
# The corpus is versioned as GitHub Releases tagged `decks-*`, each carrying a
# stable `decks.tar.gz` plus a `decks.sha256` checksum. Nothing about the version
# is committed to the repo, so publishing a new release never edits a tracked
# file -- no merge conflicts when people cut releases. Idempotent: a corpus
# already at the release's sha is left untouched.
#
# Requires the GitHub CLI (`gh auth login`).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_ROOT="$PROJECT_ROOT"
TAG=""

usage() {
  cat <<'EOF'
Usage: fetch_decks.sh [--root PATH] [decks-vN]

Download and verify a deck-corpus release. By default it installs into the
repository's decks/ directory. --root PATH installs into PATH/decks instead,
which is useful for high-capacity scratch storage.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --root needs a path" >&2
        exit 2
      fi
      INSTALL_ROOT="$2"
      shift 2
      ;;
    --root=*)
      INSTALL_ROOT="${1#--root=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "ERROR: unknown option '$1'" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [ -n "$TAG" ]; then
        echo "ERROR: only one release tag may be supplied" >&2
        exit 2
      fi
      TAG="$1"
      shift
      ;;
  esac
done

if [ -z "$INSTALL_ROOT" ]; then
  echo "ERROR: --root must not be empty" >&2
  exit 2
fi
mkdir -p "$INSTALL_ROOT"
INSTALL_ROOT="$(cd "$INSTALL_ROOT" && pwd)"
DECK_DIR="$INSTALL_ROOT/decks"

cd "$PROJECT_ROOT"

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: the GitHub CLI (gh) is required. Install it and run 'gh auth login'." >&2
  exit 1
fi

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

MARKER="$DECK_DIR/.release-sha256"
if [ -f "$MARKER" ] && [ "$(cat "$MARKER" 2>/dev/null)" = "$WANT_SHA" ]; then
  echo "Deck corpus at $DECK_DIR already has ${TAG} (${WANT_SHA:0:12}…); nothing to do."
  exit 0
fi

echo "Downloading corpus for ${TAG}…"
gh release download "$TAG" --pattern 'decks.tar.gz' --dir "$TMP"

echo "Verifying checksum…"
echo "${WANT_SHA}  ${TMP}/decks.tar.gz" | sha256sum -c -

if [ -d "$DECK_DIR" ] && [ "$(ls -A "$DECK_DIR")" ]; then
  read -p "Overwrite $DECK_DIR? (This will wipe all existing files except example.csv) [y/N] " response
  if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
  fi
  echo "Cleaning up old decks…"
  find "$DECK_DIR" -mindepth 1 -not -name 'example.csv' -delete
fi

echo "Unpacking into ${DECK_DIR}…"
tar -xzf "${TMP}/decks.tar.gz" -C "$INSTALL_ROOT"
echo "$WANT_SHA" >"$MARKER"

echo "Installed ${TAG}: $(find "$DECK_DIR" -name '*.csv' ! -name 'example.csv' | wc -l) decks in $DECK_DIR."
