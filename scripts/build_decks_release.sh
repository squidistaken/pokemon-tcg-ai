#!/bin/bash
# Build the standard training deck corpus into a reproducible release artifact.
#
#   ./scripts/build_decks_release.sh    # -> dist/decks.tar.gz + dist/decks.sha256
#
# Packs decks/**/*.csv plus manifest.json (excluding the committed example.csv)
# with normalised metadata (sorted names, fixed mtime/owner, no gzip timestamp),
# so the same corpus always hashes to the same bytes. The asset names are stable
# across versions; the release TAG carries the version. Publish both files with
# scripts/update_decks_release.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p dist
FILELIST="$(mktemp)"
trap 'rm -f "$FILELIST"' EXIT

# The release carries the scraped corpus: every archetype CSV plus the manifest.
# The loose decks/example.csv (the committed default deck) stays in the repo.
find decks -type f \( -name '*.csv' -o -name 'manifest.json' \) \
  -not -path '*/__pycache__/*' -not -name 'example.csv' | LC_ALL=C sort >"$FILELIST"
count=$(wc -l <"$FILELIST")
if [ "$count" -eq 0 ]; then
  echo "ERROR: no corpus files found under decks/" >&2
  exit 1
fi

tar --sort=name --mtime='2020-01-01 00:00:00Z' \
  --owner=0 --group=0 --numeric-owner \
  --files-from "$FILELIST" -cf - | gzip -n >dist/decks.tar.gz

( cd dist && sha256sum decks.tar.gz >decks.sha256 )
SHA="$(cut -d' ' -f1 <dist/decks.sha256)"

echo "Packed ${count} files -> dist/decks.tar.gz"
echo "sha256=${SHA}"
