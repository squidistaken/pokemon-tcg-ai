#!/bin/bash
# Publish one narrow deck corpus as an additive GitHub release.
#
#   ./scripts/publish_mini_corpus.sh top20 deck_v4_mini --notes "…"
#
# Unlike scripts/update_decks_release.sh, this packs a single corpus subtree and
# names its assets <corpus>.tar.gz / <corpus>.sha256 rather than decks.tar.gz.
# That is deliberate: fetch_decks.sh matches 'decks.sha256' and wipes decks/
# before unpacking, so a mini corpus published under the standard asset names
# could replace the full corpus. Training needs both present at once (the narrow
# one for env sampling, the full one for submission deck paths), so this release
# installs alongside instead.
#
# The tag also stays outside fetch_decks.sh's '^decks-v[0-9]+$' pattern, so a
# mini release never becomes the default corpus.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CORPUS=""
TAG=""
NOTES=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --notes) NOTES="${2:?--notes needs a value}"; shift 2 ;;
    -*) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    *) if [ -z "$CORPUS" ]; then CORPUS="$1"; else TAG="$1"; fi; shift ;;
  esac
done

if [ -z "$CORPUS" ] || [ -z "$TAG" ]; then
  echo "usage: $0 <corpus-dir-under-decks> <release-tag> --notes '…'" >&2
  exit 2
fi
if [ -z "$NOTES" ]; then
  echo "ERROR: --notes is required to describe this corpus." >&2
  exit 1
fi
if [[ "$TAG" =~ ^decks-v[0-9]+$ ]]; then
  echo "ERROR: tag '$TAG' matches fetch_decks.sh's default pattern, which would make" >&2
  echo "this narrow corpus the standard one. Pick a tag outside ^decks-v[0-9]+\$." >&2
  exit 1
fi
if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: the GitHub CLI (gh) is required. Install it and run 'gh auth login'." >&2
  exit 1
fi
if [ ! -f "decks/$CORPUS/manifest.json" ]; then
  echo "ERROR: decks/$CORPUS/manifest.json is missing; observation weighting needs it." >&2
  exit 1
fi
if gh release view "$TAG" >/dev/null 2>&1; then
  echo "ERROR: release $TAG already exists; delete it or pick another tag." >&2
  exit 1
fi

mkdir -p dist
FILELIST="$(mktemp)"
trap 'rm -f "$FILELIST"' EXIT

find "decks/$CORPUS" -type f \( -name '*.csv' -o -name 'manifest.json' \) \
  -not -path '*/__pycache__/*' | LC_ALL=C sort >"$FILELIST"
COUNT=$(grep -c '\.csv$' "$FILELIST" || true)
if [ "$COUNT" -eq 0 ]; then
  echo "ERROR: no deck CSVs found under decks/$CORPUS" >&2
  exit 1
fi

# Same normalisation as build_decks_release.sh, so the same corpus always hashes
# to the same bytes: sorted names, fixed mtime/owner, no gzip timestamp.
tar --sort=name --mtime='2020-01-01 00:00:00Z' \
  --owner=0 --group=0 --numeric-owner \
  --files-from "$FILELIST" -cf - | gzip -n >"dist/$CORPUS.tar.gz"
( cd dist && sha256sum "$CORPUS.tar.gz" >"$CORPUS.sha256" )
SHA="$(cut -d' ' -f1 <"dist/$CORPUS.sha256")"
ARCHETYPES=$(find "decks/$CORPUS" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')

NOTES="${NOTES}

*(Narrow corpus \`decks/${CORPUS}\`: ${COUNT} lists across ${ARCHETYPES} archetypes, sha256 ${SHA})*

Installs **alongside** the standard corpus, it does not replace it:

\`\`\`bash
gh release download ${TAG} --pattern '${CORPUS}.tar.gz' --dir /tmp
tar -xzf /tmp/${CORPUS}.tar.gz -C .
\`\`\`

\`scripts/fetch_decks.sh\` will not install this release: it matches asset
\`decks.sha256\` and wipes \`decks/\` first, and this tag is outside its
\`^decks-v[0-9]+\$\` version pattern."

TARGET="$(gh repo view --json defaultBranchRef --jq '.defaultBranchRef.name')"
gh release create "$TAG" "dist/$CORPUS.tar.gz" "dist/$CORPUS.sha256" \
  --title "$TAG" --notes "$NOTES" --target "$TARGET"

echo
echo "Published $TAG: $COUNT lists, $ARCHETYPES archetypes, sha256=$SHA"
