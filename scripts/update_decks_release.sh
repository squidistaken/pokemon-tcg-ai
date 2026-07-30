#!/bin/bash
# Publish a new standard deck corpus release. Auto-increments the version.
#
#   ./scripts/update_decks_release.sh                 # next decks-vN automatically
#   ./scripts/update_decks_release.sh v5              # force a specific version
#   ./scripts/update_decks_release.sh --notes "…"     # with custom release notes
#
# Builds a reproducible tarball from the current decks/ and creates a GitHub
# Release tagged decks-vN with decks.tar.gz + decks.sha256 attached. Touches NO
# tracked files, so cutting releases never causes merge conflicts; fetch_decks.sh
# picks up the highest-numbered release automatically. Requires gh (authenticated).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: the GitHub CLI (gh) is required. Install it and run 'gh auth login'." >&2
  exit 1
fi

VERSION=""
NOTES=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --notes) NOTES="${2:?--notes needs a value}"; shift 2 ;;
    -*) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    *) VERSION="$1"; shift ;;
  esac
done

# No version given: auto-increment past the highest existing decks-vN.
if [ -z "$VERSION" ]; then
  last="$(gh release list --limit 100 --json tagName \
    --jq '[.[] | select(.tagName | test("^decks-v[0-9]+$")) | (.tagName | ltrimstr("decks-v") | tonumber)] | max // 0')"
  VERSION="v$(( last + 1 ))"
elif [[ "$VERSION" =~ ^[0-9]+$ ]]; then
  VERSION="v${VERSION}"
fi
TAG="decks-${VERSION}"

if gh release view "$TAG" >/dev/null 2>&1; then
  echo "ERROR: release ${TAG} already exists; pick another version." >&2
  exit 1
fi

echo "Building corpus artifact…"
bash scripts/build_decks_release.sh >/dev/null
SHA="$(cut -d' ' -f1 <dist/decks.sha256)"

if [ -z "$NOTES" ]; then
  count="$(find decks -name '*.csv' ! -name 'example.csv' | wc -l | tr -d ' ')"
  NOTES="Standard training deck corpus (${count} decks, sha256 ${SHA})."
fi

echo "Creating release ${TAG}…"
# Pin the tag to the remote default branch so a stray/unpushed local tag of the
# same name can't block the release (data releases aren't tied to code state).
TARGET="$(gh repo view --json defaultBranchRef --jq '.defaultBranchRef.name')"
gh release create "$TAG" dist/decks.tar.gz dist/decks.sha256 \
  --title "$TAG" --notes "$NOTES" --target "$TARGET"

echo
echo "Published ${TAG} (sha256=${SHA})."
echo "Nothing to commit — fetch_decks.sh picks it up as the newest release."
