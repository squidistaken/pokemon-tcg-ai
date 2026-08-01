#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd -P)}"
LOG_DIR="$PROJECT_ROOT/slurm-conf/logs"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  mkdir -p "$LOG_DIR"
  exec sbatch \
    --chdir="$PROJECT_ROOT" \
    --job-name=pokemon-tcg-scrape \
    --time=48:00:00 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=4G \
    --partition=regular \
    --output="$LOG_DIR/%x_%j.out" \
    --error="$LOG_DIR/%x_%j.err" \
    "$0" "$@"
fi

PYTHON_MODULE="${PYTHON_MODULE:-Python/3.13.5-GCCcore-14.3.0}"
UV_MODULE="${UV_MODULE:-uv/0.10.7}"
UV_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PROJECT_ROOT/.venv}"

module purge
module load "$PYTHON_MODULE"
module load "$UV_MODULE"

cd "$PROJECT_ROOT"
export UV_PROJECT_ENVIRONMENT="$UV_ENVIRONMENT"

if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
  echo "ERROR: uv environment is missing: $UV_PROJECT_ENVIRONMENT" >&2
  exit 1
fi

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"

srun uv run --frozen --no-sync python -m scraper \
  --source all \
  --card-swap-strategy all \
  --limit "${SCRAPER_LIMIT:-200}" \
  --max-pages "${SCRAPER_MAX_PAGES:-0}" \
  --max-decks "${SCRAPER_MAX_DECKS:-5000}" \
  --per-tournament "${SCRAPER_PER_TOURNAMENT:-8}" \
  --since "${SCRAPER_SINCE:-2026-01-01}" \
  --category "${BULBAPEDIA_CATEGORY:-Deck archetypes}" \
  --out "${SCRAPER_OUT:-decks}" \
  --verbose \
  "$@"

echo "End: $(date --iso-8601=seconds)"
