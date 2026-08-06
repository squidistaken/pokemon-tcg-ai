#!/bin/bash

#SBATCH --job-name=pokemon-tcg-scrape
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-conf/logs/%x_%j.out
#SBATCH --error=slurm-conf/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this script from the project root with sbatch}"

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
  --minimum-mapping-confidence "${MAPPING_MIN_CONFIDENCE:-1}" \
  --limit "${SCRAPER_LIMIT:-200}" \
  --max-pages "${SCRAPER_MAX_PAGES:-0}" \
  --per-tournament "${SCRAPER_PER_TOURNAMENT:-8}" \
  --since "${SCRAPER_SINCE:-2026-01-01}" \
  --category "${BULBAPEDIA_CATEGORY:-Deck archetypes}" \
  --bulbapedia-max-pages "${BULBAPEDIA_MAX_PAGES:-0}" \
  --out "${SCRAPER_OUT:-decks}" \
  --verbose \
  "$@"

echo "End: $(date --iso-8601=seconds)"
