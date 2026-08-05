#!/bin/bash
#SBATCH --job-name=ptcg-scraper
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=regularmedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

module purge
module load Python/3.13.5-GCCcore-14.3.0
module load uv/0.10.7

uv run --frozen --no-sync python3 -m scraper \
  --since 2022-11-01 \
  --until 2026-03-27 \
  --max-pages 0 \
  --source limitless \
  --page 50 \
  --out /scratch/s5195179/ptcg_decks_test \
  -v \
