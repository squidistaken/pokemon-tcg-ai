#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=""
USE_RTX=false

while [ "$#" -gt 0 ]; do
  case "$1" in
    --use-rtx)
      USE_RTX=true
      ;;
    -h|--help)
      echo "Usage: setup_uv.sh [--use-rtx] [PROJECT_ROOT]"
      exit 0
      ;;
    -*)
      echo "ERROR: unknown option: $1" >&2
      exit 2
      ;;
    *)
      if [ -n "$PROJECT_ROOT" ]; then
        echo "ERROR: only one project root may be provided" >&2
        exit 2
      fi
      PROJECT_ROOT="$1"
      ;;
  esac
  shift
done

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
INSTALL_SCRIPT="$SCRIPT_DIR/install_uv.sh"

if [ ! -f "$PROJECT_ROOT/pyproject.toml" ] || [ ! -f "$PROJECT_ROOT/uv.lock" ]; then
  echo "ERROR: project root must contain pyproject.toml and uv.lock: $PROJECT_ROOT" >&2
  exit 1
fi

if [ "$USE_RTX" = false ]; then
  exec bash "$INSTALL_SCRIPT" .venv "$PROJECT_ROOT"
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch is required for RTX setup; run this on a Slurm login node" >&2
  exit 1
fi

LOG_DIR="$PROJECT_ROOT/slurm-conf/logs"
mkdir -p "$LOG_DIR"

echo "Submitting a 15-minute RTX environment setup job..."
exec sbatch \
  --job-name=pokemon-tcg-setup-rtx \
  --time=00:15:00 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=16G \
  --partition=gpu \
  --gpus-per-node=rtx_pro_6000:1 \
  --chdir="$PROJECT_ROOT" \
  --output="$LOG_DIR/%x_%j.out" \
  --error="$LOG_DIR/%x_%j.err" \
  "$INSTALL_SCRIPT" .venv-rtx "$PROJECT_ROOT"
