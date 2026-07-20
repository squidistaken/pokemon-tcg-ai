#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=""
USE_RTX=false
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.13.5-GCCcore-14.3.0}"
UV_MODULE="${UV_MODULE:-uv/0.10.7}"

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
if [ "$USE_RTX" = true ]; then
  export UV_PROJECT_ENVIRONMENT=".venv-rtx"
else
  export UV_PROJECT_ENVIRONMENT=".venv"
fi

if ! command -v module >/dev/null 2>&1; then
  echo "ERROR: an environment-modules installation is required" >&2
  exit 1
fi
if [ ! -f "$PROJECT_ROOT/pyproject.toml" ] || [ ! -f "$PROJECT_ROOT/uv.lock" ]; then
  echo "ERROR: project root must contain pyproject.toml and uv.lock: $PROJECT_ROOT" >&2
  exit 1
fi

module purge
module load "$PYTHON_MODULE"
module load "$UV_MODULE"

cd "$PROJECT_ROOT"
echo "Python: $(python --version)"
echo "uv: $(uv --version)"
echo "Project: $PROJECT_ROOT"
echo "Environment: $UV_PROJECT_ENVIRONMENT"

# Install the exact package versions from uv.lock.
uv sync --frozen

# Check that Python can load PyTorch and the included Linux game engine.
# This does not build or change the engine.
uv run --frozen --no-sync python - <<'PY'
import platform

import dotenv
import torch
import wandb
from cg import sim

print(f"Platform: {platform.platform()}")
print(f"PyTorch: {torch.__version__}")
print(f"W&B: {wandb.__version__}")
print(f"Engine: {sim.lib._name}")
print("Environment check passed")
PY
