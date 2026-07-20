#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.13.5-GCCcore-14.3.0}"
UV_MODULE="${UV_MODULE:-uv/0.10.7}"

if [ "$#" -eq 0 ]; then
  echo "Usage: train.sh --config NAME --slurm-config NAME [HYDRA_OVERRIDE ...]" >&2
  exit 2
fi

module purge
module load "$PYTHON_MODULE"
module load "$UV_MODULE"

cd "$PROJECT_ROOT"
if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
  CONTROL_ENVIRONMENT=".venv"
elif [ -x "$PROJECT_ROOT/.venv-rtx/bin/python" ]; then
  CONTROL_ENVIRONMENT=".venv-rtx"
else
  echo "ERROR: no uv environment is available; run: ./slurm-conf/setup_uv.sh" >&2
  exit 1
fi
export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/$CONTROL_ENVIRONMENT"

exec uv run --frozen --no-sync python "$SCRIPT_DIR/submit.py" "$@"
