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

SLURM_CONFIG=""
EXPECT_SLURM_CONFIG=false
for ARG in "$@"; do
  if [ "$EXPECT_SLURM_CONFIG" = true ]; then
    SLURM_CONFIG="$ARG"
    EXPECT_SLURM_CONFIG=false
  elif [ "$ARG" = "--slurm-config" ]; then
    EXPECT_SLURM_CONFIG=true
  fi
done

if [ -z "$SLURM_CONFIG" ]; then
  echo "ERROR: --slurm-config must be followed by a config name" >&2
  exit 2
fi

case "$(basename "$SLURM_CONFIG")" in
  train_gpu_rtx|train_gpu_rtx.yaml)
    UV_ENVIRONMENT=".venv-rtx"
    SETUP_COMMAND="./slurm-conf/setup_uv.sh --use-rtx"
    ;;
  *)
    UV_ENVIRONMENT=".venv"
    SETUP_COMMAND="./slurm-conf/setup_uv.sh"
    ;;
esac

module purge
module load "$PYTHON_MODULE"
module load "$UV_MODULE"

cd "$PROJECT_ROOT"
export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/$UV_ENVIRONMENT"
if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
  echo "ERROR: required environment $UV_ENVIRONMENT is missing; run: $SETUP_COMMAND" >&2
  exit 1
fi

exec uv run --frozen --no-sync python "$SCRIPT_DIR/submit.py" "$@"
