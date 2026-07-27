#!/bin/bash

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: install_uv.sh UV_ENVIRONMENT [PROJECT_ROOT]" >&2
  exit 2
fi

UV_ENVIRONMENT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${2:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.13.5-GCCcore-14.3.0}"
UV_MODULE="${UV_MODULE:-uv/0.10.7}"

case "$UV_ENVIRONMENT" in
  .venv)
    VERIFY_RTX=0
    ;;
  .venv-rtx)
    VERIFY_RTX=1
    if [ -z "${SLURM_JOB_ID:-}" ] || [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
      echo "ERROR: .venv-rtx must be installed inside an RTX Slurm job" >&2
      exit 1
    fi
    ;;
  *)
    echo "ERROR: unsupported uv environment: $UV_ENVIRONMENT" >&2
    exit 2
    ;;
esac

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
export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/$UV_ENVIRONMENT"
export VERIFY_RTX

echo "Python: $(python --version)"
echo "uv: $(uv --version)"
echo "Project: $PROJECT_ROOT"
echo "Environment: $UV_PROJECT_ENVIRONMENT"

uv sync --frozen

uv run --frozen --no-sync python - <<'PY'
import os
import platform

import dotenv
import torch
import wandb
from cg import sim

print(f"Platform: {platform.platform()}")
print(f"PyTorch: {torch.__version__}")
print(f"W&B: {wandb.__version__}")
print(f"Engine: {sim.lib._name}")

if os.environ["VERIFY_RTX"] == "1":
    if not torch.cuda.is_available():
        raise SystemExit("RTX setup job cannot use its allocated GPU")
    capability = torch.cuda.get_device_capability(0)
    if capability < (12, 0):
        raise SystemExit(f"Expected RTX compute capability 12.0, found {capability}")
    result = (torch.ones((32, 32), device="cuda") @ torch.ones((32, 32), device="cuda")).sum()
    torch.cuda.synchronize()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {capability}")
    print(f"CUDA calculation result: {result.item():.0f}")

print("Environment check passed")
PY
