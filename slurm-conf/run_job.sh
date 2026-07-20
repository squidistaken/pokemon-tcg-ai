#!/bin/bash

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: run_job.sh PROFILE_PATH CONFIG_NAME UV_ENVIRONMENT [HYDRA_OVERRIDE ...]" >&2
  exit 2
fi

PROFILE_PATH="$1"
CONFIG_NAME="$2"
UV_ENVIRONMENT="$3"
shift 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.13.5-GCCcore-14.3.0}"
UV_MODULE="${UV_MODULE:-uv/0.10.7}"

module purge
module load "$PYTHON_MODULE"
module load "$UV_MODULE"

cd "$PROJECT_ROOT"
export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/$UV_ENVIRONMENT"
if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
  echo "ERROR: uv environment is missing: $UV_PROJECT_ENVIRONMENT" >&2
  exit 1
fi

JOB_TMP_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
export TMPDIR="$JOB_TMP_ROOT/pokemon-tcg-${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"

# The default run uses 16 collector processes on 16 allocated cores. Prevent
# each process from creating another full pool of math-library threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Profile: $PROFILE_PATH"
echo "Hydra config: conf/$CONFIG_NAME.yaml"
echo "uv environment: $UV_ENVIRONMENT"
echo "Job: ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start: $(date --iso-8601=seconds)"

uv run --frozen --no-sync python - <<'PY'
import torch
from cg import sim

if not torch.cuda.is_available():
    raise SystemExit("CUDA was requested by the Slurm profile but PyTorch cannot use it")

device = torch.device("cuda")
result = (torch.ones((32, 32), device=device) @ torch.ones((32, 32), device=device)).sum()
torch.cuda.synchronize()

print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
print(f"CUDA calculation result: {result.item():.0f}")
print(f"Engine: {sim.lib._name}")
PY

srun uv run --frozen --no-sync python -m src.train --config-name "$CONFIG_NAME" "$@"

echo "End: $(date --iso-8601=seconds)"
