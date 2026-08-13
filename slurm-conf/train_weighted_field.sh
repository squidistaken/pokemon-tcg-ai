#!/bin/bash

#SBATCH --job-name=pokemon-tcg-weighted-field
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-conf/logs/%x_%j.out
#SBATCH --error=slurm-conf/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# Continues the weighted-field arm from the train_state.pt copied off the
# desktop. The supervisor reads the frame count out of that file and collects
# only the remainder, so TOTAL_FRAMES stays the absolute target. TOTAL_FRAMES
# is deliberately set past what 24h can collect (~227M at ~890 fps): the job
# runs the full wallclock allocation and the supervisor stops it cleanly at the
# time limit, leaving train_state.pt ready for the next resume.
#
# This submits directly rather than through submit.py because the arm needs
# scripts/train_tf_weighted_field.sh: that wrapper carries the override set and
# the 10-attempt restart loop, and run_job.sh calls `python -m src.train`
# straight. Resources mirror slurm-conf/train_gpu.yaml, with
# cpus_per_task and the collector worker count raised together.

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this script from the project root with sbatch}"

PYTHON_MODULE="${PYTHON_MODULE:-Python/3.13.5-GCCcore-14.3.0}"
UV_MODULE="${UV_MODULE:-uv/0.10.7}"
# Pick the environment from the GPU Slurm actually handed us, the same pairing
# submit.py's _uv_environment enforces: .venv-rtx for an RTX Pro 6000, .venv for
# anything else. Choosing from the hardware rather than from a flag means
# `sbatch --gpus-per-node=a100:1` needs no second override to stay consistent.
if [ -z "${UV_PROJECT_ENVIRONMENT:-}" ]; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)"
  case "$GPU_NAME" in
    *"RTX PRO 6000"*|*"RTX Pro 6000"*) UV_ENVIRONMENT="$PROJECT_ROOT/.venv-rtx" ;;
    *) UV_ENVIRONMENT="$PROJECT_ROOT/.venv" ;;
  esac
  echo "Detected GPU: ${GPU_NAME:-unknown}"
else
  UV_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT"
fi

module purge
module load "$PYTHON_MODULE"
module load "$UV_MODULE"

cd "$PROJECT_ROOT"
export UV_PROJECT_ENVIRONMENT="$UV_ENVIRONMENT"
export CHECKPOINT_KEYS_FILE="$PROJECT_ROOT/logs/checkpoint_keys.csv"

if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
  echo "ERROR: uv environment is missing: $UV_PROJECT_ENVIRONMENT" >&2
  echo "       Run: ./slurm-conf/setup_uv.sh          (.venv, for a100)" >&2
  echo "            ./slurm-conf/setup_uv.sh --use-rtx (.venv-rtx, for rtx_pro_6000)" >&2
  exit 1
fi

JOB_TMP_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
export TMPDIR="$JOB_TMP_ROOT/pokemon-tcg-${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"

# 32 collector processes on 32 allocated cores. Prevent each from creating
# another full pool of math-library threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Live W&B, from WANDB_API_KEY in .env or a verified wandb login.
export WANDB_MODE="${WANDB_MODE:-online}"

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "uv environment: $UV_PROJECT_ENVIRONMENT"
echo "Start: $(date --iso-8601=seconds)"

# Same preflight as run_job.sh: fail here rather than an hour in if CUDA is
# unusable or the vendored engine library cannot load on this node.
uv run --frozen --no-sync python - <<'PY'
import torch
from cg import sim

if not torch.cuda.is_available():
    raise SystemExit("CUDA was requested by the Slurm profile but PyTorch cannot use it")

device = torch.device("cuda")
result = (torch.ones((32, 32), device=device) @ torch.ones((32, 32), device=device)).sum()
torch.cuda.synchronize()

print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA calculation result: {result.item():.0f}")
print(f"Engine: {sim.lib._name}")
PY

# No srun: the supervisor runs a sequence of training attempts, and each one
# would become its own step.
RESUME=1 \
NUM_WORKERS="${NUM_WORKERS:-32}" \
TOTAL_FRAMES="${TOTAL_FRAMES:-300000000}" \
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}" \
  ./scripts/train_tf_weighted_field.sh

echo "End: $(date --iso-8601=seconds)"
