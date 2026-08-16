#!/bin/bash

#SBATCH --job-name=pokemon-tcg-kl-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-conf/logs/%x_%j.out
#SBATCH --error=slurm-conf/logs/%x_%j.err

# Cheap validation of the KL-anchored run before committing a 24-hour RTX Pro
# 6000 allocation. Same code path, same config, same guards; only the resources
# and the frame budget differ.
#
#   sbatch slurm-conf/train_kl_anchored_smoke.sh
#
# No gpu_type is requested, so Slurm hands over whatever is free and the job
# queues for the shortest time. 30 minutes and 8 CPUs keep it in the small-job
# lane. 200,000 frames is enough to pass the first evaluation at 50,000 frames,
# which is where a missing eval opponent or a bad deck pool would surface.
#
# ATTEMPTS=1 on purpose: the supervisor must not hide a failure behind a retry
# when the whole point is to see whether it fails.

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this script from the project root with sbatch}"

PYTHON_MODULE="${PYTHON_MODULE:-Python/3.13.5-GCCcore-14.3.0}"
UV_MODULE="${UV_MODULE:-uv/0.10.7}"

# Pick the environment from the GPU Slurm actually handed us: .venv-rtx for an
# RTX Pro 6000, .venv for anything else. This job takes any GPU, so both
# branches are live here.
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

if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
  echo "ERROR: uv environment is missing: $UV_PROJECT_ENVIRONMENT" >&2
  echo "       Run: ./slurm-conf/setup_uv.sh          (.venv)" >&2
  echo "            ./slurm-conf/setup_uv.sh --use-rtx (.venv-rtx)" >&2
  exit 1
fi

JOB_TMP_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
export TMPDIR="$JOB_TMP_ROOT/pokemon-tcg-${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

export WANDB_MODE="${WANDB_MODE:-online}"

echo "Job: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "uv environment: $UV_PROJECT_ENVIRONMENT"
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
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA calculation result: {result.item():.0f}")
print(f"Engine: {sim.lib._name}")
PY

# A separate run directory and W&B name, so the smoke test cannot collide with
# the real run or leave a train_state.pt that blocks it.
# The real run evaluates every 500,000 frames, which this budget never reaches,
# so the interval is shortened here. Evaluation is the step that loads the two
# snapshot opponents, and a missing or unreadable one is exactly the failure
# this job exists to catch.
WANDB_GROUP="kl-anchored-smoke-habrok" \
RUN_NAME="kl0.05-smoke-${SLURM_JOB_ID:-manual}" \
NUM_WORKERS="${NUM_WORKERS:-8}" \
TOTAL_FRAMES="${TOTAL_FRAMES:-200000}" \
ATTEMPTS=1 \
  ./scripts/train_kl_anchored_selfplay.sh \
    train.eval_interval=50000 \
    train.eval_episodes=6

echo "End: $(date --iso-8601=seconds)"
echo "SMOKE OK: the same path the 24h job uses completed without error."
