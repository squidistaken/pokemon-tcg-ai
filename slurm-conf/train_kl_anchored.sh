#!/bin/bash

#SBATCH --job-name=pokemon-tcg-kl-anchored
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

# Self-play PPO from the behaviour-cloned policy, held near it by a KL term for
# the whole run. Hyperparameters live in conf/experiment/kl_anchored_selfplay.yaml.
#
#   sbatch slurm-conf/train_kl_anchored.sh
#   RESUME=1 sbatch slurm-conf/train_kl_anchored.sh      # continue after a stop
#
# TOTAL_FRAMES is deliberately past what 24h collects (~77M at ~890 fps): the
# supervisor stops cleanly at the time limit and leaves train_state.pt ready for
# the next resume.
#
# This submits directly rather than through submit.py because the run needs
# scripts/train_kl_anchored_selfplay.sh, which carries the override set and the
# restart loop; run_job.sh calls `python -m src.train` straight.

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this script from the project root with sbatch}"

PYTHON_MODULE="${PYTHON_MODULE:-Python/3.13.5-GCCcore-14.3.0}"
UV_MODULE="${UV_MODULE:-uv/0.10.7}"

# Pick the environment from the GPU Slurm actually handed us, the same pairing
# submit.py enforces: .venv-rtx for an RTX Pro 6000, .venv for anything else.
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
  echo "       Run: ./slurm-conf/setup_uv.sh          (.venv, for a100)" >&2
  echo "            ./slurm-conf/setup_uv.sh --use-rtx (.venv-rtx, for rtx_pro_6000)" >&2
  exit 1
fi

BC_CHECKPOINT="${BC_CHECKPOINT:-outputs/bc/bc-v6-submit.pt}"
if [ ! -f "$BC_CHECKPOINT" ]; then
  echo "ERROR: behaviour-cloned checkpoint not found: $BC_CHECKPOINT" >&2
  echo "       Copy it across before submitting, for example:" >&2
  echo "       rsync -av outputs/bc/bc-v6-submit.pt habrok:$PROJECT_ROOT/outputs/bc/" >&2
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

export WANDB_MODE="${WANDB_MODE:-online}"

echo "Job: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "uv environment: $UV_PROJECT_ENVIRONMENT"
echo "Clone: $BC_CHECKPOINT"
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
NUM_WORKERS="${NUM_WORKERS:-32}" \
TOTAL_FRAMES="${TOTAL_FRAMES:-300000000}" \
BC_CHECKPOINT="$BC_CHECKPOINT" \
RESUME="${RESUME:-0}" \
  ./scripts/train_kl_anchored_selfplay.sh

echo "End: $(date --iso-8601=seconds)"
