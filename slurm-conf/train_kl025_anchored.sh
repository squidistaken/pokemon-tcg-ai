#!/bin/bash

#SBATCH --job-name=pokemon-tcg-kl025
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

# The kl_anchor_coeff=0.25 arm, continued from the desktop run kl0.25-local-16w.
#
#   sbatch slurm-conf/train_kl025_anchored.sh
#
# Submit it from /scratch/s4325621/pokemon-tcg-ai-kl025, the tree
# scripts/habrok_sync_kl025_resume.sh builds. That tree is separate from the one
# the kl0.05 arm is running out of, so neither job can change the other's code,
# config or state.
#
# Three things differ from slurm-conf/train_kl_anchored.sh:
#
#   agent.kl_anchor_coeff=0.25   0.05 lets kl_to_bc climb past 0.36 and the
#                                policy loses to its own pre-RL ancestor. 0.25
#                                held it at 0.11 over the first 2.6M frames.
#   train.eval_opponents         the behaviour-cloned policy replaces
#                                first_snapshot, so the panel measures "better
#                                than the clone" directly instead of through a
#                                snapshot taken 500,000 frames after it.
#   RESUME=1                     the run directory already holds the state this
#                                job continues.
#
# The clone is also the league's permanent anchor: the launcher passes
# train.warmup_checkpoint, which replaces the RandomOpponent in every worker's
# pool (src/env/opponents/snapshot_opponent_pool.py:131).

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

WANDB_GROUP="${WANDB_GROUP:-kl-coeff-ablation-20260816}"
RUN_NAME="${RUN_NAME:-kl0.25-bcv6-habrok32}"
RUN_DIR="outputs/$WANDB_GROUP/$RUN_NAME"

# Habrok does not export the submitting shell's environment into the job, so
# the seeded state is checked here rather than trusted from a flag: a missing
# file would otherwise start this arm over from the clone at 0 frames and look
# like a working run.
if [ ! -f "$RUN_DIR/train_state.pt" ]; then
  echo "ERROR: $RUN_DIR/train_state.pt is missing, so there is nothing to continue." >&2
  echo "       Run ./scripts/habrok_sync_kl025_resume.sh from the desktop first." >&2
  exit 1
fi

BC_CHECKPOINT="${BC_CHECKPOINT:-outputs/bc/bc-v6-submit.pt}"
if [ ! -f "$BC_CHECKPOINT" ]; then
  echo "ERROR: behaviour-cloned checkpoint not found: $BC_CHECKPOINT" >&2
  echo "       It is the clone, the KL reference, the league anchor and now an" >&2
  echo "       eval reference; the run cannot start without it." >&2
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
echo "Run dir: $RUN_DIR"
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
#
# TOTAL_FRAMES is absolute and deliberately past what 24h collects (~77M at
# ~890 fps): the supervisor stops cleanly at the time limit and leaves
# train_state.pt ready for the next resume.
NUM_WORKERS="${NUM_WORKERS:-32}" \
TOTAL_FRAMES="${TOTAL_FRAMES:-99991552}" \
WANDB_GROUP="$WANDB_GROUP" \
RUN_NAME="$RUN_NAME" \
BC_CHECKPOINT="$BC_CHECKPOINT" \
RESUME=1 \
  ./scripts/train_kl_anchored_selfplay.sh \
    agent.kl_anchor_coeff=0.25 \
    "train.eval_opponents=[$BC_CHECKPOINT,checkpoint,random]"

echo "End: $(date --iso-8601=seconds)"
