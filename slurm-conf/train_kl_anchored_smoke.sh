#!/bin/bash

#SBATCH --job-name=pokemon-tcg-kl-smoke
#SBATCH --partition=gpushort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-conf/logs/%x_%j.out
#SBATCH --error=slurm-conf/logs/%x_%j.err

# Cheap validation of the KL-anchored run before committing an RTX Pro 6000
# allocation. Same code path, same config, same guards; only the resources, the
# frame budget and the run directory differ.
#
#   sbatch slurm-conf/train_kl_anchored_smoke.sh 1    # test the resume path
#   sbatch slurm-conf/train_kl_anchored_smoke.sh      # test a cold start
#
# The trailing argument is the resume flag, matching train_kl_anchored.sh. It is
# an argument rather than an environment variable because Habrok does not export
# the submitting shell's environment into the job: `RESUME=1 sbatch ...` arrives
# unset, which is what killed job 30638714 eighty seconds in.
#
# With the flag set, the smoke run directory is seeded from the real run's
# train_state.pt and snapshots, so this exercises the same resume that the long
# job will do. Without it, the run cold-starts from the clone.
#
# gpushort caps at 4 hours and has the most idle nodes. The GPU is pinned to
# a100:1 rather than left untyped, because a bare --gpus-per-node=1 lets Slurm
# hand over a V100, and the torch==2.13.0+cu130 wheels in this repo have no
# kernel below compute capability 7.5 (V100 is 7.0). Smoke job 30638940 died on
# exactly that: "no kernel image is available for execution on the device".
#
# 30 minutes and 8 CPUs keep it in the small-job lane. 200,000 frames passes the
# first evaluation at 50,000, which is where a missing eval opponent or a bad
# deck pool would surface.
#
# ATTEMPTS=1 on purpose: the supervisor must not hide a failure behind a retry
# when the whole point is to see whether it fails.

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this script from the project root with sbatch}"

RESUME_FLAG="${1:-${RESUME:-0}}"
if [ "$RESUME_FLAG" != "0" ] && [ "$RESUME_FLAG" != "1" ]; then
  echo "ERROR: resume flag must be 0 or 1, got '$RESUME_FLAG'." >&2
  echo "       Usage: sbatch slurm-conf/train_kl_anchored_smoke.sh [0|1]" >&2
  exit 1
fi

# Fixed rather than job-id-suffixed, because a resume has to find the state a
# previous submission left. The name carries "smoke" and the guard below refuses
# to touch a directory that does not, so a typo cannot reach the real run.
SMOKE_GROUP="kl-anchored-smoke-habrok"
SMOKE_NAME="kl0.05-smoke"
SMOKE_DIR="$PROJECT_ROOT/outputs/$SMOKE_GROUP/$SMOKE_NAME"

REAL_DIR="$PROJECT_ROOT/outputs/kl-anchored-selfplay-20260816/kl0.05-bcv6-expertpool"

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

# Rebuild the smoke directory from scratch every submission, so one run cannot
# inherit half a state from the last. Refuse outright if the path does not name
# the smoke run: this is the only rm in the file and it must never be able to
# reach outputs/kl-anchored-selfplay-*.
case "$SMOKE_DIR" in
  *"/outputs/$SMOKE_GROUP/$SMOKE_NAME") ;;
  *) echo "ERROR: refusing to clear '$SMOKE_DIR'; it is not the smoke run." >&2; exit 1 ;;
esac
rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR/checkpoints"

if [ "$RESUME_FLAG" = "1" ]; then
  if [ ! -f "$REAL_DIR/train_state.pt" ]; then
    echo "ERROR: nothing to resume from: $REAL_DIR/train_state.pt is missing." >&2
    echo "       Submit without the flag to test a cold start instead." >&2
    exit 1
  fi
  # Copies only. The real run directory is read here and never written.
  cp "$REAL_DIR/train_state.pt" "$SMOKE_DIR/train_state.pt"

  # The league needs the newest pool_size snapshots, and first_snapshot resolves
  # to the lowest frame number in the directory, so the oldest one comes too.
  mapfile -t SNAPSHOTS < <(find "$REAL_DIR/checkpoints" -maxdepth 1 -name 'snapshot_*.pt' | sort)
  if [ "${#SNAPSHOTS[@]}" -eq 0 ]; then
    echo "ERROR: no snapshots in $REAL_DIR/checkpoints to seed the league." >&2
    exit 1
  fi
  cp "${SNAPSHOTS[0]}" "$SMOKE_DIR/checkpoints/"
  for snapshot in "${SNAPSHOTS[@]: -10}"; do
    cp "$snapshot" "$SMOKE_DIR/checkpoints/"
    # An `if` rather than `&&`: a false test as the last command of the body
    # returns 1, and under `set -e` that ends the job.
    if [ -f "${snapshot%.pt}.json" ]; then
      cp "${snapshot%.pt}.json" "$SMOKE_DIR/checkpoints/"
    fi
  done
  echo "Seeded $SMOKE_DIR from $REAL_DIR:"
  echo "  train_state.pt plus $(ls -1 "$SMOKE_DIR/checkpoints"/snapshot_*.pt | wc -l) snapshot(s)"

  # TOTAL_FRAMES is an absolute target that the supervisor subtracts the resumed
  # frame count from, so the default 200,000 would already be behind the 8.5M in
  # train_state.pt and the run would exit without collecting anything. Read the
  # count and add the budget on top.
  STATE_FRAMES="$(uv run --frozen --no-sync python - "$SMOKE_DIR/train_state.pt" <<'PY'
import sys

import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(payload.get("frames", 0)) if isinstance(payload, dict) else 0)
PY
)"
  if [ "$STATE_FRAMES" -le 0 ]; then
    echo "ERROR: $SMOKE_DIR/train_state.pt records no frame count." >&2
    exit 1
  fi
  SMOKE_BUDGET="${SMOKE_BUDGET:-200000}"
  TOTAL_FRAMES=$((STATE_FRAMES + SMOKE_BUDGET))
  echo "Resuming at $STATE_FRAMES frames; collecting $SMOKE_BUDGET more to $TOTAL_FRAMES."
else
  TOTAL_FRAMES="${TOTAL_FRAMES:-200000}"
fi

# The real run evaluates every 500,000 frames, which this budget never reaches,
# so the interval is shortened here. Evaluation is the step that loads the two
# snapshot opponents, and a missing or unreadable one is exactly the failure
# this job exists to catch.
WANDB_GROUP="$SMOKE_GROUP" \
RUN_NAME="$SMOKE_NAME" \
NUM_WORKERS="${NUM_WORKERS:-8}" \
TOTAL_FRAMES="$TOTAL_FRAMES" \
ATTEMPTS=1 \
RESUME="$RESUME_FLAG" \
  ./scripts/train_kl_anchored_selfplay.sh \
    train.eval_interval=50000 \
    train.eval_episodes=6

echo "End: $(date --iso-8601=seconds)"
echo "SMOKE OK: the same path the long job uses completed without error."
