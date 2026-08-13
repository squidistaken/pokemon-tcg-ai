#!/bin/bash

#SBATCH --job-name=pokemon-tcg-deck-probe-8way
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=slurm-conf/logs/%x_%j.out
#SBATCH --error=slurm-conf/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# Eight deck-pinned arms from one checkpoint, sharing one GPU for a day.
#
#   CHECKPOINT=/scratch/s4325621/pokemon-tcg-ai/outputs/weighted-field-20260808/\
#   tf-ptr-weighted-15m-s42/checkpoints/snapshot_000176357376.pt \
#     sbatch slurm-conf/train_deck_probe_8way.sh
#
# One GPU rather than eight. The bottleneck is CPU-side collection, not the
# card: the model is 0.93M parameters, only the main process of each arm holds a
# CUDA context (the collector workers step the engine on CPU), and one arm
# measured 7.5 GB of VRAM. Eight arms is ~60 GB of the RTX Pro 6000's 96 GB.
# What the arms actually compete for is the 64 cores, 8 collector workers each.
#
# Resources against slurm-conf/train_weighted_field.sh, which runs a single arm
# at 32 cores / 32 workers / 901 fps: twice the cores, split eight ways, so each
# arm gets a quarter of the workers and lands near 225 fps. That is ~19M frames
# per arm over the 24h allocation, and ~155M frames of total experience.
#
# This submits directly rather than through submit.py because the arms need
# scripts/train_deck_probe_arm.sh: that wrapper carries the override set and the
# 10-attempt restart loop, and run_job.sh calls `python -m src.train` straight.

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

# 64 collector workers plus 8 trainers on 64 allocated cores. Prevent each from
# creating another full pool of math-library threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Live W&B, from WANDB_API_KEY in .env or a verified wandb login.
export WANDB_MODE="${WANDB_MODE:-online}"

GROUP="${GROUP:-deck-probe-8way}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# Frames on top of the warm start, turned into an absolute target below. Past
# what a day collects on purpose, so the wallclock stops the arms rather than
# the budget and no arm hands its cores to the others by finishing early.
BUDGET="${BUDGET:-100000000}"

if [ -z "${CHECKPOINT:-}" ]; then
  echo "ERROR: CHECKPOINT is required: the snapshot all eight arms start from." >&2
  echo "       CHECKPOINT=/path/snapshot_000NNNNNNNNN.pt sbatch $0" >&2
  exit 2
fi
if [ ! -f "$CHECKPOINT" ]; then
  echo "ERROR: CHECKPOINT $CHECKPOINT does not exist." >&2
  exit 2
fi

# The eight candidate decks, chosen from tournament data over the 60 days before
# 2026-08-13: appearance counts pick the pool, aggregate match record decides
# who stays. Every one of these is played and at least breaking even; the decks
# left out lose (Hop's Trevenant 0.445, Dragapult Dusknoir 0.461, Lucario
# Hariyama 0.483). See docs/deck-probe-8way.md for the full selection.
#
# Five of the eight sit outside decks/top20, so the full corpus has to be
# present. The pinned deck is loaded by path and does not join the pool, and the
# pool itself only builds the eval panel: training is 100% mirror on the pin.
DECKS=(
  "decks/heuristic-resolved/rockets-honchkrow/rockets-honchkrow-2.csv"
  "decks/heuristic-resolved/alakazam-dudunsparce/alakazam-dudunsparce-4.csv"
  "decks/heuristic-resolved/slowking/slowking.csv"
  "decks/heuristic-resolved/dragapult-blaziken/dragapult-blaziken-3.csv"
  "decks/heuristic-resolved/basic-box/basic-box-37.csv"
  "decks/heuristic-resolved/lopunny-dudunsparce/lopunny-dudunsparce-2.csv"
  "decks/heuristic-resolved/ogerpon-meganium-hydrapple/ogerpon-meganium-hydrapple-2.csv"
  "decks/heuristic-resolved/flareon-noctowl/flareon-noctowl-13.csv"
)
NAMES=(
  honchkrow
  alakazam
  slowking
  blaziken
  basic-box
  lopunny
  ogerpon-hydrapple
  flareon
)

echo "Job: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "uv environment: $UV_PROJECT_ENVIRONMENT"
echo "Start: $(date --iso-8601=seconds)"

# Same preflight as run_job.sh: fail here rather than an hour in if CUDA is
# unusable or the vendored engine library cannot load on this node. Done once
# for all eight arms, since they share the card.
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

# Read the warm start's frame count once, here, so every arm gets the same
# absolute target. Letting each arm read it would be eight torch.load calls of
# the same file and, worse, would let a partially written checkpoint give two
# arms different budgets.
WARM_FRAMES="$(uv run --frozen --no-sync python - "$CHECKPOINT" <<'PY'
import sys

import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
frames = payload.get("frames") if isinstance(payload, dict) else None
print(int(frames) if frames else 0)
PY
)"
if [ "$WARM_FRAMES" -eq 0 ]; then
  echo "ERROR: $CHECKPOINT records no frame count; the arms would restart at zero." >&2
  exit 2
fi
TOTAL_FRAMES=$((WARM_FRAMES + BUDGET))

echo "Anchor: $CHECKPOINT at $WARM_FRAMES frames"
echo "Target: $TOTAL_FRAMES frames absolute ($BUDGET on top, deliberately out of reach)"
echo "Arms:   ${#DECKS[@]} x $NUM_WORKERS workers on one GPU"

mkdir -p logs/deck-probe

ARM_PIDS=()
for i in "${!DECKS[@]}"; do
  name="${NAMES[$i]}"
  log="logs/deck-probe/${GROUP}-${name}-${SLURM_JOB_ID:-manual}.log"
  echo "  arm $((i + 1))/${#DECKS[@]}: $name -> $log"
  # setsid puts each arm in its own process group. Without it every arm shares
  # this script's group, and the group-wide kill below would take the script
  # down with them before it could wait for their state writes.
  setsid env \
    DECK="${DECKS[$i]}" \
    RUN_NAME="pinned-$name" \
    CHECKPOINT="$CHECKPOINT" \
    TOTAL_FRAMES="$TOTAL_FRAMES" \
    NUM_WORKERS="$NUM_WORKERS" \
    GROUP="$GROUP" \
    RESUME="${RESUME:-0}" \
    ./scripts/train_deck_probe_arm.sh > "$log" 2>&1 &
  ARM_PIDS+=("$!")
done

# --signal=B:TERM@300 delivers SIGTERM to this script five minutes before the
# wallclock. Forward it to every arm's process group so each trainer writes
# train_state.pt and flushes W&B before Slurm's SIGKILL arrives. The supervisor
# reads 143 as a deliberate stop and does not restart into it
# (scripts/train_supervised.sh:201).
forward_term() {
  echo "Received SIGTERM at $(date --iso-8601=seconds); stopping ${#ARM_PIDS[@]} arms."
  for pid in "${ARM_PIDS[@]}"; do
    kill -TERM -"$pid" 2>/dev/null || true
  done
}
trap forward_term TERM

# One wait per arm rather than a bare `wait`: the trap interrupts whichever wait
# is current, and the loop then keeps waiting for the arms still shutting down.
status=0
for pid in "${ARM_PIDS[@]}"; do
  wait "$pid" || status=$?
done

echo "End: $(date --iso-8601=seconds)"
echo "Last arm exit status: $status (143 is a clean stop at the wallclock)"
