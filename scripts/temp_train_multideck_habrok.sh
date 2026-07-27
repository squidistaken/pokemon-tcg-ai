#!/bin/bash
# =============================================================================
# Habrok (University of Groningen) SLURM batch script for the multi-deck
# self-play PPO run. Submit from the repo root:
#
#     sbatch scripts/train_multideck_habrok.sh
#
# Quick sanity submit (tiny budget, fast queue) — override on the command line:
#     sbatch --partition=short --time=00:20:00 \
#            scripts/train_multideck_habrok.sh 20000
#
# The first positional arg overrides TOTAL_FRAMES (default below).
# =============================================================================

#SBATCH --job-name=ptcg-multideck
#SBATCH --partition=regular          # CPU node; engine steps on CPU (see notes)
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # one training process; parallelism is the env workers
#SBATCH --cpus-per-task=16           # must match env.num_workers below
#SBATCH --mem=32G                    # ~14G used at 16 workers; headroom avoids the OOM we hit locally
#SBATCH --time=24:00:00              # raise for larger TOTAL_FRAMES
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# --- GPU is intentionally OFF: the model is a small MLP and the engine is
# --- CPU-bound, so a GPU barely helps. To use one (e.g. once you switch to the
# --- transformer backbone), submit to a GPU partition and add device=cuda:
# #SBATCH --partition=gpu
# #SBATCH --gpus-per-node=1

set -euo pipefail

# ----------------------------------------------------------------------------
# Run parameters (override TOTAL_FRAMES as the first positional arg)
# ----------------------------------------------------------------------------
TOTAL_FRAMES="${1:-2000000}"         # 2M is a solid real run; scale up as needed
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-16}"
# Any args after TOTAL_FRAMES are forwarded verbatim as extra Hydra overrides,
# so one script launches every multideck variant, e.g.:
#     sbatch scripts/train_multideck_habrok.sh 5000000 env.deck_weighting=winrate
EXTRA_OVERRIDES=("${@:2}")

# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
# Run from the directory the job was submitted from (the repo root).
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# HPC modules. uv manages the exact Python (3.12+) itself, so we only need a
# recent toolchain available. Adjust the module name to what `module avail`
# shows on Habrok; uv itself is installed per-user (see the check below).
module purge
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null || echo "note: Python module not loaded; relying on uv-managed Python"

# uv is the project's package manager (uv.lock is committed). Install it once
# with:  curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found on PATH. Install it (see comment above)."; exit 1; }

# CRUCIAL on a many-core node: pin intra-op math threads to 1 so each of the
# NUM_WORKERS env processes stays single-threaded. Otherwise every worker spawns
# a full BLAS/OMP thread pool and they thrash each other (the parallelism we
# want is across env workers, not inside each one).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Weights & Biases. Online mode streams the run live for real-time logging.
# NOTE: this needs outbound internet from the compute node -- Habrok nodes often
# reach the web only through an HTTP proxy, so if the run hangs at "wandb: "
# start-up, either export the proxy (e.g. `export HTTPS_PROXY=...` per RUG docs)
# or fall back to WANDB_MODE=offline and `wandb sync` from a login node (the
# command is printed at the end). Set your key via the environment
# (e.g. `export WANDB_API_KEY=...` in ~/.bashrc or a sourced secrets file).
export WANDB_DIR="${TMPDIR:-/scratch/$USER}/wandb"
mkdir -p "$WANDB_DIR"
WANDB_MODE=online

# Sync the environment against the lockfile (fast if the cache is warm).
uv sync --frozen

# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------
echo "=================================================================="
echo "job:        ${SLURM_JOB_NAME:-local} (${SLURM_JOB_ID:-n/a})"
echo "node:       $(hostname)"
echo "cpus/task:  ${NUM_WORKERS}"
echo "mem:        ${SLURM_MEM_PER_NODE:-n/a} MB"
echo "total_frames: ${TOTAL_FRAMES}"
echo "python:     $(uv run python -c 'import sys; print(sys.version.split()[0])')"
echo "started:    $(date)"
echo "=================================================================="

# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------
# env=multideck already sets the corpus pool, independent training matchup,
# mirror eval matchup, and a 20% held-out split for the generalization curve.
srun uv run python -m src.train \
    agent=ppo \
    train=ppo_selfplay \
    env=multideck \
    env.num_workers="${NUM_WORKERS}" \
    env.mp_start_method=fork \
    collector.total_frames="${TOTAL_FRAMES}" \
    train.snapshot_interval=50000 \
    train.eval_interval=50000 \
    train.eval_episodes=150 \
    wandb.mode="${WANDB_MODE}" \
    wandb.group=multideck-selfplay-habrok \
    ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}

echo "=================================================================="
echo "finished:   $(date)"
if [ "${WANDB_MODE}" = "offline" ]; then
    echo "W&B recorded offline. From a login node with internet, sync with:"
    echo "    WANDB_DIR=${WANDB_DIR} uv run wandb sync ${WANDB_DIR}/offline-run-*"
fi
echo "=================================================================="
