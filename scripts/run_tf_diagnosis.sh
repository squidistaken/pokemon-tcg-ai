#!/bin/bash
#
# Submit the Issue-45 transformer diagnosis sweep to Habrok.
#
# One SLURM job per arm, each an A100 job launched exactly the way the first
# transformer run was (slurm-conf/train.sh --config ppo_transformer
# --slurm-config train_gpu), differing only by the +experiment= overlay it
# carries. Every arm names itself after that overlay (wandb.name lives in
# conf/experiment/tf_*.yaml) and shares one wandb.group, so the sweep reads as
# one comparison in the UI.
#
# The device is not passed here: submit.py derives it from the profile's
# gpu_type and appends ++agent.device= itself, and rejects a user override of
# it. Selecting --slurm-config is how you choose cpu vs cuda.
#
#   ./scripts/run_tf_diagnosis.sh                    # submit every arm
#   ./scripts/run_tf_diagnosis.sh --dry-run          # print, submit nothing
#   ./scripts/run_tf_diagnosis.sh reference combined # submit a subset
#   ./scripts/run_tf_diagnosis.sh --group my-sweep   # override the W&B group
#
# IMPORTANT: SLURM jobs read the working tree when they *start*, not when they
# are submitted. Queued arms will pick up any edit made to conf/ or src/ in the
# meantime, which silently breaks the comparison. Submit the sweep, then leave
# the tree alone until every job has started.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG="ppo_transformer"
SLURM_CONFIG="train_gpu"
GROUP="transformer-diagnosis-45"
DRY_RUN=0

# Arms in submission order: the reference first (nothing to compare against
# without it), then the confounded upper-bound probe, then one change each.
ALL_ARMS=(
  reference   # base config, no overlay — the same-budget control
  combined    # every change at once; run this if only one job fits
  subbatch    # sub_batch_size 4096 -> 256 (4 -> 64 optimizer steps/batch)
  capacity    # num_layers 1 -> 2, ff_dim 256 -> 512
  schedule    # lr 3e-4 -> 1e-4 annealed, entropy annealed
  preln       # post-LN -> pre-LN, plus a final LayerNorm
  pooling     # mean readout -> CLS token
  tokens      # per-entity pokemon tokens (10 -> 28 tokens)
  pointer     # PointerPolicyHead over per-option tokens
)

ARMS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --group) GROUP="$2"; shift 2 ;;
    --slurm-config) SLURM_CONFIG="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "ERROR: unknown flag '$1'" >&2; exit 2 ;;
    *) ARMS+=("${1#tf_}"); shift ;;
  esac
done
[ "${#ARMS[@]}" -eq 0 ] && ARMS=("${ALL_ARMS[@]}")

for arm in "${ARMS[@]}"; do
  if [ "$arm" != "reference" ] && [ ! -f "$PROJECT_ROOT/conf/experiment/tf_${arm}.yaml" ]; then
    echo "ERROR: no such arm '$arm' (conf/experiment/tf_${arm}.yaml is missing)" >&2
    echo "       available: ${ALL_ARMS[*]}" >&2
    exit 2
  fi
done

cd "$PROJECT_ROOT"
echo "sweep group : $GROUP"
echo "commit      : $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (+uncommitted changes)')"
echo "arms        : ${ARMS[*]}"
echo

for arm in "${ARMS[@]}"; do
  overrides=("wandb.group=$GROUP")
  if [ "$arm" = "reference" ]; then
    # The base config's wandb.name interpolates from the Hydra choices, which
    # are identical across arms; name it explicitly so it sorts with the rest.
    overrides+=("wandb.name=tf-reference-s42")
  else
    overrides+=("+experiment=tf_${arm}")
  fi

  cmd=(./slurm-conf/train.sh --config "$CONFIG" --slurm-config "$SLURM_CONFIG" "${overrides[@]}")
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '%-11s %s\n' "$arm" "${cmd[*]}"
  else
    printf '%-11s submitting... ' "$arm"
    "${cmd[@]}"
  fi
done

if [ "$DRY_RUN" -eq 0 ]; then
  echo
  echo "queued:"
  squeue -u "$USER" -o '%.12i %.20j %.10T %.10M %R'
  echo
  echo "W&B: https://wandb.ai/pokemon-tcg-ai/pokemon-tcg-ai/groups/$GROUP"
fi
