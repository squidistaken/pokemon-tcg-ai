#!/bin/bash

set -euo pipefail

MODE="both"
DRY_RUN=false
STORAGE_ROOT="${FINETUNE_STORAGE_ROOT:-}"
FORWARDED=()

usage() {
  cat <<'EOF'
Usage: run_fixed_deck_finetune_slurm.sh [--dry-run] [--mode frozen|refresh|both] [--storage-root PATH] [HYDRA_OVERRIDE ...]

Submits the 30,015,488-frame fixed-deck fine-tuning experiment through the
train_gpu_rtx Slurm profile. Remaining arguments are forwarded to Hydra after
the mode-specific defaults. --storage-root PATH reads decks and baseline
checkpoints from PATH and writes run outputs beneath it.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --mode)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --mode needs frozen, refresh, or both" >&2
        exit 2
      fi
      MODE="$2"
      shift 2
      ;;
    --mode=*)
      MODE="${1#--mode=}"
      shift
      ;;
    --storage-root)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --storage-root needs a path" >&2
        exit 2
      fi
      STORAGE_ROOT="$2"
      shift 2
      ;;
    --storage-root=*)
      STORAGE_ROOT="${1#--storage-root=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      FORWARDED+=("$1")
      shift
      ;;
  esac
done

case "$MODE" in
  frozen)
    MODES=(frozen)
    ;;
  refresh)
    MODES=(refresh)
    ;;
  both)
    MODES=(frozen refresh)
    ;;
  *)
    echo "ERROR: --mode must be frozen, refresh, or both; got '$MODE'" >&2
    exit 2
    ;;
esac

GROUP="${FINETUNE_WANDB_GROUP:-fixed-deck-finetune-30m-s42}"
if [ -n "$STORAGE_ROOT" ]; then
  OUTPUT_ROOT="${FINETUNE_OUTPUT_ROOT:-$STORAGE_ROOT/outputs/fixed-deck-finetune-30m-s42}"
  STORAGE_OVERRIDES=(
    "paths.data_dir=$STORAGE_ROOT/decks"
    "finetune_checkpoint_dir=$STORAGE_ROOT/checkpoints/baseline-training-checkpoints"
  )
else
  OUTPUT_ROOT="${FINETUNE_OUTPUT_ROOT:-outputs/fixed-deck-finetune-30m-s42}"
  STORAGE_OVERRIDES=()
fi

for opponent_mode in "${MODES[@]}"; do
  COMMAND=(
    ./slurm-conf/train.sh
    --config ppo_fixed_deck_finetune
    --slurm-config train_gpu_rtx
  )
  if [ "$DRY_RUN" = true ]; then
    COMMAND+=(--dry-run)
  fi
  COMMAND+=(
    "train.opponent_pool_mode=$opponent_mode"
    "wandb.group=$GROUP"
    "wandb.name=fixed-deck-${opponent_mode}-30m-s42"
    "paths.output_dir=${OUTPUT_ROOT}/${opponent_mode}"
    "${STORAGE_OVERRIDES[@]}"
    "${FORWARDED[@]}"
  )
  "${COMMAND[@]}"
done
