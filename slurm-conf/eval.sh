#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -eq 0 ]; then
  echo "Usage: eval.sh --config NAME --slurm-config NAME [HYDRA_OVERRIDE ...]" >&2
  exit 2
fi

exec "$SCRIPT_DIR/train.sh" --module src.eval_deck_field "$@"
