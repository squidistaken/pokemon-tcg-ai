#!/bin/bash
# Run a training job and restart it from its own state whenever it dies.
#
# WHY THIS EXISTS
#
# Trainer already survives a dead worker pool in-process (collector.max_restarts).
# It cannot survive a dead CUDA context, and that is what killed
# mlp-pointer-seatsplit-15m-s42 at 3.78M of 15M frames: the WSL paging layer
# logged `dxgk: dxgkio_make_resident: Ioctl failed: -12` and every CUDA call
# after it raised "CUDA driver error: device not ready". Once the context is
# gone nothing in the process can be salvaged -- the model, the optimizer and
# the collector all live on that device. The only recovery is a new process.
#
# agent.cuda_memory_fraction makes that particular failure far less likely by
# capping what the allocator may reserve, so memory pressure raises a catchable
# OutOfMemoryError instead of reaching the driver. This script covers the rest:
# any crash at all costs at most train.train_state_interval frames, because the
# next attempt resumes from the rolling train_state.pt with Adam's moments and
# the curriculum intact.
#
# WHAT A RESUME CARRIES
#
#   train_state.pt          weights + optimizer moments + frame count
#   curriculum/*.pt         matchup scores, so the level buffer is not relearned
#   checkpoints/            the league, which refills from the loaded weights
#
# USAGE
#
#   scripts/train_supervised.sh --run-dir outputs/my-run --total-frames 15000000 \
#     -- --config-name ppo_selfplay_multideck agent.device=cuda ...
#
# Everything after `--` is passed to `python -m src.train` untouched, except
# that this script owns hydra.run.dir, collector.total_frames,
# train.resume_state and env.curriculum.init_state -- do not set those yourself.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# The repo's own modules (submission/, src/, main.py) collide with same-named
# top-level modules on an inherited PYTHONPATH, so the run gets a clean one.
export PYTHONPATH=

# GAE runs the critic over the whole collected batch in one forward and the
# epoch loop clones the batch once per epoch, so the allocator sees a handful of
# multi-GB transients against steady small ones -- the fragmentation case
# expandable_segments exists for.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

RUN_DIR=""
TOTAL_FRAMES=""
MAX_ATTEMPTS=10

while [ $# -gt 0 ]; do
  case "$1" in
    --run-dir)      RUN_DIR="$2";      shift 2 ;;
    --total-frames) TOTAL_FRAMES="$2"; shift 2 ;;
    --attempts)     MAX_ATTEMPTS="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "ERROR: unknown option '$1'; training overrides go after '--'." >&2; exit 2 ;;
  esac
done

if [ -z "$RUN_DIR" ] || [ -z "$TOTAL_FRAMES" ]; then
  echo "usage: $0 --run-dir DIR --total-frames N [--attempts N] -- <src.train overrides...>" >&2
  exit 2
fi
if [ $# -eq 0 ]; then
  echo "ERROR: no training overrides given after '--'." >&2
  exit 2
fi

# Create before absolutizing: cd into a path that does not exist yet fails, and
# a failed command substitution inside an assignment is not caught by set -e --
# it just yields an empty string, which turned the run directory into an
# absolute path at the filesystem root.
mkdir -p "$RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
STATE="$RUN_DIR/train_state.pt"

# Frames recorded in a train_state.pt, or 0 when there is no usable state yet.
# A resume that cannot read its own frame count would restart the run at zero
# and silently overwrite the league, so an unreadable file counts as no state.
state_frames() {
  if [ ! -f "$STATE" ]; then
    echo 0
    return
  fi
  uv run --frozen --no-sync python - "$STATE" <<'PY' 2>/dev/null || echo 0
import sys

import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
frames = payload.get("frames") if isinstance(payload, dict) else None
print(int(frames) if frames else 0)
PY
}

# Newest curriculum dump, so a restart keeps the matchup scores it paid for.
#
# The directory only exists once a curriculum run has written one, so the guard
# is load-bearing rather than defensive: `find` on a missing path exits 1, and
# under `set -e` with `pipefail` that status propagates out of the command
# substitution and kills the supervisor. It did, silently, on the first resume
# of a run with `env.curriculum.enabled=false` -- after the "continuing it"
# message and before the first attempt, so the run simply stopped with no error
# and an exit code nobody was looking at.
latest_curriculum() {
  [ -d "$RUN_DIR/curriculum" ] || return 0
  find "$RUN_DIR/curriculum" -name 'curriculum_*.pt' 2>/dev/null | sort | tail -1
}

DONE_FRAMES="$(state_frames)"
if [ "$DONE_FRAMES" -gt 0 ]; then
  echo "Found existing state at $DONE_FRAMES frames in $RUN_DIR; continuing it."
fi

attempt=0
while [ "$DONE_FRAMES" -lt "$TOTAL_FRAMES" ]; do
  attempt=$((attempt + 1))
  if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
    echo "ERROR: spent the $MAX_ATTEMPTS-attempt budget at $DONE_FRAMES/$TOTAL_FRAMES frames." >&2
    exit 1
  fi

  remaining=$((TOTAL_FRAMES - DONE_FRAMES))
  args=(
    hydra.run.dir="$RUN_DIR"
    collector.total_frames="$remaining"
  )
  if [ "$DONE_FRAMES" -gt 0 ]; then
    args+=(train.resume_state="$STATE")
    curriculum="$(latest_curriculum)"
    if [ -n "$curriculum" ]; then
      args+=(env.curriculum.init_state="$curriculum")
    fi
  fi

  echo "=== attempt $attempt/$MAX_ATTEMPTS: $DONE_FRAMES -> $TOTAL_FRAMES frames ($remaining to collect) ==="
  status=0
  # Caller's arguments first, ours appended. Hydra parses overrides as a single
  # positional group, so a --config-name flag arriving after an override splits
  # them in two and argparse rejects the second group. The caller's flags lead,
  # so appending keeps every flag ahead of every override.
  uv run --frozen --no-sync python -m src.train "$@" "${args[@]}" || status=$?
  if [ "$status" -eq 0 ]; then
    echo "Training finished cleanly on attempt $attempt."
    exit 0
  fi

  # 128+N is death by signal N: 130 is Ctrl-C, 143 is SIGTERM from a kill,
  # a timeout, or a session teardown. Someone stopping the run is not a crash.
  # Restarting into it would make the run unstoppable, and reporting it as a
  # failure that a restart cannot fix is simply wrong -- nothing is broken.
  if [ "$status" -eq 130 ] || [ "$status" -eq 143 ]; then
    signal="interrupt (Ctrl-C)"
    [ "$status" -eq 143 ] && signal="termination signal"
    progressed="$(state_frames)"
    echo "Stopped by $signal at $progressed frames; not restarting." >&2
    if [ "$progressed" -gt 0 ]; then
      echo "       Continue where it left off: RESUME=1 <your launcher>" >&2
    fi
    exit "$status"
  fi

  progressed="$(state_frames)"
  echo "Attempt $attempt died with status $status at $progressed frames."
  if [ "$progressed" -le "$DONE_FRAMES" ]; then
    # Restarting into the same state would reproduce the same failure forever,
    # which is what a crash on startup (a bad config, a missing deck corpus, a
    # checkpoint that will not load) looks like from here.
    echo "ERROR: no frames survived this attempt; the failure is not one a restart fixes." >&2
    exit "$status"
  fi
  DONE_FRAMES="$progressed"
done

echo "Reached $DONE_FRAMES/$TOTAL_FRAMES frames."
