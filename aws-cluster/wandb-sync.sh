#!/bin/bash
# Sync offline wandb runs to wandb.ai. RUN ONLY ON LOGIN POD (needs X2P proxy).
#
# Usage:
#   bash wandb-sync.sh <RUN_NAME>   # sync all offline runs for a specific RUN_NAME
#   bash wandb-sync.sh              # sync ALL offline runs from every RUN_NAME
#
# Assumes wandb-args.sh wrote offline runs under /data/wandb/<RUN_NAME>/wandb/
# (in-container path) which maps to /genai/fsx-project/hhzhang01/wandb/<RUN_NAME>/wandb/
# (host path).
#
# The API key is read from /home/hhzhang01/.wandb-key (chmod 600). Never
# passed on the command line or exported into shell history.
#
# Uses `python3 -m wandb sync --append`. Slime now creates ONE offline dir per
# Ray actor (RolloutManager + Megatron), each a distinct wandb run joined by
# group=RUN_NAME. Cloud UI auto-groups them, no shared run_id / no _step
# collision. See slime/utils/wandb_utils.py + aws-cluster/README.md for
# the multi-run-per-group design.
#
# Why --append: this script is called repeatedly (every 5 min) on the same
# still-growing offline runs. Plain `wandb sync` is a one-shot uploader — it
# captures a run at its current EOF and then CLOSES the cloud run, so a live
# run freezes at whatever step the first sync caught (dead runs look fine only
# because their file no longer grows). --append (resume="allow") resumes each
# cloud run and uploads just the new steps, so incremental/live sync works and
# resumed runs keep advancing. Idempotent: re-syncing a complete run pushes
# nothing new.

set -eu

RUN_NAME="${1:-}"
WANDB_HOST_DIR=/genai/fsx-project/hhzhang01/wandb
KEY_FILE=/home/hhzhang01/.wandb-key

if [[ ! -r "$KEY_FILE" ]]; then
  echo "wandb-sync.sh: cannot read $KEY_FILE" >&2
  exit 1
fi

export WANDB_API_KEY="$(cat "$KEY_FILE")"

shopt -s nullglob
if [[ -n "$RUN_NAME" ]]; then
  matches=( "$WANDB_HOST_DIR/$RUN_NAME"/wandb/offline-run-* )
else
  matches=( "$WANDB_HOST_DIR"/*/wandb/offline-run-* )
fi
shopt -u nullglob

if [[ ${#matches[@]} -eq 0 ]]; then
  echo "wandb-sync.sh: no offline runs matching '${RUN_NAME:-<all>}' under $WANDB_HOST_DIR/" >&2
  # Non-fatal: outer wrapper shouldn't fail if training crashed before wandb wrote anything.
  exit 0
fi

echo "wandb-sync.sh: syncing ${#matches[@]} offline dir(s) as independent runs..."
python3 -m wandb sync --append "${matches[@]}"
