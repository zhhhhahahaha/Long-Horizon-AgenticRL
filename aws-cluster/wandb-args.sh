# Sourced by every slime run script INSIDE THE CONTAINER. Requires RUN_NAME set.
# Offline mode is mandatory: compute nodes have no external network egress, so
# wandb.log() cannot reach wandb.ai in real time. Runs are written to
# /data/wandb/<RUN_NAME>/ (Lustre) and uploaded later by wandb-sync.sh from
# the login pod, which globs offline-run-* under the RUN_NAME subdir.

: "${RUN_NAME:?RUN_NAME must be set before sourcing wandb-args.sh}"

# Per-RUN_NAME subdir keeps offline runs from different launches isolated,
# which makes sync trivial (glob just this subdir).
mkdir -p "/data/wandb/${RUN_NAME}"

WANDB_ARGS=(
  --use-wandb
  --wandb-mode offline
  --wandb-project slime-math-sanity-check
  --wandb-group "${RUN_NAME}"
  --wandb-dir "/data/wandb/${RUN_NAME}"
  # --wandb-key intentionally omitted (offline mode never calls wandb.login)
  # --wandb-team omitted -> defaults to personal wandb.ai account
  # --wandb-random-suffix left at default True (safe re-run collision protection)
)
