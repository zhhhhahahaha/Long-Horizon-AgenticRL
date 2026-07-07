#!/bin/bash
# One-button orchestrator for the SUPO/BC+ training pipeline.
#
# Sequence:
#   1. Ensure a healthy search server exists whose Slurm walltime outlasts the
#      training job by at least SEARCH_BUFFER_HOURS (default 4).
#   2. Fire the training srun via run_qwen3p5_4B.sh in the background.
#   3. Poll wandb-sync.sh every WANDB_SYNC_INTERVAL_SEC (default 300 = 5 min)
#      while the training srun is alive. This gives near-live wandb curves
#      instead of a bulk-upload only after srun exits.
#   4. When srun exits, do one final sync to catch the tail.
#
# Env vars (all optional):
#   TRAIN_WALLTIME_HOURS      Wallclock hours for the training job. Default 24.
#                             Search server must have TRAIN_WALLTIME_HOURS +
#                             SEARCH_BUFFER_HOURS hours left.
#   SEARCH_BUFFER_HOURS       Slack between training end and server end.
#                             Default 4.
#   TRAIN_WALLTIME            Slurm --time value. Default HH:00:00 derived from
#                             TRAIN_WALLTIME_HOURS (e.g. 24 -> 24:00:00).
#   QOS                       Slurm QOS. Default a100_genai_shared.
#   SLURM_ACCOUNT             Slurm account. Default genai_interns.
#   WANDB_SYNC_INTERVAL_SEC   Seconds between mid-run wandb syncs. Default 300.
#
# Required env:
#   LLAMA_API_KEY   Judge auth (LLM|... key).

set -euo pipefail

: "${LLAMA_API_KEY:?LLAMA_API_KEY must be set (LLM|... judge key)}"

TRAIN_WALLTIME_HOURS="${TRAIN_WALLTIME_HOURS:-24}"
SEARCH_BUFFER_HOURS="${SEARCH_BUFFER_HOURS:-4}"
TRAIN_WALLTIME="${TRAIN_WALLTIME:-${TRAIN_WALLTIME_HOURS}:00:00}"
QOS="${QOS:-a100_genai_shared}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-genai_interns}"
WANDB_SYNC_INTERVAL_SEC="${WANDB_SYNC_INTERVAL_SEC:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Precompute RUN_NAME so we know which offline-run dir to sync while training
# is still going. Exported so run_qwen3p5_4B.sh reuses it instead of computing
# its own timestamp.
export RUN_NAME="${RUN_NAME:-supo-bcplus-qwen3p5-4b-smoke-$(date +%Y%m%d-%H%M)}"
echo "[launch_all] RUN_NAME=${RUN_NAME}"

# Step 1: ensure search server is up with enough runway.
export MIN_HOURS_REMAINING="$((TRAIN_WALLTIME_HOURS + SEARCH_BUFFER_HOURS))"
export QOS
export SLURM_ACCOUNT
echo "[launch_all] ensuring search server has >= ${MIN_HOURS_REMAINING}h left"
bash "${SCRIPT_DIR}/launch_search_server.sh"

HOST_FILE=/genai/fsx-project/hhzhang01/logs/search-server.hostname
if [[ ! -f "${HOST_FILE}" ]]; then
    echo "[launch_all] ERROR: expected ${HOST_FILE} after launch_search_server.sh" >&2
    exit 1
fi
export LOCAL_SEARCH_URL="http://$(cat "${HOST_FILE}")"
echo "[launch_all] LOCAL_SEARCH_URL=${LOCAL_SEARCH_URL}"

# Step 2: submit training in the background. run_qwen3p5_4B.sh's outer part
# execs srun, so its PID = srun's PID; wait/kill on it maps to the job.
export TRAIN_WALLTIME
export QOS
export SLURM_ACCOUNT
echo "[launch_all] submitting training (walltime=${TRAIN_WALLTIME}, qos=${QOS})"
bash "${SCRIPT_DIR}/run_qwen3p5_4B.sh" &
TRAIN_PID=$!
echo "[launch_all] training PID (login-side srun): ${TRAIN_PID}"

# Step 3: periodic wandb sync while training is alive.
SYNC_SCRIPT="${SLIME_ROOT}/aws-cluster/wandb-sync.sh"
echo "[launch_all] periodic wandb sync every ${WANDB_SYNC_INTERVAL_SEC}s for ${RUN_NAME}"
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    sleep "${WANDB_SYNC_INTERVAL_SEC}"
    # Suppress non-fatal complaints (e.g. no offline-run dir yet in first
    # minute of training) but let real errors surface.
    if ! bash "${SYNC_SCRIPT}" "${RUN_NAME}" 2>&1 | sed 's/^/[wandb-sync] /'; then
        echo "[launch_all] mid-run wandb-sync returned non-zero; continuing"
    fi
done

# Wait for the training script to fully exit and capture its status.
wait "${TRAIN_PID}"
TRAIN_STATUS=$?
echo "[launch_all] training finished with status ${TRAIN_STATUS}"

# Step 4: final sync to catch anything flushed after the last mid-run tick.
echo "[launch_all] final wandb sync for ${RUN_NAME}"
bash "${SYNC_SCRIPT}" "${RUN_NAME}" 2>&1 | sed 's/^/[wandb-sync] /'

exit "${TRAIN_STATUS}"
