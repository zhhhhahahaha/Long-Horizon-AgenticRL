#!/bin/bash
# One-button orchestrator for the SUPO/BC+ training pipeline (multi-node).
#
# Sequence:
#   1. Ensure a healthy long-lived retrieval search server (idempotent reuse).
#   2. Sbatch a fresh SGLang rollout server (external — separate slurm job,
#      typically 4 GPU on a different node from the actor). Weight sync
#      happens over NCCL each iteration.
#   3. Fire the training srun via run_qwen3p5_4B.sh in the background.
#   4. Poll wandb-sync.sh every WANDB_SYNC_INTERVAL_SEC while training runs.
#   5. When training exits, scancel the SGLang server (weights are stale)
#      and do one final wandb sync.
#
# All logs for a single run live in $RUN_LOG_DIR (see below) — one directory
# per RUN_NAME, plus a `latest` symlink pointing at the most recent one.
#
# RUN_NAME layout (Plan C — human-labelled with optional suffix):
#   Default: ${MODEL_TAG}-${VARIANT_TAG}-YYYYMMDD-HHMM[-${RUN_TAG}]
#   Example: qwen3p5-4b-cp2-tp4-20260707-1930-oom-hunt
#
# Env vars (all optional; sensible defaults):
#   MODEL_TAG                 Model identifier for RUN_NAME. Default qwen3p5-4b.
#                             Change when swapping model (e.g. qwen3p5-9b).
#   VARIANT_TAG               Parallelism/config identifier for RUN_NAME.
#                             Default cp2-tp4. Change when adjusting TP/CP/etc.
#   RUN_TAG                   Optional free-form suffix appended after the
#                             timestamp (e.g. RUN_TAG=oom-hunt).
#   RUN_NAME                  Overrides the whole computed name. If you set
#                             this, MODEL_TAG/VARIANT_TAG/RUN_TAG are ignored.
#   TRAIN_WALLTIME_HOURS      Wallclock hours for the training job. Default 24.
#   SEARCH_BUFFER_HOURS       Slack between training end and search-server end.
#                             Default 4.
#   TRAIN_WALLTIME            Slurm --time value. Default HH:00:00 derived from
#                             TRAIN_WALLTIME_HOURS.
#   SGLANG_WALLTIME           Slurm --time value for the SGLang server job.
#                             Default = TRAIN_WALLTIME + 1h buffer.
#   SGLANG_NUM_GPUS           SGLang server tp size / GPU count. Default 4.
#   QOS                       Slurm QOS. Default a100_genai_shared.
#   SLURM_ACCOUNT             Slurm account. Default genai_interns.
#   WANDB_SYNC_INTERVAL_SEC   Seconds between mid-run wandb syncs. Default 300.
#
# Required env:
#   LLAMA_API_KEY   Judge auth (LLM|... key).
#
# Outputs:
#   $RUN_LOG_DIR/launch_all.log       — orchestrator log (this script's stdout)
#   $RUN_LOG_DIR/sglang-server.log    — SGLang rollout server stdout/stderr
#   $RUN_LOG_DIR/train.log            — training srun stdout/stderr
#   $RUN_LOG_DIR/info.txt             — run metadata (hostnames, jobids, wandb URL)
#   $RUNS_ROOT/latest -> $RUN_LOG_DIR — symlink to the most recent run

set -euo pipefail

: "${LLAMA_API_KEY:?LLAMA_API_KEY must be set (LLM|... judge key)}"

MODEL_TAG="${MODEL_TAG:-qwen3p5-4b}"
VARIANT_TAG="${VARIANT_TAG:-cp2-tp4}"
RUN_TAG="${RUN_TAG:-}"

TRAIN_WALLTIME_HOURS="${TRAIN_WALLTIME_HOURS:-24}"
SEARCH_BUFFER_HOURS="${SEARCH_BUFFER_HOURS:-4}"
TRAIN_WALLTIME="${TRAIN_WALLTIME:-${TRAIN_WALLTIME_HOURS}:00:00}"
SGLANG_WALLTIME="${SGLANG_WALLTIME:-$((TRAIN_WALLTIME_HOURS + 1)):00:00}"
SGLANG_NUM_GPUS="${SGLANG_NUM_GPUS:-4}"
QOS="${QOS:-a100_genai_shared}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-genai_interns}"
WANDB_SYNC_INTERVAL_SEC="${WANDB_SYNC_INTERVAL_SEC:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Precompute RUN_NAME so we know which offline-run dir to sync while training
# is still going. Exported so run_qwen3p5_4B.sh reuses it.
if [[ -z "${RUN_NAME:-}" ]]; then
    _tag_suffix="${RUN_TAG:+-${RUN_TAG}}"
    export RUN_NAME="${MODEL_TAG}-${VARIANT_TAG}-$(date +%Y%m%d-%H%M)${_tag_suffix}"
fi
echo "[launch_all] RUN_NAME=${RUN_NAME}"

# Per-run log directory + `latest` symlink for easy tailing.
RUNS_ROOT="/genai/fsx-project/hhzhang01/logs/runs"
export RUN_LOG_DIR="${RUNS_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_LOG_DIR}"
ln -sfn "${RUN_LOG_DIR}" "${RUNS_ROOT}/latest"
echo "[launch_all] RUN_LOG_DIR=${RUN_LOG_DIR}"
echo "[launch_all] follow live: tail -f ${RUNS_ROOT}/latest/train.log"

# Re-exec so all stdout/stderr from here on is captured in launch_all.log,
# while also being echoed to the caller's terminal for interactive visibility.
if [[ -z "${_LAUNCH_ALL_TEE_STARTED:-}" ]]; then
    export _LAUNCH_ALL_TEE_STARTED=1
    exec > >(tee -a "${RUN_LOG_DIR}/launch_all.log") 2>&1
fi

# Also mirror to the legacy top-level log so old muscle memory still works.
echo "[launch_all] $(date -u +%Y-%m-%dT%H:%M:%SZ) begin run ${RUN_NAME}"

# Step 1: ensure retrieval search server is up with enough runway.
export MIN_HOURS_REMAINING="$((TRAIN_WALLTIME_HOURS + SEARCH_BUFFER_HOURS))"
export QOS
export SLURM_ACCOUNT
echo "[launch_all] ensuring search server has >= ${MIN_HOURS_REMAINING}h left"
bash "${SCRIPT_DIR}/launch_search_server.sh"

SEARCH_HOST_FILE=/genai/fsx-project/hhzhang01/logs/search-server.hostname
if [[ ! -f "${SEARCH_HOST_FILE}" ]]; then
    echo "[launch_all] ERROR: expected ${SEARCH_HOST_FILE} after launch_search_server.sh" >&2
    exit 1
fi
export LOCAL_SEARCH_URL="http://$(cat "${SEARCH_HOST_FILE}")"
echo "[launch_all] LOCAL_SEARCH_URL=${LOCAL_SEARCH_URL}"

# Step 2: sbatch a fresh SGLang rollout server. Always fresh because weights
# match the starting checkpoint and get updated each iter via NCCL.
export SGLANG_WALLTIME
export SGLANG_NUM_GPUS
export SGLANG_JOB_NAME="sgl-${RUN_NAME}"
# Route sglang server log into the per-run dir (launch_sglang_server.sh
# respects SGLANG_SERVER_LOG_PATH; falls back to its own default otherwise).
export SGLANG_SERVER_LOG_PATH="${RUN_LOG_DIR}/sglang-server.log"
echo "[launch_all] launching sglang rollout server (${SGLANG_NUM_GPUS} GPU, walltime ${SGLANG_WALLTIME})"
SGLANG_LAUNCH_OUT=$(bash "${SCRIPT_DIR}/launch_sglang_server.sh")
eval "${SGLANG_LAUNCH_OUT}"  # exports SGLANG_SERVER_URL + SGLANG_JOB_ID
echo "[launch_all] SGLANG_SERVER_URL=${SGLANG_SERVER_URL}, SGLANG_JOB_ID=${SGLANG_JOB_ID}"

# Trap ensures we scancel the sglang server no matter how launch_all exits.
cleanup_sglang() {
    if [[ -n "${SGLANG_JOB_ID:-}" ]]; then
        echo "[launch_all] cleanup: scancel sglang server job ${SGLANG_JOB_ID}"
        scancel "${SGLANG_JOB_ID}" 2>/dev/null || true
    fi
}
trap cleanup_sglang EXIT

# Dump run metadata for later grep-ability.
{
    echo "RUN_NAME=${RUN_NAME}"
    echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "MODEL_TAG=${MODEL_TAG}  VARIANT_TAG=${VARIANT_TAG}  RUN_TAG=${RUN_TAG}"
    echo ""
    echo "-- Slurm jobs --"
    SEARCH_JOB_ID=$(squeue -u "${USER}" -h -n supo-search-server -o '%i' | head -n1 || true)
    echo "Search server: ${LOCAL_SEARCH_URL} (JobId ${SEARCH_JOB_ID})"
    echo "SGLang server: ${SGLANG_SERVER_URL} (JobId ${SGLANG_JOB_ID})"
    echo "Training:      submitted below (see squeue -u ${USER} -n ${RUN_NAME})"
    echo ""
    echo "-- Timings --"
    echo "TRAIN_WALLTIME=${TRAIN_WALLTIME}  SGLANG_WALLTIME=${SGLANG_WALLTIME}"
    echo "QOS=${QOS}  SLURM_ACCOUNT=${SLURM_ACCOUNT}"
    echo ""
    echo "-- Wandb --"
    echo "Project: slime-math-sanity-check"
    echo "Group:   ${RUN_NAME}"
    echo "Offline: /genai/fsx-project/hhzhang01/wandb/${RUN_NAME}"
    echo "Web:     https://wandb.ai/zhhhhahahaha/slime-math-sanity-check (filter by group=${RUN_NAME})"
    echo ""
    echo "-- Log files (this directory) --"
    echo "launch_all.log    orchestrator (this script's stdout)"
    echo "sglang-server.log SGLang rollout server"
    echo "train.log         Ray + Megatron training srun output"
} > "${RUN_LOG_DIR}/info.txt"
echo "[launch_all] wrote ${RUN_LOG_DIR}/info.txt"

# Step 3: submit training in the background. run_qwen3p5_4B.sh's outer part
# execs srun, so its PID = srun's PID.
export TRAIN_WALLTIME
export QOS
export SLURM_ACCOUNT
export TRAIN_LOG_PATH="${RUN_LOG_DIR}/train.log"
echo "[launch_all] submitting training (walltime=${TRAIN_WALLTIME}, qos=${QOS})"
bash "${SCRIPT_DIR}/run_qwen3p5_4B.sh" &
TRAIN_PID=$!
echo "[launch_all] training PID (login-side srun): ${TRAIN_PID}"

# Step 4: periodic wandb sync while training is alive.
SYNC_SCRIPT="${SLIME_ROOT}/aws-cluster/wandb-sync.sh"
echo "[launch_all] periodic wandb sync every ${WANDB_SYNC_INTERVAL_SEC}s for ${RUN_NAME}"
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    sleep "${WANDB_SYNC_INTERVAL_SEC}"
    if ! bash "${SYNC_SCRIPT}" "${RUN_NAME}" 2>&1 | sed 's/^/[wandb-sync] /'; then
        echo "[launch_all] mid-run wandb-sync returned non-zero; continuing"
    fi
done

wait "${TRAIN_PID}"
TRAIN_STATUS=$?
echo "[launch_all] training finished with status ${TRAIN_STATUS}"

# Step 5: final sync.
echo "[launch_all] final wandb sync for ${RUN_NAME}"
bash "${SYNC_SCRIPT}" "${RUN_NAME}" 2>&1 | sed 's/^/[wandb-sync] /'

exit "${TRAIN_STATUS}"
