#!/bin/bash
# Launch a SGLang rollout server on Slurm for external-rollout training.
#
# Unlike the retrieval server (long-lived, corpus never changes), this SGLang
# server is spun up fresh every training run because the model weights it
# hosts must match the starting checkpoint. The training script (`run_qwen3p5_4B.sh`)
# then pushes weight updates over NCCL via `update_weights_from_distributed`
# each iteration.
#
# Behavior:
#   * Always submits a fresh sbatch job (no reuse — weights would be stale).
#   * Waits for /health to return 200, then writes the hostname to
#     $HOST_FILE so downstream scripts can auto-discover it.
#
# Env vars (all optional):
#   SLIME_HOST_DIR        Where the slime repo lives on the host. Default
#                         /home/hhzhang01/slime.
#   ENROOT_ROOTFS         Name of the pre-imported enroot rootfs. Default slime-test.
#   GENAI_ROOT            Lustre root. Default /genai/fsx-project/hhzhang01.
#   SLURM_ACCOUNT         Slurm account. Default genai_interns.
#   QOS                   Slurm QOS. Default a100_genai_shared.
#   SGLANG_NUM_GPUS       GPUs the sglang server uses. Default 4 (= tp_size).
#   SGLANG_WALLTIME       Server walltime (must outlast the training job).
#                         Default 25:00:00.
#   HF_CKPT               HF checkpoint the sglang server loads. Default
#                         /genai_hh/models/Qwen3.5-4B (in-container path).
#   MEM_FRACTION_STATIC   SGLang KV/param memory fraction. Default 0.85.
#   SERVER_PORT           SGLang HTTP port. Default 30000.
#   SGLANG_JOB_NAME       Slurm job name. Default supo-sglang-<HHMMSS>.
#
# Outputs:
#   * $GENAI_ROOT/logs/sglang-server.hostname  ← "<host>:<port>"
#   * $GENAI_ROOT/logs/sglang-server-<jobname>.log  ← server stdout/stderr
#   * Prints "export SGLANG_SERVER_URL=..." on stdout.
#   * Prints "export SGLANG_JOB_ID=..." on stdout (for cleanup by caller).

set -euo pipefail

SLIME_HOST_DIR="${SLIME_HOST_DIR:-/home/hhzhang01/slime}"
ENROOT_ROOTFS="${ENROOT_ROOTFS:-slime-test}"
GENAI_ROOT="${GENAI_ROOT:-/genai/fsx-project/hhzhang01}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-genai_interns}"
QOS="${QOS:-a100_genai_shared}"
SGLANG_NUM_GPUS="${SGLANG_NUM_GPUS:-4}"
SGLANG_WALLTIME="${SGLANG_WALLTIME:-25:00:00}"
HF_CKPT="${HF_CKPT:-/genai_hh/models/Qwen3.5-4B}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
SERVER_PORT="${SERVER_PORT:-30000}"
# SGLang server context length cap. 128k is the practical upper bound we ever
# want a rollout to reach; smaller than Qwen3.5's 256k `max_position_embeddings`
# so KV cache pre-allocation is halved. Actual per-sample budget is enforced
# separately by `--rollout-max-context-len` in run_qwen3p5_4B.sh (usually 32k
# during debug, tune upward for real runs).
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-131072}"
SGLANG_JOB_NAME="${SGLANG_JOB_NAME:-supo-sglang-$(date +%H%M%S)}"

LOG_DIR="${GENAI_ROOT}/logs"
HOST_FILE="${LOG_DIR}/sglang-server.hostname"
mkdir -p "${LOG_DIR}"

log() { echo "[launch_sglang_server] $*" >&2; }

check_health() {
    curl -sf --max-time 5 "http://$1/health" > /dev/null
}

wait_for_health() {
    local target="$1"
    log "waiting for ${target}/health (up to 15 min)..."
    for i in $(seq 1 90); do
        if check_health "${target}"; then
            log "health OK after ~$((i*10))s"
            return 0
        fi
        sleep 10
    done
    log "ERROR: ${target} never became healthy"
    return 1
}

wait_for_running() {
    local jobid="$1"
    log "waiting for job ${jobid} to reach RUNNING state..."
    for i in $(seq 1 180); do
        local state
        state=$(squeue -h -j "${jobid}" -o '%T' 2>/dev/null || true)
        if [[ "${state}" == "RUNNING" ]]; then
            local nodelist
            nodelist=$(squeue -h -j "${jobid}" -o '%N')
            local host
            host=$(scontrol show hostnames "${nodelist}" | head -n1)
            log "job ${jobid} RUNNING on ${host}"
            printf '%s' "${host}"
            return 0
        fi
        if [[ -z "${state}" ]]; then
            log "ERROR: job ${jobid} disappeared from queue"
            return 1
        fi
        sleep 10
    done
    log "ERROR: job ${jobid} did not reach RUNNING within 30 minutes"
    return 1
}

submit_new_server() {
    log "submitting SGLang server job: ${SGLANG_NUM_GPUS} GPU, walltime ${SGLANG_WALLTIME}, qos=${QOS}"
    # SGLANG_SERVER_LOG_PATH lets launch_all.sh route this into the per-run
    # log dir. Fall back to the legacy top-level path when not set (backwards
    # compatible with direct manual invocations).
    local server_log="${SGLANG_SERVER_LOG_PATH:-${LOG_DIR}/sglang-server-${SGLANG_JOB_NAME}.log}"
    mkdir -p "$(dirname "${server_log}")"

    local wrap_cmd
    wrap_cmd="ENROOT_TEMP_PATH=/dev/shm \
        ENROOT_DATA_PATH=/storage/home/hhzhang01/.local/share/enroot \
        ENROOT_MOUNT_HOME=false \
        enroot start \
            --env PYTHONUNBUFFERED=1 \
            --mount ${SLIME_HOST_DIR}:/slime \
            --mount ${GENAI_ROOT}:/genai_hh \
            ${ENROOT_ROOTFS} \
            bash -c 'python -u -m sglang.launch_server \
                --model-path ${HF_CKPT} \
                --tp ${SGLANG_NUM_GPUS} \
                --host 0.0.0.0 --port ${SERVER_PORT} \
                --context-length ${SGLANG_CONTEXT_LENGTH} \
                --mem-fraction-static ${MEM_FRACTION_STATIC} \
                --disable-custom-all-reduce \
                --trust-remote-code'"

    local jobid
    jobid=$(sbatch \
        --nodes=1 --gpus=${SGLANG_NUM_GPUS} --time="${SGLANG_WALLTIME}" \
        --cpus-per-task=32 --mem=256G \
        --account="${SLURM_ACCOUNT}" --qos="${QOS}" \
        --job-name="${SGLANG_JOB_NAME}" \
        --output="${server_log}" \
        --wrap="${wrap_cmd}" \
        --parsable)
    log "sbatch returned JobId=${jobid}"
    log "server log: ${server_log}"
    printf '%s' "${jobid}"
}

main() {
    local jobid host target
    jobid=$(submit_new_server)
    host=$(wait_for_running "${jobid}")
    target="${host}:${SERVER_PORT}"
    wait_for_health "${target}"
    echo "${target}" > "${HOST_FILE}"
    log "wrote ${HOST_FILE}: ${target}"
    echo "export SGLANG_SERVER_URL=http://${target}"
    echo "export SGLANG_JOB_ID=${jobid}"
}

main "$@"
