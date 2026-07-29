#!/bin/bash
# Launch the BrowseComp-Plus retrieval server on Slurm (idempotent).
#
# Behavior:
#   * Looks for a running Slurm job named "supo-search-server" owned by $USER.
#   * If none exists → sbatch a fresh 7-day 1-GPU job.
#   * If one exists → check TimeLeft; reuse if it has >= MIN_HOURS_REMAINING
#     hours left, otherwise scancel + resubmit (with a loud warning first).
#   * Waits for /health to return 200, then writes the hostname to
#     $HOST_FILE so downstream scripts can auto-discover it.
#
# Env vars (all optional):
#   MIN_HOURS_REMAINING   Minimum hours the server must have left to be
#                         considered "fresh enough" for reuse. Default 48.
#   SLIME_HOST_DIR        Where the slime repo lives on the host. Default
#                         /home/hhzhang01/slime.
#   ENROOT_ROOTFS         Name of the pre-imported enroot rootfs (see
#                         `enroot list`). Default slime-test. The login pod
#                         has no squashfuse, so we cannot mount a .sqsh
#                         directly — must import once and use the rootfs.
#   GENAI_ROOT            Lustre root that holds models/ and datasets/.
#                         Default /genai/fsx-project/hhzhang01.
#   SLURM_ACCOUNT         Slurm account. Default genai_interns (needed to
#                         access a100_* QOS on this cluster).
#   QOS                   Slurm QOS. Default a100_dev (partition is inferred
#                         from QOS prefix on this cluster). a100_dev has high
#                         priority and doesn't consume the a100_*_high quota,
#                         and it allows the full 7-day walltime — a good home
#                         for this long-lived retrieval service.
#   SERVER_PORT           Port the retrieval server binds to. Default 8000.
#   SEARCH_JOB_NAME       Slurm job name (used for squeue matching too).
#                         Default supo-search-server.
#
# Outputs:
#   * $GENAI_ROOT/logs/search-server.hostname  ← one line: "<hostname>:<port>"
#   * $GENAI_ROOT/logs/search-server.log       ← server stdout/stderr
#   * Prints "export LOCAL_SEARCH_URL=..." on stdout for eval-friendly use.

set -euo pipefail

MIN_HOURS_REMAINING="${MIN_HOURS_REMAINING:-48}"
SLIME_HOST_DIR="${SLIME_HOST_DIR:-/home/hhzhang01/slime}"
ENROOT_ROOTFS="${ENROOT_ROOTFS:-slime-test}"
GENAI_ROOT="${GENAI_ROOT:-/genai/fsx-project/hhzhang01}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-genai_interns}"
QOS="${QOS:-a100_dev}"
SERVER_PORT="${SERVER_PORT:-8000}"
SEARCH_JOB_NAME="${SEARCH_JOB_NAME:-supo-search-server}"

LOG_DIR="${GENAI_ROOT}/logs"
HOST_FILE="${LOG_DIR}/search-server.hostname"
SERVER_LOG="${LOG_DIR}/search-server.log"

mkdir -p "${LOG_DIR}"

log() { echo "[launch_search_server] $*" >&2; }

check_health() {
    # $1 = "host:port"; returns 0 if /health responds with 200.
    curl -sf --max-time 5 "http://$1/health" > /dev/null
}

wait_for_health() {
    # $1 = "host:port"; polls up to ~10 minutes for corpus load.
    local target="$1"
    log "waiting for ${target}/health (up to 10 minutes)..."
    for i in $(seq 1 60); do
        if check_health "${target}"; then
            log "health OK after ~$((i*10))s"
            return 0
        fi
        sleep 10
    done
    log "ERROR: ${target} never became healthy"
    return 1
}

submit_new_server() {
    log "submitting new 7-day search-server job on qos=${QOS}"
    # Build the enroot command as a single string to hand to sbatch --wrap.
    # ENROOT_TEMP_PATH=/dev/shm avoids overlay whiteout failures on this
    # cluster (memory: slime-enroot-import). We start from an already-imported
    # rootfs (not from a .sqsh) because login/compute nodes lack squashfuse.
    # ENROOT_DATA_PATH is pinned to the shared FSx-home enroot dir where the
    # rootfs was originally imported — Slurm's prolog otherwise sets it to a
    # node-local path (/opt/sunk/tmp/enroot-data/...) that has no rootfs.
    local wrap_cmd
    wrap_cmd="ENROOT_TEMP_PATH=/dev/shm \
        ENROOT_DATA_PATH=/storage/home/hhzhang01/.local/share/enroot \
        ENROOT_MOUNT_HOME=false \
        enroot start \
            --env PYTHONUNBUFFERED=1 \
            --mount ${SLIME_HOST_DIR}:/slime \
            --mount ${GENAI_ROOT}:/genai_hh \
            ${ENROOT_ROOTFS} \
            bash -c 'cd /slime/examples/supo_browsecomp && python -u search_server.py \
                --model /genai_hh/models/Qwen3-Embedding-8B \
                --corpus /genai_hh/datasets/browsecomp-plus-corpus \
                --corpus-embedding-dataset /genai_hh/datasets/browsecomp-plus-embeds \
                --host 0.0.0.0 --port ${SERVER_PORT}'"

    local jobid
    jobid=$(sbatch \
        --nodes=1 --gpus=1 --time=7-00:00:00 --mem=128G --cpus-per-task=8 \
        --account="${SLURM_ACCOUNT}" --qos="${QOS}" \
        --job-name="${SEARCH_JOB_NAME}" \
        --output="${SERVER_LOG}" \
        --wrap="${wrap_cmd}" \
        --parsable)
    log "sbatch returned JobId=${jobid}"
    printf '%s' "${jobid}"
}

wait_for_running() {
    # $1 = jobid. Waits until the job is RUNNING (or fails). Returns hostname.
    local jobid="$1"
    log "waiting for job ${jobid} to reach RUNNING state..."
    for i in $(seq 1 120); do  # up to 20 min in queue
        local state
        state=$(squeue -h -j "${jobid}" -o '%T' 2>/dev/null || true)
        if [[ "${state}" == "RUNNING" ]]; then
            local nodelist
            nodelist=$(squeue -h -j "${jobid}" -o '%N')
            # NodeList may be a range expression (e.g. a100-[001,003]); ask
            # scontrol to expand and take the first host.
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
    log "ERROR: job ${jobid} did not reach RUNNING within 20 minutes"
    return 1
}

timeleft_to_hours() {
    # Convert slurm TimeLeft ("D-HH:MM:SS" or "HH:MM:SS" or "MM:SS") into
    # floating-point hours. Prints result to stdout.
    local tl="$1"
    local days=0 hms
    if [[ "${tl}" == *-* ]]; then
        days="${tl%%-*}"
        hms="${tl#*-}"
    else
        hms="${tl}"
    fi
    # split hms into up to 3 fields
    local h=0 m=0 s=0
    IFS=: read -r a b c <<< "${hms}"
    if [[ -n "${c:-}" ]]; then
        h="${a}"; m="${b}"; s="${c}"
    elif [[ -n "${b:-}" ]]; then
        h="${a}"; m="${b}"
    else
        m="${a}"
    fi
    # Strip leading zeros so python doesn't complain "leading zeros in decimal
    # integer literals are not permitted" (07 -> octal ambiguity).
    days=$((10#${days:-0}))
    h=$((10#${h:-0}))
    m=$((10#${m:-0}))
    s=$((10#${s:-0}))
    python3 -c "print(${days}*24 + ${h} + ${m}/60 + ${s}/3600)"
}

resolve_existing() {
    # Print "jobid|state|timeleft|nodelist" for a matching job, or nothing.
    squeue -h -u "${USER}" -n "${SEARCH_JOB_NAME}" \
        -o '%i|%T|%L|%N' 2>/dev/null | head -n1
}

main() {
    local existing
    existing=$(resolve_existing || true)

    local target=""
    if [[ -n "${existing}" ]]; then
        local jobid state tl nodelist
        IFS='|' read -r jobid state tl nodelist <<< "${existing}"
        log "found existing job ${jobid} state=${state} timeleft=${tl}"

        if [[ "${state}" == "RUNNING" ]]; then
            local hours_left
            hours_left=$(timeleft_to_hours "${tl}")
            log "hours remaining: ${hours_left} (threshold ${MIN_HOURS_REMAINING})"
            if awk "BEGIN{exit !(${hours_left} >= ${MIN_HOURS_REMAINING})}"; then
                # enough runway — reuse
                local host
                host=$(scontrol show hostnames "${nodelist}" | head -n1)
                target="${host}:${SERVER_PORT}"
                if check_health "${target}"; then
                    log "reusing running server at ${target}"
                else
                    log "existing job ${jobid} is up but /health not responding; will wait"
                    wait_for_health "${target}"
                fi
            else
                log "WARNING: existing server ${jobid} has only ${hours_left}h left; scancel + resubmit"
                log "         if another experiment relies on it, Ctrl-C now (10s grace)..."
                sleep 10
                scancel "${jobid}"
                sleep 5
                target=""
            fi
        elif [[ "${state}" == "PENDING" ]]; then
            log "existing job ${jobid} still PENDING; waiting on it instead of resubmitting"
            local host
            host=$(wait_for_running "${jobid}")
            target="${host}:${SERVER_PORT}"
            wait_for_health "${target}"
        else
            log "existing job ${jobid} in unexpected state ${state}; scancel + resubmit"
            scancel "${jobid}" || true
            sleep 5
            target=""
        fi
    fi

    if [[ -z "${target}" ]]; then
        local jobid host
        jobid=$(submit_new_server)
        host=$(wait_for_running "${jobid}")
        target="${host}:${SERVER_PORT}"
        wait_for_health "${target}"
    fi

    echo "${target}" > "${HOST_FILE}"
    log "wrote ${HOST_FILE}: ${target}"
    log "server log: ${SERVER_LOG}"
    # stdout: eval-friendly export line (all log messages went to stderr)
    echo "export LOCAL_SEARCH_URL=http://${target}"
}

main "$@"
