#!/bin/bash
# Single-checkpoint EVAL launcher for the 4B BC+ model (1 node, fire-and-forget).
#
# Evaluates ONE checkpoint (or the base model) on the 150-question BrowseComp-Plus
# test set and dumps every eval rollout (token ids + decoded trajectory + judge
# score + _bcplus metadata) to EVAL_DUMP_DIR/rollout_data/eval_0.pt.
#
# How it works: reuses run_qwen3p5_4B_colocate.sh's inner (ray-head + weight-sync
# + rollout) VERBATIM via SLIME_INNER=1, with BC_EVAL_MODE=1 flipping that script
# into slime's eval-only path (--num-rollout 0 --eval-interval 1). No HF
# conversion: actor.update_weights() syncs the loaded megatron checkpoint into
# sglang before eval runs. 1 node = TP=4 x CP=2 x PP=1 (DP=1) -> one eval_0.pt.
#
# Unlike the dumpsmoke launcher this uses `sbatch` (fire-and-forget), so many
# checkpoints can be queued at once and survive independently of the launching
# shell / session. The report is built offline from the .pt dumps, so there is
# no per-job wandb-sync loop.
#
# Usage (from the login pod; normally invoked by eval_all_checkpoints.sh):
#   export LLAMA_API_KEY='LLM|...'
#   RUN_NAME=eval-4b-iter44 \
#   BC_EVAL_LOAD=/genai/fsx-project/hhzhang01/checkpoints/<run>/iter_0000044 \
#   EVAL_DUMP_DIR=/genai/fsx-project/hhzhang01/evals/<run>-sweep/iter44 \
#   BC_EVAL_N=4 \
#   bash examples/supo_browsecomp/eval/legacy/slurm/run_qwen3p5_4B_eval.sh
#
# For the base model set BC_EVAL_LOAD=base (or leave it empty).
# The search server must already be up (launch_search_server.sh); auto-discovered
# via logs/search-server.hostname.

set -euo pipefail

# ---- Required params -------------------------------------------------------
: "${RUN_NAME:?set RUN_NAME (e.g. eval-4b-iter44)}"
: "${EVAL_DUMP_DIR:?set EVAL_DUMP_DIR (host path under /genai/fsx-project/hhzhang01)}"
# BC_EVAL_LOAD = <run>/iter_0000NN (host path) to eval that iter, or "base".
# The colocate script's BC_EVAL_MODE branch turns an iter dir into
# `--load <run> --ckpt-step N` (megatron overrides the tracker iteration), so no
# per-iter tracker/symlink dirs are needed here — we just pass it through.
BC_EVAL_LOAD="${BC_EVAL_LOAD:-base}"
BC_EVAL_N="${BC_EVAL_N:-4}"

# ---- Eval workload knobs (match the training rollout config by default) ----
# These flow into run_qwen3p5_4B_colocate.sh's ROLLOUT_ARGS + RUNTIME_ENV_JSON.
BCPLUS_MAX_TURNS="${BCPLUS_MAX_TURNS:-64}"
BC_MAX_RESPONSE_LEN="${BC_MAX_RESPONSE_LEN:-32768}"
BC_MAX_CONTEXT_LEN="${BC_MAX_CONTEXT_LEN:-65536}"
# rollout/global batch sizes are unused at --num-rollout 0 but must stay valid.
BC_ROLLOUT_BATCH_SIZE="${BC_ROLLOUT_BATCH_SIZE:-32}"
BC_GLOBAL_BATCH_SIZE="${BC_GLOBAL_BATCH_SIZE:-256}"

# ---- LLAMA_API_KEY (judge) -------------------------------------------------
if [[ -z "${LLAMA_API_KEY:-}" && -n "${LLAMA_KEY_FILE:-}" && -f "${LLAMA_KEY_FILE}" ]]; then
  export LLAMA_API_KEY="$(tr -d ' \t\r\n' < "${LLAMA_KEY_FILE}")"
fi
: "${LLAMA_API_KEY:?export LLAMA_API_KEY (or set LLAMA_KEY_FILE) before launching}"

# ---- Auto-discover the search server ---------------------------------------
if [[ -z "${LOCAL_SEARCH_URL:-}" ]]; then
    HOST_FILE=/genai/fsx-project/hhzhang01/logs/search-server.hostname
    if [[ -f "${HOST_FILE}" ]]; then
        SEARCH_TARGET=$(cat "${HOST_FILE}")
        if curl -sf --max-time 5 "http://${SEARCH_TARGET}/health" > /dev/null; then
            export LOCAL_SEARCH_URL="http://${SEARCH_TARGET}"
            echo "auto-discovered LOCAL_SEARCH_URL=${LOCAL_SEARCH_URL}"
        else
            echo "ERROR: search server at ${SEARCH_TARGET} not responding." >&2
            exit 1
        fi
    else
        echo "ERROR: LOCAL_SEARCH_URL not set and ${HOST_FILE} missing." >&2
        exit 1
    fi
fi

# ---- Slurm / enroot config -------------------------------------------------
SLIME_HOST_DIR=/home/hhzhang01/slime
ENROOT_ROOTFS="${ENROOT_ROOTFS:-slime-test}"
ENROOT_DIR=/storage/home/hhzhang01/.local/share/enroot   # shared FSx-home enroot dir
SLURM_ACCOUNT="${SLURM_ACCOUNT:-genai_interns}"
QOS="${QOS:-a100_genai_interns_high}"
EVAL_WALLTIME="${EVAL_WALLTIME:-6:00:00}"
EVAL_LOG_PATH="${EVAL_LOG_PATH:-/genai/fsx-project/hhzhang01/logs/${RUN_NAME}.log}"
mkdir -p "$(dirname "${EVAL_LOG_PATH}")"
mkdir -p "${EVAL_DUMP_DIR}"

# Coord dir: the reused inner (head branch) writes head.ip / done here.
COORD_DIR_HOST=/genai/fsx-project/hhzhang01/logs/ray-coord/${RUN_NAME}
COORD_DIR=/genai_hh/logs/ray-coord/${RUN_NAME}
mkdir -p "${COORD_DIR_HOST}"
rm -f "${COORD_DIR_HOST}/done" "${COORD_DIR_HOST}/head.ip"

echo "[eval-launch] RUN_NAME=${RUN_NAME}"
echo "[eval-launch] BC_EVAL_LOAD=${BC_EVAL_LOAD}"
echo "[eval-launch] EVAL_DUMP_DIR=${EVAL_DUMP_DIR}  (n=${BC_EVAL_N})"
echo "[eval-launch] QOS=${QOS}, 1 node, log -> ${EVAL_LOG_PATH}"

# One 1-node sbatch, fire-and-forget. enroot runs straight from the FSx-home
# enroot dir (read-only rootfs + /dev/shm overlay), NO cp staging (single node
# has no flock race). SLIME_INNER=1 + BC_EVAL_MODE=1 make the unmodified
# colocate script run its inner head branch in eval-only mode.
#
# Heredoc with UNQUOTED delimiter: ${...} expand now (values baked into the
# batch script); \${SLURM_*} are escaped so they expand at batch runtime.
sbatch \
    --nodes=1 --gpus-per-node=8 --ntasks-per-node=1 --exclusive \
    --cpus-per-task=64 --mem=0 \
    --account="${SLURM_ACCOUNT}" --qos="${QOS}" \
    --time="${EVAL_WALLTIME}" \
    --job-name="${RUN_NAME}" \
    --output="${EVAL_LOG_PATH}" <<SBATCH
#!/bin/bash
set -uo pipefail

# enroot starts from ONE shared FSx rootfs; concurrent nodes can lose the rootfs
# flock ("Could not acquire rootfs lock"), and Ray/GCS startup can transiently
# time out. Retry such transient failures until the eval dump appears, but fail
# FAST on deterministic errors (bad args / config / load asserts) so we don't
# burn 6 startups on something that can't succeed. Dump-presence = success (also
# treats the benign wandb atexit non-zero exit, which happens AFTER the dump is
# written, as success).
DUMP_CHECK='${EVAL_DUMP_DIR}/rollout_data/eval_0.pt'
for attempt in \$(seq 1 6); do
    # Jitter to de-synchronize concurrent starts across nodes.
    sleep \$(( (RANDOM % 25) + attempt * 8 ))
    echo "[eval-node] enroot start attempt \${attempt} (\$(date +%T))"
    ATTEMPT_LOG=/dev/shm/enroot_\${SLURM_JOB_ID:-0}_\${attempt}.log
    ENROOT_TEMP_PATH=/dev/shm \
    ENROOT_DATA_PATH=${ENROOT_DIR} \
    ENROOT_MOUNT_HOME=false \
    enroot start \
        --mount ${SLIME_HOST_DIR}:/slime \
        --mount ${SLIME_HOST_DIR}/aws-cluster:/aws-cluster \
        --mount /genai/fsx-project/hhzhang01:/genai_hh \
        --mount /genai/fsx-project/hhzhang01/wandb:/data/wandb \
        --env RUN_NAME='${RUN_NAME}' \
        --env SLIME_INNER=1 \
        --env NUM_NODES=1 \
        --env COORD_DIR='${COORD_DIR}' \
        --env LOCAL_SEARCH_URL='${LOCAL_SEARCH_URL}' \
        --env LLAMA_API_KEY='${LLAMA_API_KEY}' \
        --env SLURM_NODEID=\${SLURM_NODEID:-0} \
        --env SLURM_JOB_NODELIST=\${SLURM_JOB_NODELIST:-} \
        --env SLURM_JOB_ID=\${SLURM_JOB_ID} \
        --env BC_EVAL_MODE=1 \
        --env BC_EVAL_LOAD='${BC_EVAL_LOAD}' \
        --env BC_EVAL_N='${BC_EVAL_N}' \
        --env EVAL_DUMP_DIR='${EVAL_DUMP_DIR}' \
        --env BC_TEST_DATA='${BC_TEST_DATA:-}' \
        --env BCPLUS_MAX_TURNS='${BCPLUS_MAX_TURNS}' \
        --env BC_MAX_RESPONSE_LEN='${BC_MAX_RESPONSE_LEN}' \
        --env BC_MAX_CONTEXT_LEN='${BC_MAX_CONTEXT_LEN}' \
        --env BC_ROLLOUT_BATCH_SIZE='${BC_ROLLOUT_BATCH_SIZE}' \
        --env BC_GLOBAL_BATCH_SIZE='${BC_GLOBAL_BATCH_SIZE}' \
        ${ENROOT_ROOTFS} \
        bash /slime/examples/supo_browsecomp/aws/run_qwen3p5_4B_colocate.sh 2>&1 | tee "\${ATTEMPT_LOG}"
    rc=\${PIPESTATUS[0]}
    if [ -f "\${DUMP_CHECK}" ]; then
        echo "[eval-node] dump present -> success (enroot rc=\${rc})"; rm -f "\${ATTEMPT_LOG}"; break
    fi
    # No dump -> failure. Fail fast ONLY on deterministic errors (bad args /
    # config / load asserts) — retrying those just burns startups. Everything
    # else is treated as transient (rootfs-lock race, Ray/GCS startup timeout,
    # NCCL init, etc.) and retried with backoff.
    if grep -qiE "AssertionError|do not match|unrecognized arguments|the following arguments are required|invalid choice|error: argument" "\${ATTEMPT_LOG}"; then
        echo "[eval-node] attempt \${attempt}: DETERMINISTIC error (bad arg/config/assert) — not retrying. See log above." >&2
        rm -f "\${ATTEMPT_LOG}"; break
    fi
    echo "[eval-node] attempt \${attempt} failed with no dump (rc=\${rc}), likely transient (lock/ray/nccl) — backing off + retry"
    rm -f "\${ATTEMPT_LOG}"
    sleep \$(( attempt * 20 ))
done
SBATCH

echo "[eval-launch] submitted ${RUN_NAME} (check: squeue -n ${RUN_NAME})"
