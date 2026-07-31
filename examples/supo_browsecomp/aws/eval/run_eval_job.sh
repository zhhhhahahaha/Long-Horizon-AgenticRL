#!/bin/bash
# Slurm-side wrapper: stage frozen code/rootfs, then run one eval point in enroot.
set -euo pipefail

: "${EVAL_RUN_NAME:?set EVAL_RUN_NAME}"
: "${EVAL_POINT:?set EVAL_POINT}"
: "${EVAL_REQUESTED_STEP:?set EVAL_REQUESTED_STEP}"
: "${EVAL_OUTPUT_HOST:?set EVAL_OUTPUT_HOST}"
: "${EVAL_CODE_ARCHIVE:?set EVAL_CODE_ARCHIVE}"
: "${EVAL_CODE_ARCHIVE_SHA256:?set EVAL_CODE_ARCHIVE_SHA256}"
: "${SLURM_JOB_ID:?run through sbatch}"

if [[ -f "${EVAL_OUTPUT_HOST}/_SUCCESS" ]]; then
  echo "[eval-job] already complete: ${EVAL_OUTPUT_HOST}"
  exit 0
fi
case "${EVAL_OUTPUT_HOST}" in
  /genai/fsx-llm/interns/hhzhang01/*)
    EVAL_OUTPUT_DIR="/genai_llm${EVAL_OUTPUT_HOST#/genai/fsx-llm/interns/hhzhang01}"
    ;;
  *)
    echo "ERROR: EVAL_OUTPUT_HOST must be under /genai/fsx-llm/interns/hhzhang01" >&2
    exit 2
    ;;
esac

SEARCH_HOST_FILE="${SEARCH_HOST_FILE:-/genai/fsx-project/hhzhang01/logs/search-server.hostname}"
[[ -f "${SEARCH_HOST_FILE}" ]] || { echo "ERROR: missing ${SEARCH_HOST_FILE}" >&2; exit 1; }
LOCAL_SEARCH_URL="http://$(tr -d ' \t\r\n' < "${SEARCH_HOST_FILE}")"
if ! curl -g -sf --noproxy '*' --max-time 5 "${LOCAL_SEARCH_URL}/health" >/dev/null; then
  echo "ERROR: search server is unhealthy: ${LOCAL_SEARCH_URL}" >&2
  exit 1
fi

ACTUAL_ARCHIVE_SHA256="$(sha256sum "${EVAL_CODE_ARCHIVE}" | awk '{print $1}')"
if [[ "${ACTUAL_ARCHIVE_SHA256}" != "${EVAL_CODE_ARCHIVE_SHA256}" ]]; then
  echo "ERROR: code archive hash mismatch" >&2
  exit 1
fi

CODE_DIR="/dev/shm/slime-eval-code-${SLURM_JOB_ID}"
LOCAL_ENROOT_DATA="/dev/shm/enroot-${USER}-${SLURM_JOB_ID}"
ENROOT_ROOTFS="${ENROOT_ROOTFS:-slime-test}"
LOCAL_ROOTFS="${LOCAL_ENROOT_DATA}/${ENROOT_ROOTFS}"
ROOTFS_READY="${LOCAL_ROOTFS}/.slime-stage-complete"
cleanup() {
  rm -rf "${CODE_DIR}" "${LOCAL_ENROOT_DATA}"
}
trap cleanup EXIT

mkdir -p "${CODE_DIR}"
tar xzf "${EVAL_CODE_ARCHIVE}" -C "${CODE_DIR}"
[[ -x "${CODE_DIR}/examples/supo_browsecomp/aws/eval/run_eval.sh" ]] || {
  echo "ERROR: frozen archive lacks the AWS eval runner" >&2
  exit 1
}

mkdir -p "${LOCAL_ROOTFS}"
echo "[eval-job] staging rootfs to ${LOCAL_ROOTFS}"
rsync -a \
  --exclude='/tmp_host/' \
  --exclude='/var/spool/slurmd/pmix.*' \
  "/storage/home/hhzhang01/.local/share/enroot/${ENROOT_ROOTFS}/" \
  "${LOCAL_ROOTFS}/"
touch "${ROOTFS_READY}"

LLAMA_KEY_FILE="${LLAMA_KEY_FILE:-/home/hhzhang01/.llama_key}"
[[ -f "${LLAMA_KEY_FILE}" ]] || { echo "ERROR: missing judge key file ${LLAMA_KEY_FILE}" >&2; exit 1; }

ENROOT_ARGS=(
  start
  --mount "${CODE_DIR}:/slime"
  --mount /genai/fsx-project/hhzhang01:/genai_hh
  --mount /genai/fsx-llm/interns/hhzhang01:/genai_llm
  --mount "${LLAMA_KEY_FILE}:/run/secrets/llama_key"
  --env "EVAL_RUN_NAME=${EVAL_RUN_NAME}"
  --env "EVAL_POINT=${EVAL_POINT}"
  --env "EVAL_REQUESTED_STEP=${EVAL_REQUESTED_STEP}"
  --env "EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR}"
  --env "EVAL_CODE_ARCHIVE_SHA256=${EVAL_CODE_ARCHIVE_SHA256}"
  --env "LOCAL_SEARCH_URL=${LOCAL_SEARCH_URL}"
  --env LLAMA_KEY_FILE=/run/secrets/llama_key
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}"
  --env "EVAL_N=${EVAL_N:-4}"
  --env "EVAL_SEED=${EVAL_SEED:-42}"
  --env "EVAL_EXPECTED_QUESTIONS=${EVAL_EXPECTED_QUESTIONS:-150}"
  --env "BC_MODEL_SIZE=${BC_MODEL_SIZE:-4B}"
  --env "BCPLUS_FIXED_SEARCH_TOPK=${BCPLUS_FIXED_SEARCH_TOPK:-5}"
  --env "BCPLUS_DOC_WORDS_FULL=${BCPLUS_DOC_WORDS_FULL:-10000}"
  --env "BCPLUS_SEARCH_CONCURRENCY=${BCPLUS_SEARCH_CONCURRENCY:-64}"
  --env "BCPLUS_JUDGE_CONCURRENCY=${BCPLUS_JUDGE_CONCURRENCY:-16}"
  --env "BCPLUS_MAX_TURNS=${BCPLUS_MAX_TURNS:-64}"
  --env "BCPLUS_MAX_SUB_TRAJS=${BCPLUS_MAX_SUB_TRAJS:-5}"
  --env "BCPLUS_COMPRESS_THRESH=${BCPLUS_COMPRESS_THRESH:-0.85}"
  --env "BC_MAX_RESPONSE_LEN=${BC_MAX_RESPONSE_LEN:-32768}"
  --env "BC_MAX_CONTEXT_LEN=${BC_MAX_CONTEXT_LEN:-65536}"
  "${ENROOT_ROOTFS}"
  bash /slime/examples/supo_browsecomp/aws/eval/run_eval.sh
)

echo "[eval-job] point=${EVAL_POINT} search=${LOCAL_SEARCH_URL} output=${EVAL_OUTPUT_HOST}"
ENROOT_TEMP_PATH=/dev/shm \
ENROOT_DATA_PATH="${LOCAL_ENROOT_DATA}" \
ENROOT_MOUNT_HOME=false \
enroot "${ENROOT_ARGS[@]}"
