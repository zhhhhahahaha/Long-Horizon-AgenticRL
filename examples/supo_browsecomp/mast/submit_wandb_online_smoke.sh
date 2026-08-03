#!/bin/bash
# Submit a one-node MAST smoke job that logs directly to Meta W&B.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

usage() {
  cat >&2 <<'EOF'
Usage: submit_wandb_online_smoke.sh [--dry-run]

The real submission first validates a MAST dry-run. The API key is copied to a
private OILFS file and read only inside the compute container; it is never put
in the MAST command or job environment.
EOF
}

fail() {
  echo "submit_wandb_online_smoke.sh: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

DRY_RUN_ONLY=0
case "${1:-}" in
  --dry-run)
    DRY_RUN_ONLY=1
    shift
    ;;
  -h|--help)
    usage
    exit 0
    ;;
esac
[[ $# -eq 0 ]] || { usage; exit 2; }

MAST_RL_CLI="${MAST_RL_CLI:-/data/users/hhzhang01/fbsource/genai/msl/rl/cli.sh}"
MAST_JQ_BIN="${MAST_JQ_BIN:-jq}"
MAST_JOB_NAME="${MAST_JOB_NAME:-supo-wandb-online-smoke}"
MAST_TENANT="${MAST_TENANT:-rhea_assistant_avocado_iterations}"
MAST_REGION="${MAST_REGION:-eag}"
MAST_HOST="${MAST_HOST:-grandteton_80g_roce}"
MAST_JOB_PRIORITY="${MAST_JOB_PRIORITY:-HIGH}"
MAST_MAIN_PACKAGE="${MAST_MAIN_PACKAGE:-xlformers_pretrain1:latest}"
MAST_PROGRAM="${MAST_PROGRAM:-avocado.rev1.rl.debug_80m}"
MAST_ROLE="${MAST_ROLE:-trainer_0}"
MAST_CONDA_DOCKER_IMAGE="${MAST_CONDA_DOCKER_IMAGE:-588845226011.dkr.ecr.us-east-2.amazonaws.com/msl_infra/slime:hhz-20260629a}"
MAST_WSF_SRC="${MAST_WSF_SRC:-ws://ws.ai.eag0genai/genai_fair_llm}"

WANDB_KEY_FILE="${WANDB_KEY_FILE:-${HOME}/.wandb-key}"
MAST_WANDB_KEY_HOST_PATH="${MAST_WANDB_KEY_HOST_PATH:-/data/users/hhzhang01/wsfuse_mnt/hhzhang01/supo-slime/.wandb-online-smoke-key}"
MAST_WANDB_KEY_CONTAINER_PATH="${MAST_WANDB_KEY_CONTAINER_PATH:-/mnt/wsfuse/hhzhang01/supo-slime/.wandb-online-smoke-key}"
MAST_CODE_ARCHIVE_HOST_PATH="${MAST_CODE_ARCHIVE_HOST_PATH:-/data/users/hhzhang01/wsfuse_mnt/hhzhang01/supo-slime/slime-code-wandb-online-smoke.tgz}"
MAST_CODE_ARCHIVE_CONTAINER_PATH="${MAST_CODE_ARCHIVE_CONTAINER_PATH:-/mnt/wsfuse/hhzhang01/supo-slime/slime-code-wandb-online-smoke.tgz}"
MAST_WANDB_RESULT_PATH="${MAST_WANDB_RESULT_PATH:-/mnt/wsfuse/hhzhang01/supo-slime/wandb-online-smoke-results/${MAST_JOB_NAME}.json}"
MAST_WANDB_STATE_ROOT="${MAST_WANDB_STATE_ROOT:-${XDG_STATE_HOME:-${HOME}/.local/state}/mast-wandb-online-smoke}"
MAST_BUILD_CODE_ARCHIVE="${MAST_BUILD_CODE_ARCHIVE:-1}"

WANDB_BASE_URL="${WANDB_BASE_URL:-https://meta.wandb.io}"
WANDB_ENTITY="${WANDB_ENTITY:-hhzhang01}"
WANDB_PROJECT="${WANDB_PROJECT:-supo-bcplus-mast}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${MAST_JOB_NAME}}"

[[ "${MAST_JOB_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || fail "invalid MAST job name: ${MAST_JOB_NAME}"
[[ "${MAST_BUILD_CODE_ARCHIVE}" == "0" || "${MAST_BUILD_CODE_ARCHIVE}" == "1" ]] || \
  fail "MAST_BUILD_CODE_ARCHIVE must be 0 or 1"
[[ -s "${WANDB_KEY_FILE}" ]] || fail "W&B API key is missing or empty: ${WANDB_KEY_FILE}"
require_command "${MAST_RL_CLI}"
require_command "${MAST_JQ_BIN}"

mkdir -p "$(dirname "${MAST_WANDB_KEY_HOST_PATH}")" "${MAST_WANDB_STATE_ROOT}"
umask 077
key_tmp="${MAST_WANDB_KEY_HOST_PATH}.tmp-$$"
trap 'rm -f "${key_tmp}"' EXIT
tr -d ' \t\r\n' < "${WANDB_KEY_FILE}" > "${key_tmp}"
[[ -s "${key_tmp}" ]] || fail "W&B API key is empty after whitespace removal"
chmod 600 "${key_tmp}"
mv -f "${key_tmp}" "${MAST_WANDB_KEY_HOST_PATH}"

if [[ "${MAST_BUILD_CODE_ARCHIVE}" == "1" ]]; then
  require_command git
  [[ -z "$(git -C "${REPO_ROOT}" status --short)" ]] || \
    fail "experiment worktree must be clean before archiving"
  mkdir -p "$(dirname "${MAST_CODE_ARCHIVE_HOST_PATH}")"
  archive_tmp="${MAST_CODE_ARCHIVE_HOST_PATH}.tmp-$$"
  trap 'rm -f "${key_tmp}" "${archive_tmp:-}"' EXIT
  git -C "${REPO_ROOT}" archive --format=tar.gz --output="${archive_tmp}" HEAD
  mv -f "${archive_tmp}" "${MAST_CODE_ARCHIVE_HOST_PATH}"
elif [[ ! -s "${MAST_CODE_ARCHIVE_HOST_PATH}" ]]; then
  fail "prebuilt code archive is missing or empty: ${MAST_CODE_ARCHIVE_HOST_PATH}"
fi

printf -v q_archive '%q' "${MAST_CODE_ARCHIVE_CONTAINER_PATH}"
printf -v q_key '%q' "${MAST_WANDB_KEY_CONTAINER_PATH}"
printf -v q_base_url '%q' "${WANDB_BASE_URL}"
printf -v q_entity '%q' "${WANDB_ENTITY}"
printf -v q_project '%q' "${WANDB_PROJECT}"
printf -v q_group '%q' "${WANDB_RUN_GROUP}"
printf -v q_result '%q' "${MAST_WANDB_RESULT_PATH}"
DOCKER_CUSTOM_CMD="mkdir -p /slime-src"
DOCKER_CUSTOM_CMD+=" && tar xzf ${q_archive} -C /slime-src && cd /slime-src"
DOCKER_CUSTOM_CMD+=" && export WANDB_API_KEY=\$(tr -d ' \\t\\r\\n' < ${q_key})"
DOCKER_CUSTOM_CMD+=" WANDB_BASE_URL=${q_base_url} WANDB_ENTITY=${q_entity}"
DOCKER_CUSTOM_CMD+=" WANDB_PROJECT=${q_project} WANDB_RUN_GROUP=${q_group}"
DOCKER_CUSTOM_CMD+=" MAST_WANDB_RESULT_PATH=${q_result}"
DOCKER_CUSTOM_CMD+=" && exec python3 examples/supo_browsecomp/mast/wandb_online_smoke.py"

MAST_COMMAND=(
  "${MAST_RL_CLI}" mast
  --json
  "--tenant=${MAST_TENANT}"
  "--region=${MAST_REGION}"
  "--job_priority=${MAST_JOB_PRIORITY}"
  --workspace=None
  "--main_package=${MAST_MAIN_PACKAGE}"
  program "${MAST_PROGRAM}"
  "--roles=${MAST_ROLE}"
  "--job_name=${MAST_JOB_NAME}"
  --enable_ttls=True
  --retries=0
  --use_conda_docker=True
  "--conda_docker_image=${MAST_CONDA_DOCKER_IMAGE}"
  "--docker_custom_cmd=${DOCKER_CUSTOM_CMD}"
  "--host=${MAST_HOST}"
  "--wsf_src=${MAST_WSF_SRC}"
  "--overrides=cluster_config.trainer_parallelism.data_parallel_size=1,cluster_config.trainer_parallelism.context_parallel_size=8"
)

timestamp="$(date +%Y%m%d-%H%M%S)"
dryrun_response="${MAST_WANDB_STATE_ROOT}/dryrun-${timestamp}-$$.json"
echo "[wandb-online-smoke-submit] validating ${MAST_JOB_NAME} on ${MAST_REGION}/${MAST_HOST}"
"${MAST_COMMAND[@]}" --dryrun > "${dryrun_response}"
# shellcheck disable=SC2016  # $role is a jq variable.
"${MAST_JQ_BIN}" -e --arg role "${MAST_ROLE}" '
  .status == "ok" and .dryrun == true and
  ([.spec.hpc_job_definition.hpcTaskGroups[] | select(.name == $role) | .taskCount][0] == 1) and
  ([.spec.app_def.roles[] | select(.name == $role) | .env.ROLE_ASSIGNMENT_MAP][0] | contains($role + "=8"))
' "${dryrun_response}" >/dev/null || fail "MAST dry-run validation failed: ${dryrun_response}"
echo "[wandb-online-smoke-submit] dry-run response: ${dryrun_response}"

if ((DRY_RUN_ONLY)); then
  exit 0
fi

submit_response="${MAST_WANDB_STATE_ROOT}/submit-${timestamp}-$$.json"
echo "[wandb-online-smoke-submit] submitting ${MAST_JOB_NAME}"
"${MAST_COMMAND[@]}" > "${submit_response}"
"${MAST_JQ_BIN}" -e '.status == "ok" and .dryrun == false' "${submit_response}" >/dev/null || \
  fail "MAST submission failed: ${submit_response}"
full_job_name="$("${MAST_JQ_BIN}" -er '.job.job_name' "${submit_response}")"
mast_url="$("${MAST_JQ_BIN}" -r '.job.mast_url // empty' "${submit_response}")"
echo "[wandb-online-smoke-submit] job=${full_job_name}"
[[ -z "${mast_url}" ]] || echo "[wandb-online-smoke-submit] MAST URL: ${mast_url}"
echo "[wandb-online-smoke-submit] result path: ${MAST_WANDB_RESULT_PATH}"
echo "[wandb-online-smoke-submit] response: ${submit_response}"
