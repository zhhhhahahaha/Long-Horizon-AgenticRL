#!/bin/bash
# Submit one config-defined SUPO training experiment through MAST.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_WITH_WANDB_DEFAULT="${SCRIPT_DIR}/submit_with_wandb.sh"

usage() {
  cat >&2 <<'EOF'
Usage:
  submit_experiment.sh [--dry-run] <experiment-config.sh>

Every real submission first runs and validates a MAST dry-run. The dry-run must
contain the configured trainer task count and ROLE_ASSIGNMENT_MAP rank count.
EOF
}

fail() {
  echo "submit_experiment.sh: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

require_variable() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "experiment config must set ${name}"
}

require_positive_integer() {
  local name="$1"
  local value="${!name:-}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "${name} must be a positive integer, got: ${value:-<unset>}"
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
[[ $# -eq 1 ]] || { usage; exit 2; }

CONFIG_PATH="$1"
[[ "${CONFIG_PATH}" = /* ]] || CONFIG_PATH="${PWD}/${CONFIG_PATH}"
[[ -r "${CONFIG_PATH}" ]] || fail "experiment config is not readable: ${CONFIG_PATH}"
# shellcheck source=/dev/null
source "${CONFIG_PATH}"

require_variable MAST_JOB_NAME
require_variable MAST_NUM_NODES
require_variable BC_MODEL_SIZE
require_variable BC_NUM_ROLLOUT
require_variable BC_ROLLOUT_BATCH_SIZE
require_variable BC_N_SAMPLES
require_variable BC_GLOBAL_BATCH_SIZE
require_positive_integer MAST_NUM_NODES
require_positive_integer BC_NUM_ROLLOUT
require_positive_integer BC_ROLLOUT_BATCH_SIZE
require_positive_integer BC_N_SAMPLES
require_positive_integer BC_GLOBAL_BATCH_SIZE
if [[ -v BC_LOG_PROBS_CHUNK_SIZE ]]; then
  require_positive_integer BC_LOG_PROBS_CHUNK_SIZE
fi
if [[ -v BC_SGLANG_MEM_FRACTION_STATIC && ! "${BC_SGLANG_MEM_FRACTION_STATIC}" =~ ^(0\.[0-9]*[1-9][0-9]*|1(\.0+)?)$ ]]; then
  fail "BC_SGLANG_MEM_FRACTION_STATIC must be in (0, 1], got: ${BC_SGLANG_MEM_FRACTION_STATIC}"
fi
if [[ -v BC_SLIM_INTERMEDIATE_CHECKPOINTS ]]; then
  case "$(printf '%s' "${BC_SLIM_INTERMEDIATE_CHECKPOINTS}" | tr '[:upper:]' '[:lower:]')" in
    1|true|0|false) ;;
    *) fail "BC_SLIM_INTERMEDIATE_CHECKPOINTS must be one of: 1, true, 0, false" ;;
  esac
fi
if [[ -v BC_OVERRIDE_OPT_PARAM_SCHEDULER ]]; then
  case "$(printf '%s' "${BC_OVERRIDE_OPT_PARAM_SCHEDULER}" | tr '[:upper:]' '[:lower:]')" in
    1|true|0|false) ;;
    *) fail "BC_OVERRIDE_OPT_PARAM_SCHEDULER must be one of: 1, true, 0, false" ;;
  esac
fi
if [[ -v BC_RUN_NAME ]]; then
  [[ -n "${BC_RUN_NAME}" ]] || fail "BC_RUN_NAME must not be empty"
  [[ "${BC_RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ && "${BC_RUN_NAME}" != "." && "${BC_RUN_NAME}" != ".." ]] || \
    fail "invalid BC_RUN_NAME: ${BC_RUN_NAME}"
fi
[[ "${MAST_JOB_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || fail "invalid MAST_JOB_NAME: ${MAST_JOB_NAME}"

MAST_RL_CLI="${MAST_RL_CLI:-/data/users/hhzhang01/fbsource/genai/msl/rl/cli.sh}"
MAST_SUBMIT_WITH_WANDB="${MAST_SUBMIT_WITH_WANDB:-${SUBMIT_WITH_WANDB_DEFAULT}}"
MAST_JQ_BIN="${MAST_JQ_BIN:-jq}"
MAST_DRYRUN_ROOT="${MAST_DRYRUN_ROOT:-${XDG_STATE_HOME:-${HOME}/.local/state}/mast-experiments/dryruns}"
MAST_TENANT="${MAST_TENANT:-rhea_assistant_interns}"
MAST_REGION="${MAST_REGION:-nha}"
MAST_JOB_PRIORITY="${MAST_JOB_PRIORITY:-CRITICAL}"
MAST_WORKSPACE="${MAST_WORKSPACE:-None}"
MAST_MAIN_PACKAGE="${MAST_MAIN_PACKAGE:-xlformers_pretrain1:latest}"
MAST_PROGRAM="${MAST_PROGRAM:-avocado.rev1.rl.debug_80m}"
MAST_ROLE="${MAST_ROLE:-trainer_0}"
MAST_ENABLE_TTLS="${MAST_ENABLE_TTLS:-True}"
MAST_RETRIES="${MAST_RETRIES:-3}"
MAST_USE_CONDA_DOCKER="${MAST_USE_CONDA_DOCKER:-True}"
MAST_CONDA_DOCKER_IMAGE="${MAST_CONDA_DOCKER_IMAGE:-588845226011.dkr.ecr.us-east-2.amazonaws.com/msl_infra/slime:hhz-20260629a}"
MAST_DOCKER_HOST_CMD="${MAST_DOCKER_HOST_CMD:-sh -c 'nohup python3 /mnt/wsfuse/hhzhang01/slime-sanity/connect_proxy.py 9080 >/tmp/relay.log 2>&1 &'}"
MAST_CODE_ARCHIVE="${MAST_CODE_ARCHIVE:-/mnt/wsfuse/hhzhang01/supo-slime/slime-code.tgz}"
MAST_TRAINER_SCRIPT="${MAST_TRAINER_SCRIPT:-/slime-src/examples/supo_browsecomp/mast/run_trainer.sh}"
MAST_HOST="${MAST_HOST:-zionex_80g}"
MAST_WSF_SRC="${MAST_WSF_SRC:-ws://ws.ai.eag0genai/genai_fair_llm}"
MAST_GPUS_PER_NODE="${MAST_GPUS_PER_NODE:-8}"
MAST_DATA_PARALLEL_SIZE="${MAST_DATA_PARALLEL_SIZE:-${MAST_NUM_NODES}}"
MAST_CONTEXT_PARALLEL_SIZE="${MAST_CONTEXT_PARALLEL_SIZE:-${MAST_GPUS_PER_NODE}}"

for name in MAST_GPUS_PER_NODE MAST_DATA_PARALLEL_SIZE MAST_CONTEXT_PARALLEL_SIZE MAST_RETRIES; do
  require_positive_integer "${name}"
done

EXPECTED_ASSIGNED_RANKS=$((MAST_NUM_NODES * MAST_GPUS_PER_NODE))
CONFIGURED_ASSIGNED_RANKS=$((MAST_DATA_PARALLEL_SIZE * MAST_CONTEXT_PARALLEL_SIZE))
if ((CONFIGURED_ASSIGNED_RANKS != EXPECTED_ASSIGNED_RANKS)); then
  fail "trainer override creates ${CONFIGURED_ASSIGNED_RANKS} ranks; expected ${EXPECTED_ASSIGNED_RANKS} for ${MAST_NUM_NODES} nodes"
fi

export BC_EXPECTED_NUM_NODES="${MAST_NUM_NODES}"
if [[ -n "${BC_RUN_NAME:-}" ]]; then
  # The trainer publishes W&B snapshots under the resumed logical run name,
  # while the watcher must query the newly submitted MAST job's status.
  export MAST_WANDB_RUN_NAME="${BC_RUN_NAME}"
fi
TRAIN_ENV_VARS=(
  BC_EXPECTED_NUM_NODES
  BC_RUN_NAME
  BC_MODEL_SIZE BC_NUM_ROLLOUT BC_ROLLOUT_BATCH_SIZE BC_N_SAMPLES
  BC_GLOBAL_BATCH_SIZE BC_MAX_RESPONSE_LEN BC_MAX_CONTEXT_LEN
  BC_TP BC_CP BC_SGLANG_TP BC_MAX_TOKENS_PER_GPU
  BC_LOG_PROBS_CHUNK_SIZE BC_SGLANG_MEM_FRACTION_STATIC
  BC_SAVE_INTERVAL BC_SLIM_INTERMEDIATE_CHECKPOINTS BC_OVERRIDE_OPT_PARAM_SCHEDULER
  BC_DUMP_ROLLOUT BC_WANDB_PROJECT
  BCPLUS_MAX_TURNS BCPLUS_COMPRESS_THRESH BCPLUS_MAX_SUB_TRAJS
  BCPLUS_COMPRESS_PENALTY BCPLUS_DUMP_TRAIN_OLD
  BCPLUS_JUDGE_MODEL BCPLUS_JUDGE_BASE_URL
  BCPLUS_JUDGE_CONCURRENCY BCPLUS_SEARCH_CONCURRENCY
  BCPLUS_DYNAMIC_SAMPLING
  MAST_WANDB_SNAPSHOT_INTERVAL_SEC MAST_RAY_LOG_COPY_TIMEOUT_SEC
  MAST_PERSIST_RAY_LOGS WANDB_X_FLUSH_INTERVAL_SECONDS
)

DOCKER_CUSTOM_CMD="mkdir -p /slime-src"
printf -v QUOTED_VALUE '%q' "${MAST_CODE_ARCHIVE}"
DOCKER_CUSTOM_CMD+=" && tar xzf ${QUOTED_VALUE} -C /slime-src && cd /slime-src && export"
for variable_name in "${TRAIN_ENV_VARS[@]}"; do
  if [[ -v "${variable_name}" ]]; then
    printf -v QUOTED_VALUE '%q' "${!variable_name}"
    DOCKER_CUSTOM_CMD+=" ${variable_name}=${QUOTED_VALUE}"
  fi
done
printf -v QUOTED_VALUE '%q' "${MAST_TRAINER_SCRIPT}"
DOCKER_CUSTOM_CMD+=" && bash ${QUOTED_VALUE}"

MAST_COMMAND=(
  "${MAST_RL_CLI}" mast
  --json
  "--tenant=${MAST_TENANT}"
  "--region=${MAST_REGION}"
  "--job_priority=${MAST_JOB_PRIORITY}"
  "--workspace=${MAST_WORKSPACE}"
  "--main_package=${MAST_MAIN_PACKAGE}"
  program "${MAST_PROGRAM}"
  "--roles=${MAST_ROLE}"
  "--job_name=${MAST_JOB_NAME}"
  "--enable_ttls=${MAST_ENABLE_TTLS}"
  "--retries=${MAST_RETRIES}"
  "--use_conda_docker=${MAST_USE_CONDA_DOCKER}"
  "--conda_docker_image=${MAST_CONDA_DOCKER_IMAGE}"
  "--docker_host_cmd=${MAST_DOCKER_HOST_CMD}"
  "--docker_custom_cmd=${DOCKER_CUSTOM_CMD}"
  "--host=${MAST_HOST}"
  "--wsf_src=${MAST_WSF_SRC}"
  "--overrides=cluster_config.trainer_parallelism.data_parallel_size=${MAST_DATA_PARALLEL_SIZE},cluster_config.trainer_parallelism.context_parallel_size=${MAST_CONTEXT_PARALLEL_SIZE}"
)

validate_dryrun() {
  local response_file="$1"
  local task_count rank_map

  "${MAST_JQ_BIN}" -e '.status == "ok" and .dryrun == true' "${response_file}" >/dev/null || \
    fail "MAST dry-run did not return status=ok and dryrun=true: ${response_file}"
  # shellcheck disable=SC2016  # $role is a jq variable.
  task_count="$("${MAST_JQ_BIN}" -er --arg role "${MAST_ROLE}" \
    '.spec.hpc_job_definition.hpcTaskGroups[] | select(.name == $role) | .taskCount' \
    "${response_file}")" || fail "MAST dry-run lacks taskCount for ${MAST_ROLE}: ${response_file}"
  [[ "${task_count}" == "${MAST_NUM_NODES}" ]] || \
    fail "MAST dry-run taskCount=${task_count}; expected ${MAST_NUM_NODES}: ${response_file}"

  # shellcheck disable=SC2016  # $role is a jq variable.
  rank_map="$("${MAST_JQ_BIN}" -er --arg role "${MAST_ROLE}" \
    '.spec.app_def.roles[] | select(.name == $role) | .env.ROLE_ASSIGNMENT_MAP' \
    "${response_file}")" || fail "MAST dry-run lacks ROLE_ASSIGNMENT_MAP for ${MAST_ROLE}: ${response_file}"
  [[ ",${rank_map}," == *",${MAST_ROLE}=${EXPECTED_ASSIGNED_RANKS},"* ]] || \
    fail "MAST dry-run ROLE_ASSIGNMENT_MAP=${rank_map}; expected ${MAST_ROLE}=${EXPECTED_ASSIGNED_RANKS}: ${response_file}"

  echo "[mast-experiment] dry-run verified: taskCount=${task_count} ROLE_ASSIGNMENT_MAP=${rank_map}"
}

run_dryrun() {
  local timestamp response_file
  timestamp="$(date +%Y%m%d-%H%M%S)"
  mkdir -p "${MAST_DRYRUN_ROOT}"
  response_file="${MAST_DRYRUN_ROOT}/${MAST_JOB_NAME}-${timestamp}-$$.json"
  echo "[mast-experiment] validating ${MAST_JOB_NAME}: nodes=${MAST_NUM_NODES} ranks=${EXPECTED_ASSIGNED_RANKS}"
  if ! "${MAST_COMMAND[@]}" --dryrun > "${response_file}"; then
    fail "MAST dry-run command failed; partial output: ${response_file}"
  fi
  validate_dryrun "${response_file}"
  echo "[mast-experiment] dry-run response: ${response_file}"
}

require_command "${MAST_RL_CLI}"
require_command "${MAST_JQ_BIN}"
run_dryrun
if ((DRY_RUN_ONLY)); then
  exit 0
fi

[[ -x "${MAST_SUBMIT_WITH_WANDB}" ]] || fail "W&B submit wrapper is not executable: ${MAST_SUBMIT_WITH_WANDB}"
echo "[mast-experiment] submitting ${MAST_JOB_NAME} after successful dry-run"
exec "${MAST_SUBMIT_WITH_WANDB}" -- "${MAST_COMMAND[@]}"
