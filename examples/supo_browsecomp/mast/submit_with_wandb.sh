#!/bin/bash
# Submit a MAST job from the devserver and attach its W&B sync watcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNC_SCRIPT="${MAST_WANDB_SYNC_SCRIPT:-${SCRIPT_DIR}/wandb_sync.sh}"
JQ_BIN="${JQ_BIN:-jq}"
TMUX_BIN="${TMUX_BIN:-tmux}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-${HOME}/.wandb-key}"
WANDB_BIN="${WANDB_BIN:-/data/users/hhzhang01/slime-sanity/hfvenv/bin/wandb}"
WITH_PROXY_BIN="${WITH_PROXY_BIN:-with-proxy}"
MAST_BIN="${MAST_BIN:-mast}"
MAST_WANDB_ROOT="${MAST_WANDB_ROOT:-/data/users/hhzhang01/wsfuse_mnt/hhzhang01/supo-slime/wandb}"
MAST_WANDB_SNAPSHOT_ROOT="${MAST_WANDB_SNAPSHOT_ROOT:-/data/users/hhzhang01/wsfuse_mnt/hhzhang01/supo-slime/wandb-snapshots}"
WATCHER_ROOT="${MAST_WANDB_WATCHER_ROOT:-${XDG_STATE_HOME:-${HOME}/.local/state}/mast-wandb}"
STARTUP_WAIT_SEC="${MAST_WANDB_WATCHER_STARTUP_WAIT_SEC:-2}"

usage() {
  cat >&2 <<'EOF'
Usage:
  submit_with_wandb.sh -- <rl/cli.sh mast --json ...>
  submit_with_wandb.sh watch-only <full-mast-job-name>

The submit form requires structured `rl/cli.sh mast --json` output. It submits
the command unchanged, extracts `.job.job_name`, and starts the devserver-side
W&B sync watcher in tmux. `watch-only` restores a missing watcher without
submitting another job.
EOF
}

log() {
  printf '[mast-wandb-submit] %s\n' "$*"
}

fail() {
  echo "submit_with_wandb.sh: $*" >&2
  exit 1
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "required command not found: $1"
  fi
}

validate_job_name() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] || fail "invalid MAST job name: $1"
}

preflight_watcher() {
  require_command "${JQ_BIN}"
  require_command "${TMUX_BIN}"
  require_command "${WITH_PROXY_BIN}"
  require_command "${MAST_BIN}"
  [[ -r "${SYNC_SCRIPT}" ]] || fail "W&B sync script is not readable: ${SYNC_SCRIPT}"
  [[ -x "${WANDB_BIN}" ]] || fail "W&B CLI is not executable: ${WANDB_BIN}"
  [[ -r "${WANDB_KEY_FILE}" ]] || fail "Meta W&B key is not readable: ${WANDB_KEY_FILE}"
  [[ -n "$(tr -d ' \t\r\n' < "${WANDB_KEY_FILE}")" ]] || fail "Meta W&B key is empty: ${WANDB_KEY_FILE}"
  if [[ ! -d "${MAST_WANDB_ROOT}" && ! -d "${MAST_WANDB_SNAPSHOT_ROOT}" ]]; then
    fail "neither OILFS W&B root is mounted: ${MAST_WANDB_ROOT}, ${MAST_WANDB_SNAPSHOT_ROOT}"
  fi
  [[ "${STARTUP_WAIT_SEC}" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
    fail "MAST_WANDB_WATCHER_STARTUP_WAIT_SEC must be a non-negative number"
  mkdir -p "${WATCHER_ROOT}"
}

watcher_session_name() {
  local name="mast-wandb-$1"
  name="${name//./_}"
  printf '%s' "${name:0:180}"
}

append_export() {
  local variable_name="$1"
  local value="${!variable_name-}"
  local quoted
  printf -v quoted '%q' "${value}"
  WATCHER_COMMAND+="export ${variable_name}=${quoted}; "
}

start_watcher() {
  local job_name="$1"
  local session log_dir log_file quoted_sync quoted_job quoted_log
  validate_job_name "${job_name}"
  session="$(watcher_session_name "${job_name}")"
  log_dir="${WATCHER_ROOT}/${job_name}"
  log_file="${log_dir}/watcher.log"
  mkdir -p "${log_dir}"

  if "${TMUX_BIN}" has-session -t "${session}" 2>/dev/null; then
    log "watcher already running: job=${job_name} tmux=${session}"
    log "watcher log: ${log_file}"
    return 0
  fi

  printf '[%s] starting watcher for %s\n' "$(date -Is)" "${job_name}" >> "${log_file}"
  WATCHER_COMMAND=""
  append_export WANDB_KEY_FILE
  append_export WANDB_BIN
  append_export WITH_PROXY_BIN
  append_export MAST_BIN
  append_export JQ_BIN
  append_export MAST_WANDB_ROOT
  append_export MAST_WANDB_SNAPSHOT_ROOT
  for optional_variable in \
    WANDB_BASE_URL WANDB_SYNC_INTERVAL_SEC WANDB_FINAL_SYNC_SETTLE_SEC \
    WANDB_FINAL_SYNC_RETRIES WANDB_FINAL_SYNC_RETRY_INTERVAL_SEC \
    WANDB_SYNC_LOCK_DIR WANDB_SYNC_CACHE_DIR; do
    if [[ -v "${optional_variable}" ]]; then
      append_export "${optional_variable}"
    fi
  done
  printf -v quoted_sync '%q' "${SYNC_SCRIPT}"
  printf -v quoted_job '%q' "${job_name}"
  printf -v quoted_log '%q' "${log_file}"
  WATCHER_COMMAND+="exec bash ${quoted_sync} watch ${quoted_job} >> ${quoted_log} 2>&1"

  if ! "${TMUX_BIN}" new-session -d -s "${session}" "${WATCHER_COMMAND}"; then
    echo "submit_with_wandb.sh: failed to create watcher tmux session: ${session}" >&2
    return 1
  fi
  sleep "${STARTUP_WAIT_SEC}"
  if ! "${TMUX_BIN}" has-session -t "${session}" 2>/dev/null; then
    echo "submit_with_wandb.sh: watcher exited during startup; inspect ${log_file}" >&2
    return 1
  fi

  log "watcher started: job=${job_name} tmux=${session}"
  log "watcher log: ${log_file}"
}

has_json_flag() {
  local argument
  for argument in "$@"; do
    case "${argument}" in
      --json|--json=true|--json=True) return 0 ;;
    esac
  done
  return 1
}

submit_and_watch() {
  local command_rc status dryrun job_name mast_url timestamp response_file

  [[ $# -gt 0 ]] || { usage; exit 2; }
  has_json_flag "$@" || fail "submission command must include --json"
  require_command "$1"

  timestamp="$(date +%Y%m%d-%H%M%S)"
  mkdir -p "${WATCHER_ROOT}/submissions"
  response_file="${WATCHER_ROOT}/submissions/submit-${timestamp}-$$.json"
  log "submitting MAST command; structured response: ${response_file}"

  set +e
  "$@" > "${response_file}"
  command_rc=$?
  set -e

  if ! "${JQ_BIN}" -e 'type == "object" and (.status | type == "string")' \
      "${response_file}" >/dev/null 2>&1; then
    echo "submit_with_wandb.sh: submission output was not valid structured JSON" >&2
    echo "submit_with_wandb.sh: verify MAST before retrying; no watcher was started" >&2
    echo "submit_with_wandb.sh: raw response: ${response_file}" >&2
    exit 2
  fi

  status="$("${JQ_BIN}" -r '.status' "${response_file}")"
  if ((command_rc != 0)) || [[ "${status}" != "ok" ]]; then
    echo "submit_with_wandb.sh: MAST submission failed: $(
      "${JQ_BIN}" -r '.error.message // "unknown error"' "${response_file}"
    )" >&2
    echo "submit_with_wandb.sh: response: ${response_file}" >&2
    ((command_rc != 0)) && exit "${command_rc}"
    exit 1
  fi

  dryrun="$("${JQ_BIN}" -r '.dryrun' "${response_file}")"
  if [[ "${dryrun}" == "true" ]]; then
    log "dry-run completed; no MAST job or W&B watcher was created"
    log "dry-run response: ${response_file}"
    return 0
  fi

  if ! job_name="$("${JQ_BIN}" -er '.job.job_name | strings | select(length > 0)' "${response_file}")"; then
    echo "submit_with_wandb.sh: MAST reported success but no .job.job_name was returned" >&2
    echo "submit_with_wandb.sh: verify MAST before retrying; response: ${response_file}" >&2
    exit 2
  fi
  validate_job_name "${job_name}"
  mast_url="$("${JQ_BIN}" -r '.job.mast_url // empty' "${response_file}")"

  mkdir -p "${WATCHER_ROOT}/${job_name}"
  mv "${response_file}" "${WATCHER_ROOT}/${job_name}/submit-${timestamp}.json"
  response_file="${WATCHER_ROOT}/${job_name}/submit-${timestamp}.json"
  log "MAST job submitted: ${job_name}"
  [[ -z "${mast_url}" ]] || log "MAST URL: ${mast_url}"
  log "submission response: ${response_file}"

  if ! start_watcher "${job_name}"; then
    echo "submit_with_wandb.sh: MAST job ${job_name} is already submitted, but its watcher did not start" >&2
    echo "submit_with_wandb.sh: recover without resubmitting:" >&2
    echo "  $0 watch-only ${job_name}" >&2
    exit 3
  fi
}

case "${1:-}" in
  watch-only)
    [[ $# -eq 2 ]] || { usage; exit 2; }
    preflight_watcher
    start_watcher "$2"
    ;;
  --)
    shift
    preflight_watcher
    submit_and_watch "$@"
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
