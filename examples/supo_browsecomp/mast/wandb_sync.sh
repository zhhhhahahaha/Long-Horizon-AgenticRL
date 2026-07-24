#!/bin/bash
# Upload one MAST job's offline W&B runs from completed OILFS snapshots.
# Keep this process outside MAST: it owns the Meta W&B key and fwdproxy access.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  wandb_sync.sh once  <full-mast-job-name>
  wandb_sync.sh watch <full-mast-job-name>

"once" performs one incremental sync. "watch" syncs every 5 minutes while
MAST reports PENDING/RUNNING/SHUTTING_DOWN, then performs a final sync at
COMPLETE/FAILED/DEAD. Snapshots are extracted to devserver-local disk before
W&B reads them.
Run watch in tmux so it survives disconnects.
EOF
}

log() {
  printf '[wandb-sync] %s\n' "$*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "wandb_sync.sh: required command not found: $1" >&2
    exit 1
  fi
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

MODE="$1"
RUN_NAME="$2"
case "${MODE}" in
  once|watch) ;;
  *) usage; exit 2 ;;
esac
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "wandb_sync.sh: invalid MAST job name: ${RUN_NAME}" >&2
  exit 2
fi

WANDB_ROOT="${MAST_WANDB_ROOT:-/data/users/hhzhang01/wsfuse_mnt/hhzhang01/supo-slime/wandb}"
SNAPSHOT_ROOT="${MAST_WANDB_SNAPSHOT_ROOT:-/data/users/hhzhang01/wsfuse_mnt/hhzhang01/supo-slime/wandb-snapshots}"
KEY_FILE="${WANDB_KEY_FILE:-${HOME}/.wandb-key}"
WANDB_BIN="${WANDB_BIN:-/data/users/hhzhang01/slime-sanity/hfvenv/bin/wandb}"
WITH_PROXY_BIN="${WITH_PROXY_BIN:-with-proxy}"
MAST_BIN="${MAST_BIN:-mast}"
JQ_BIN="${JQ_BIN:-jq}"
SYNC_INTERVAL_SEC="${WANDB_SYNC_INTERVAL_SEC:-300}"
FINAL_SETTLE_SEC="${WANDB_FINAL_SYNC_SETTLE_SEC:-10}"
FINAL_RETRIES="${WANDB_FINAL_SYNC_RETRIES:-3}"
FINAL_RETRY_INTERVAL_SEC="${WANDB_FINAL_SYNC_RETRY_INTERVAL_SEC:-10}"
LOCK_DIR="${WANDB_SYNC_LOCK_DIR:-${XDG_RUNTIME_DIR:-/tmp}/mast-wandb-sync-${USER}}"
SYNC_CACHE_DIR="${WANDB_SYNC_CACHE_DIR:-${XDG_CACHE_HOME:-/tmp}/mast-wandb-sync-cache-${USER}}"
RUN_ROOT="${WANDB_ROOT}/${RUN_NAME}/wandb"
SNAPSHOT_RUN_ROOT="${SNAPSHOT_ROOT}/${RUN_NAME}"

require_command "${WITH_PROXY_BIN}"
require_command flock
require_command find
require_command tar
if [[ ! -d "${WANDB_ROOT}" && ! -d "${SNAPSHOT_ROOT}" ]]; then
  echo "wandb_sync.sh: neither OILFS W&B root is mounted: ${WANDB_ROOT}, ${SNAPSHOT_ROOT}" >&2
  exit 1
fi
if [[ ! -x "${WANDB_BIN}" ]]; then
  echo "wandb_sync.sh: W&B CLI is not executable: ${WANDB_BIN}" >&2
  exit 1
fi
if [[ ! -r "${KEY_FILE}" ]]; then
  echo "wandb_sync.sh: cannot read Meta W&B key: ${KEY_FILE}" >&2
  exit 1
fi

WANDB_API_KEY="$(tr -d ' \t\r\n' < "${KEY_FILE}")"
if [[ -z "${WANDB_API_KEY}" ]]; then
  echo "wandb_sync.sh: Meta W&B key is empty: ${KEY_FILE}" >&2
  exit 1
fi
export WANDB_API_KEY
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://meta.wandb.io}"

mkdir -p "${LOCK_DIR}" "${SYNC_CACHE_DIR}/${RUN_NAME}"
exec 9>"${LOCK_DIR}/${RUN_NAME}.lock"
if ! flock -n 9; then
  echo "wandb_sync.sh: another sync process is already active for ${RUN_NAME}" >&2
  exit 1
fi

declare -A LAST_SYNCED_SNAPSHOT=()

sync_once() {
  local candidates=()
  local pending_keys=()
  local pending_snapshots=()
  local path
  local publisher_dir publisher_key latest_snapshot
  local cache_parent cache_current cache_previous cache_tmp
  local found_run snapshot_error=0
  local index

  shopt -s nullglob
  for path in "${RUN_ROOT}"/offline-run-*; do
    [[ -d "${path}" ]] && candidates+=("${path}")
  done
  shopt -u nullglob

  if [[ -d "${SNAPSHOT_RUN_ROOT}" ]]; then
    while IFS= read -r -d '' publisher_dir; do
      latest_snapshot="$(
        find "${publisher_dir}" -maxdepth 1 -type f -name 'snapshot-*.tar' \
          -printf '%f\t%p\n' | sort | tail -n 1 | cut -f 2-
      )"
      [[ -n "${latest_snapshot}" ]] || continue

      publisher_key="$(basename "${publisher_dir}")"
      if [[ "${LAST_SYNCED_SNAPSHOT[${publisher_key}]:-}" == "${latest_snapshot}" ]]; then
        continue
      fi

      cache_parent="${SYNC_CACHE_DIR}/${RUN_NAME}/${publisher_key}"
      cache_current="${cache_parent}/current"
      cache_previous="${cache_parent}/previous"
      cache_tmp="${cache_parent}/.extract-$$"
      mkdir -p "${cache_parent}"
      rm -rf "${cache_tmp}"
      mkdir -p "${cache_tmp}"
      if ! tar -xf "${latest_snapshot}" -C "${cache_tmp}"; then
        log "failed to extract completed snapshot ${latest_snapshot}"
        rm -rf "${cache_tmp}"
        snapshot_error=1
        continue
      fi

      rm -rf "${cache_previous}"
      [[ ! -e "${cache_current}" ]] || mv "${cache_current}" "${cache_previous}"
      mv "${cache_tmp}" "${cache_current}"
      rm -rf "${cache_previous}"

      found_run=0
      while IFS= read -r -d '' path; do
        candidates+=("${path}")
        found_run=1
      done < <(find "${cache_current}" -type d -name 'offline-run-*' -print0)
      if [[ "${found_run}" == "1" ]]; then
        pending_keys+=("${publisher_key}")
        pending_snapshots+=("${latest_snapshot}")
      else
        log "snapshot has no offline runs yet: ${latest_snapshot}"
      fi
    done < <(find "${SNAPSHOT_RUN_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0)
  fi

  if [[ ${#candidates[@]} -eq 0 ]]; then
    log "no new offline runs under ${RUN_ROOT} or ${SNAPSHOT_RUN_ROOT}"
    return "${snapshot_error}"
  fi

  log "syncing ${#candidates[@]} offline run(s) to ${WANDB_BASE_URL}"
  if ! "${WITH_PROXY_BIN}" "${WANDB_BIN}" sync --append --no-sync-tensorboard "${candidates[@]}"; then
    return 1
  fi
  for ((index = 0; index < ${#pending_keys[@]}; index++)); do
    LAST_SYNCED_SNAPSHOT["${pending_keys[index]}"]="${pending_snapshots[index]}"
  done
  return "${snapshot_error}"
}

mast_state() {
  local response
  if ! response="$("${WITH_PROXY_BIN}" "${MAST_BIN}" --output json get-status "${RUN_NAME}")"; then
    return 1
  fi
  printf '%s' "${response}" | "${JQ_BIN}" -er '.data.state | strings'
}

final_sync() {
  local attempt

  if [[ "${FINAL_SETTLE_SEC}" != "0" ]]; then
    log "waiting ${FINAL_SETTLE_SEC}s for final OILFS writes"
    sleep "${FINAL_SETTLE_SEC}"
  fi
  for ((attempt = 1; attempt <= FINAL_RETRIES; attempt++)); do
    if sync_once; then
      return 0
    fi
    log "final sync attempt ${attempt}/${FINAL_RETRIES} failed"
    if ((attempt < FINAL_RETRIES)); then
      sleep "${FINAL_RETRY_INTERVAL_SEC}"
    fi
  done
  return 1
}

if [[ "${MODE}" == "once" ]]; then
  sync_once
  exit $?
fi

require_command "${MAST_BIN}"
require_command "${JQ_BIN}"
log "watching ${RUN_NAME}; interval=${SYNC_INTERVAL_SEC}s"
while true; do
  state=""
  if state="$(mast_state)"; then
    log "MAST state=${state}"
    case "${state}" in
      COMPLETE|FAILED|DEAD)
        if final_sync; then
          log "final sync complete; watcher exiting"
          exit 0
        fi
        log "final sync failed after ${FINAL_RETRIES} attempts"
        exit 1
        ;;
    esac
  else
    log "MAST status lookup failed; keeping watcher alive"
  fi

  if ! sync_once; then
    log "incremental sync failed; retrying next interval"
  fi
  sleep "${SYNC_INTERVAL_SEC}"
done
