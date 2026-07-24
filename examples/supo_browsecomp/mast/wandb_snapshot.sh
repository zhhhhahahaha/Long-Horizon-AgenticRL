#!/bin/bash
# Publish point-in-time copies of a node-local W&B directory to OILFS.
# W&B must never write its transaction log directly to OILFS: wandb-core uses
# file operations that OILFS does not reliably support.  A completed tar file
# is renamed into view only after its sequential OILFS copy finishes.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  wandb_snapshot.sh once  <local-wandb-root> <oilfs-publisher-dir>
  wandb_snapshot.sh watch <local-wandb-root> <oilfs-publisher-dir> [interval-sec]
EOF
}

log() {
  printf '[wandb-snapshot] %s\n' "$*"
}

publish_once() {
  local local_root="$1"
  local publisher_dir="$2"
  local keep="${MAST_WANDB_SNAPSHOT_KEEP:-2}"
  local stamp local_archive remote_tmp remote_final
  local snapshots=()
  local remove_count index

  if ! find "${local_root}" -type f -print -quit 2>/dev/null | grep -q .; then
    log "no local W&B files yet under ${local_root}"
    return 0
  fi

  stamp="$(date +%s)-$(date +%N)-$$"
  local_archive="$(mktemp "${TMPDIR:-/tmp}/wandb-snapshot.XXXXXX.tar")"
  remote_tmp="${publisher_dir}/.snapshot-${stamp}.tar.tmp"
  remote_final="${publisher_dir}/snapshot-${stamp}.tar"

  cleanup_publish() {
    rm -f "${local_archive}" "${remote_tmp}" 2>/dev/null || true
  }
  trap cleanup_publish RETURN

  tar -C "${local_root}" -cf "${local_archive}" .
  mkdir -p "${publisher_dir}"
  cp "${local_archive}" "${remote_tmp}"
  mv "${remote_tmp}" "${remote_final}"
  log "published ${remote_final}"

  if [[ "${keep}" =~ ^[1-9][0-9]*$ ]]; then
    mapfile -t snapshots < <(
      find "${publisher_dir}" -maxdepth 1 -type f -name 'snapshot-*.tar' \
        -printf '%f\n' | sort
    )
    remove_count=$((${#snapshots[@]} - keep))
    for ((index = 0; index < remove_count; index++)); do
      rm -f "${publisher_dir}/${snapshots[index]}"
    done
  fi

  trap - RETURN
  cleanup_publish
}

if [[ $# -lt 3 || $# -gt 4 ]]; then
  usage
  exit 2
fi

MODE="$1"
LOCAL_ROOT="$2"
PUBLISHER_DIR="$3"
INTERVAL_SEC="${4:-${MAST_WANDB_SNAPSHOT_INTERVAL_SEC:-60}}"

case "${MODE}" in
  once)
    [[ $# -eq 3 ]] || { usage; exit 2; }
    publish_once "${LOCAL_ROOT}" "${PUBLISHER_DIR}"
    ;;
  watch)
    if [[ ! "${INTERVAL_SEC}" =~ ^[1-9][0-9]*$ ]]; then
      echo "wandb_snapshot.sh: interval must be a positive integer: ${INTERVAL_SEC}" >&2
      exit 2
    fi
    log "watching ${LOCAL_ROOT}; interval=${INTERVAL_SEC}s"
    while true; do
      if ! publish_once "${LOCAL_ROOT}" "${PUBLISHER_DIR}"; then
        log "publish failed; retrying next interval"
      fi
      sleep "${INTERVAL_SEC}"
    done
    ;;
  *)
    usage
    exit 2
    ;;
esac
