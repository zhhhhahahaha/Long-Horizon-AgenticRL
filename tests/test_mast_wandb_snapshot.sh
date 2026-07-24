#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SNAPSHOT_SCRIPT="${REPO_ROOT}/examples/supo_browsecomp/mast/wandb_snapshot.sh"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

fail() {
  echo "test_mast_wandb_snapshot.sh: $*" >&2
  exit 1
}

LOCAL_ROOT="${TMP_DIR}/local"
PUBLISHER_DIR="${TMP_DIR}/oilfs/job/attempt-0-task-0"
RUN_DIR="${LOCAL_ROOT}/wandb/offline-run-test"
mkdir -p "${RUN_DIR}/files"
printf 'record-1\n' > "${RUN_DIR}/run-test.wandb"
printf 'config\n' > "${RUN_DIR}/files/config.yaml"

MAST_WANDB_SNAPSHOT_KEEP=2 bash "${SNAPSHOT_SCRIPT}" once \
  "${LOCAL_ROOT}" "${PUBLISHER_DIR}"
mapfile -t snapshots < <(find "${PUBLISHER_DIR}" -maxdepth 1 -name 'snapshot-*.tar' | sort)
[[ ${#snapshots[@]} -eq 1 ]] || fail "expected one published snapshot"
find "${PUBLISHER_DIR}" -maxdepth 1 -name '*.tmp' | grep -q . && \
  fail "temporary OILFS file remained visible"
tar -xOf "${snapshots[0]}" ./wandb/offline-run-test/run-test.wandb | \
  grep -Fqx 'record-1' || fail "snapshot did not contain the local transaction log"

printf 'record-2\n' >> "${RUN_DIR}/run-test.wandb"
MAST_WANDB_SNAPSHOT_KEEP=1 bash "${SNAPSHOT_SCRIPT}" once \
  "${LOCAL_ROOT}" "${PUBLISHER_DIR}"
mapfile -t snapshots < <(find "${PUBLISHER_DIR}" -maxdepth 1 -name 'snapshot-*.tar' | sort)
[[ ${#snapshots[@]} -eq 1 ]] || fail "snapshot retention did not prune the old archive"
tar -xOf "${snapshots[0]}" ./wandb/offline-run-test/run-test.wandb | \
  grep -Fqx 'record-2' || fail "latest snapshot did not contain appended data"

EMPTY_ROOT="${TMP_DIR}/empty"
mkdir -p "${EMPTY_ROOT}"
bash "${SNAPSHOT_SCRIPT}" once "${EMPTY_ROOT}" "${TMP_DIR}/empty-publisher"
[[ ! -e "${TMP_DIR}/empty-publisher" ]] || fail "empty local root published a snapshot"

echo "test_mast_wandb_snapshot.sh: PASS"
