#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SYNC_SCRIPT="${REPO_ROOT}/examples/supo_browsecomp/mast/wandb_sync.sh"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

fail() {
  echo "test_mast_wandb_sync.sh: $*" >&2
  exit 1
}

mkdir -p "${TMP_DIR}/bin" "${TMP_DIR}/wandb" "${TMP_DIR}/snapshots" \
  "${TMP_DIR}/cache" "${TMP_DIR}/locks"
printf 'local-wandb_v1_test\n' > "${TMP_DIR}/wandb-key"

cat > "${TMP_DIR}/bin/with-proxy" <<'EOF'
#!/bin/bash
exec "$@"
EOF

cat > "${TMP_DIR}/bin/wandb" <<'EOF'
#!/bin/bash
count=0
[[ -f "${FAKE_WANDB_COUNT_FILE}" ]] && read -r count < "${FAKE_WANDB_COUNT_FILE}"
count=$((count + 1))
printf '%s\n' "${count}" > "${FAKE_WANDB_COUNT_FILE}"
{
  printf 'wandb'
  printf '\t%s' "$@"
  printf '\n'
} >> "${FAKE_WANDB_LOG}"
if ((count <= ${FAKE_WANDB_FAILURES:-0})); then
  exit 1
fi
EOF

cat > "${TMP_DIR}/bin/mast" <<'EOF'
#!/bin/bash
index=0
[[ -f "${FAKE_MAST_INDEX_FILE}" ]] && read -r index < "${FAKE_MAST_INDEX_FILE}"
mapfile -t states < "${FAKE_MAST_STATES_FILE}"
if ((index >= ${#states[@]})); then
  index=$((${#states[@]} - 1))
fi
state="${states[index]}"
printf '%s\n' "$((index + 1))" > "${FAKE_MAST_INDEX_FILE}"
if [[ "${state}" == "ERROR" ]]; then
  exit 1
fi
printf '{"status":"ok","data":{"state":"%s"}}\n' "${state}"
EOF
chmod +x "${TMP_DIR}/bin/"*

export FAKE_WANDB_LOG="${TMP_DIR}/wandb.log"
export FAKE_WANDB_COUNT_FILE="${TMP_DIR}/wandb.count"
export FAKE_MAST_STATES_FILE="${TMP_DIR}/mast.states"
export FAKE_MAST_INDEX_FILE="${TMP_DIR}/mast.index"

run_sync() {
  env \
    MAST_WANDB_ROOT="${TMP_DIR}/wandb" \
    MAST_WANDB_SNAPSHOT_ROOT="${TMP_DIR}/snapshots" \
    WANDB_SYNC_CACHE_DIR="${TMP_DIR}/cache" \
    WANDB_KEY_FILE="${TMP_DIR}/wandb-key" \
    WANDB_BIN="${TMP_DIR}/bin/wandb" \
    WITH_PROXY_BIN="${TMP_DIR}/bin/with-proxy" \
    MAST_BIN="${TMP_DIR}/bin/mast" \
    JQ_BIN="$(command -v jq)" \
    WANDB_SYNC_LOCK_DIR="${TMP_DIR}/locks" \
    WANDB_SYNC_INTERVAL_SEC=0 \
    WANDB_FINAL_SYNC_SETTLE_SEC=0 \
    WANDB_FINAL_SYNC_RETRY_INTERVAL_SEC=0 \
    "$@"
}

reset_fakes() {
  : > "${FAKE_WANDB_LOG}"
  rm -f "${FAKE_WANDB_COUNT_FILE}" "${FAKE_MAST_INDEX_FILE}"
  unset FAKE_WANDB_FAILURES
}

JOB=avocado-test-1234
RUN_ROOT="${TMP_DIR}/wandb/${JOB}/wandb"
mkdir -p "${RUN_ROOT}/offline-run-a" "${RUN_ROOT}/offline-run-b"

reset_fakes
run_sync bash "${SYNC_SCRIPT}" once "${JOB}"
expected="wandb	sync	--append	--no-sync-tensorboard	${RUN_ROOT}/offline-run-a	${RUN_ROOT}/offline-run-b"
grep -Fqx "${expected}" "${FAKE_WANDB_LOG}" || fail "once did not pass the expected sync arguments"

reset_fakes
run_sync bash "${SYNC_SCRIPT}" once no-offline-runs
[[ ! -s "${FAKE_WANDB_LOG}" ]] || fail "no-run sync unexpectedly invoked W&B"

reset_fakes
SNAPSHOT_JOB=avocado-snapshot-1234
SNAPSHOT_SOURCE="${TMP_DIR}/snapshot-source"
SNAPSHOT_PUBLISHER="${TMP_DIR}/snapshots/${SNAPSHOT_JOB}/attempt-0-task-0"
mkdir -p "${SNAPSHOT_SOURCE}/wandb/offline-run-snapshot" "${SNAPSHOT_PUBLISHER}"
printf 'transaction-log\n' > \
  "${SNAPSHOT_SOURCE}/wandb/offline-run-snapshot/run-snapshot.wandb"
tar -C "${SNAPSHOT_SOURCE}" -cf \
  "${SNAPSHOT_PUBLISHER}/snapshot-0000000001.tar" .
run_sync bash "${SYNC_SCRIPT}" once "${SNAPSHOT_JOB}"
expected_snapshot_dir="${TMP_DIR}/cache/${SNAPSHOT_JOB}/attempt-0-task-0/current/wandb/offline-run-snapshot"
grep -Fqx "$(printf 'wandb\tsync\t--append\t--no-sync-tensorboard\t%s' "${expected_snapshot_dir}")" \
  "${FAKE_WANDB_LOG}" || \
  fail "snapshot run was not extracted to local cache before sync: $(cat "${FAKE_WANDB_LOG}")"

if env \
  MAST_WANDB_ROOT="${TMP_DIR}/missing-oilfs-mount" \
  MAST_WANDB_SNAPSHOT_ROOT="${TMP_DIR}/missing-snapshot-mount" \
  WANDB_KEY_FILE="${TMP_DIR}/wandb-key" \
  WANDB_BIN="${TMP_DIR}/bin/wandb" \
  WITH_PROXY_BIN="${TMP_DIR}/bin/with-proxy" \
  bash "${SYNC_SCRIPT}" once "${JOB}" >/dev/null 2>&1; then
  fail "sync accepted a missing OILFS W&B root"
fi

reset_fakes
printf '%s\n' PENDING RUNNING COMPLETE > "${FAKE_MAST_STATES_FILE}"
run_sync bash "${SYNC_SCRIPT}" watch "${JOB}"
[[ "$(wc -l < "${FAKE_WANDB_LOG}")" -eq 3 ]] || fail "watch did not sync during active states and at completion"

reset_fakes
printf '%s\n' ERROR COMPLETE > "${FAKE_MAST_STATES_FILE}"
run_sync bash "${SYNC_SCRIPT}" watch "${JOB}"
[[ "$(wc -l < "${FAKE_WANDB_LOG}")" -eq 2 ]] || fail "watch did not survive a transient MAST status failure"

reset_fakes
printf '%s\n' FAILED > "${FAKE_MAST_STATES_FILE}"
set +e
run_sync timeout 5 bash "${SYNC_SCRIPT}" watch "${JOB}"
rc=$?
set -e
[[ "${rc}" -eq 0 ]] || fail "watch did not exit cleanly at FAILED state: ${rc}"
[[ "$(wc -l < "${FAKE_WANDB_LOG}")" -eq 1 ]] || fail "watch did not perform one final sync at FAILED state"

reset_fakes
export FAKE_WANDB_FAILURES=1
printf '%s\n' COMPLETE > "${FAKE_MAST_STATES_FILE}"
run_sync bash "${SYNC_SCRIPT}" watch "${JOB}"
[[ "$(wc -l < "${FAKE_WANDB_LOG}")" -eq 2 ]] || fail "final sync did not retry"

reset_fakes
flock "${TMP_DIR}/locks/${JOB}.lock" sleep 10 &
lock_holder=$!
for _ in $(seq 1 50); do
  if ! flock -n "${TMP_DIR}/locks/${JOB}.lock" true; then
    break
  fi
  sleep 0.02
done
if run_sync bash "${SYNC_SCRIPT}" once "${JOB}" >/dev/null 2>&1; then
  kill "${lock_holder}" 2>/dev/null || true
  fail "a second sync process acquired the per-job lock"
fi
kill "${lock_holder}" 2>/dev/null || true
wait "${lock_holder}" 2>/dev/null || true

echo "test_mast_wandb_sync.sh: PASS"
