#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WRAPPER="${REPO_ROOT}/examples/supo_browsecomp/mast/submit_with_wandb.sh"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

fail() {
  echo "test_mast_submit_with_wandb.sh: $*" >&2
  exit 1
}

mkdir -p "${TMP_DIR}/bin" "${TMP_DIR}/tmux-sessions" \
  "${TMP_DIR}/wandb" "${TMP_DIR}/snapshots" "${TMP_DIR}/state"
printf 'local-wandb_v1_test\n' > "${TMP_DIR}/wandb-key"

cat > "${TMP_DIR}/bin/tmux" <<'EOF'
#!/bin/bash
set -euo pipefail
command="$1"
shift
case "${command}" in
  has-session)
    [[ "$1" == "-t" ]]
    [[ -f "${FAKE_TMUX_SESSION_DIR}/$2" ]]
    ;;
  new-session)
    [[ "$1" == "-d" && "$2" == "-s" ]]
    session="$3"
    shift 3
    printf 'new-session\t%s\t%s\n' "${session}" "$*" >> "${FAKE_TMUX_LOG}"
    if [[ "${FAKE_TMUX_DROP_SESSION:-0}" != "1" ]]; then
      touch "${FAKE_TMUX_SESSION_DIR}/${session}"
    fi
    ;;
  *)
    echo "unexpected fake tmux command: ${command}" >&2
    exit 2
    ;;
esac
EOF

cat > "${TMP_DIR}/bin/submit" <<'EOF'
#!/bin/bash
set -euo pipefail
{
  printf 'submit'
  printf '\t%s' "$@"
  printf '\n'
} >> "${FAKE_SUBMIT_LOG}"
case "${FAKE_SUBMIT_MODE:-success}" in
  success)
    printf '{"status":"ok","dryrun":false,"job":{"job_name":"%s","mast_url":"https://mlhub/%s"}}\n' \
      "${FAKE_JOB_NAME:-avocado-test-abcd1234}" "${FAKE_JOB_NAME:-avocado-test-abcd1234}"
    ;;
  dryrun)
    printf '{"status":"ok","dryrun":true,"spec":{}}\n'
    ;;
  error)
    printf '{"status":"error","error":{"message":"submission rejected"}}\n'
    exit 7
    ;;
  invalid)
    printf 'launcher noise without JSON\n'
    ;;
esac
EOF

for command in mast wandb with-proxy; do
  cat > "${TMP_DIR}/bin/${command}" <<'EOF'
#!/bin/bash
exit 0
EOF
done
chmod +x "${TMP_DIR}/bin/"*

export FAKE_TMUX_SESSION_DIR="${TMP_DIR}/tmux-sessions"
export FAKE_TMUX_LOG="${TMP_DIR}/tmux.log"
export FAKE_SUBMIT_LOG="${TMP_DIR}/submit.log"
: > "${FAKE_TMUX_LOG}"
: > "${FAKE_SUBMIT_LOG}"

run_wrapper() {
  env \
    TMUX_BIN="${TMP_DIR}/bin/tmux" \
    JQ_BIN="$(command -v jq)" \
    WITH_PROXY_BIN="${TMP_DIR}/bin/with-proxy" \
    MAST_BIN="${TMP_DIR}/bin/mast" \
    WANDB_BIN="${TMP_DIR}/bin/wandb" \
    WANDB_KEY_FILE="${TMP_DIR}/wandb-key" \
    MAST_WANDB_ROOT="${TMP_DIR}/wandb" \
    MAST_WANDB_SNAPSHOT_ROOT="${TMP_DIR}/snapshots" \
    MAST_WANDB_WATCHER_ROOT="${TMP_DIR}/state" \
    MAST_WANDB_WATCHER_STARTUP_WAIT_SEC=0 \
    FAKE_TMUX_SESSION_DIR="${FAKE_TMUX_SESSION_DIR}" \
    FAKE_TMUX_LOG="${FAKE_TMUX_LOG}" \
    FAKE_SUBMIT_LOG="${FAKE_SUBMIT_LOG}" \
    FAKE_SUBMIT_MODE="${FAKE_SUBMIT_MODE:-success}" \
    FAKE_JOB_NAME="${FAKE_JOB_NAME:-avocado-test-abcd1234}" \
    FAKE_TMUX_DROP_SESSION="${FAKE_TMUX_DROP_SESSION:-0}" \
    bash "${WRAPPER}" "$@"
}

output="$(run_wrapper -- "${TMP_DIR}/bin/submit" --json --region=nha program test)"
grep -Fq 'MAST job submitted: avocado-test-abcd1234' <<< "${output}" || \
  fail "successful submission did not report the job name"
grep -Fq 'watcher started: job=avocado-test-abcd1234' <<< "${output}" || \
  fail "successful submission did not start the watcher"
grep -Fqx $'submit\t--json\t--region=nha\tprogram\ttest' "${FAKE_SUBMIT_LOG}" || \
  fail "wrapper did not preserve submission arguments"
[[ "$(find "${TMP_DIR}/state/avocado-test-abcd1234" -name 'submit-*.json' | wc -l)" -eq 1 ]] || \
  fail "structured submission response was not retained"
[[ "$(grep -c '^new-session' "${FAKE_TMUX_LOG}")" -eq 1 ]] || \
  fail "expected exactly one watcher session"

output="$(run_wrapper watch-only avocado-test-abcd1234)"
grep -Fq 'watcher already running' <<< "${output}" || \
  fail "watch-only did not detect the existing watcher"
[[ "$(grep -c '^new-session' "${FAKE_TMUX_LOG}")" -eq 1 ]] || \
  fail "watch-only duplicated an existing watcher"

export FAKE_SUBMIT_MODE=dryrun
run_wrapper -- "${TMP_DIR}/bin/submit" --json --dryrun >/dev/null
[[ "$(grep -c '^new-session' "${FAKE_TMUX_LOG}")" -eq 1 ]] || \
  fail "dry-run unexpectedly started a watcher"

export FAKE_SUBMIT_MODE=error
set +e
output="$(run_wrapper -- "${TMP_DIR}/bin/submit" --json program test 2>&1)"
rc=$?
set -e
[[ "${rc}" -eq 7 ]] || fail "submission failure exit code was not preserved: ${rc}"
grep -Fq 'submission rejected' <<< "${output}" || fail "submission error was not reported"
[[ "$(grep -c '^new-session' "${FAKE_TMUX_LOG}")" -eq 1 ]] || \
  fail "failed submission unexpectedly started a watcher"

export FAKE_SUBMIT_MODE=invalid
set +e
output="$(run_wrapper -- "${TMP_DIR}/bin/submit" --json program test 2>&1)"
rc=$?
set -e
[[ "${rc}" -eq 2 ]] || fail "invalid JSON should return exit 2, got ${rc}"
grep -Fq 'verify MAST before retrying' <<< "${output}" || \
  fail "ambiguous submission did not warn against retrying"

export FAKE_SUBMIT_MODE=success
before="$(wc -l < "${FAKE_SUBMIT_LOG}")"
set +e
run_wrapper -- "${TMP_DIR}/bin/submit" program test >/dev/null 2>&1
rc=$?
set -e
[[ "${rc}" -eq 1 ]] || fail "missing --json should fail before submission"
[[ "$(wc -l < "${FAKE_SUBMIT_LOG}")" -eq "${before}" ]] || \
  fail "command without --json was submitted"

export FAKE_JOB_NAME=avocado-test-watcherfail
export FAKE_TMUX_DROP_SESSION=1
set +e
output="$(run_wrapper -- "${TMP_DIR}/bin/submit" --json program test 2>&1)"
rc=$?
set -e
[[ "${rc}" -eq 3 ]] || fail "watcher startup failure should return exit 3, got ${rc}"
grep -Fq 'is already submitted, but its watcher did not start' <<< "${output}" || \
  fail "watcher failure did not distinguish the already-submitted job"
grep -Fq 'watch-only avocado-test-watcherfail' <<< "${output}" || \
  fail "watcher failure did not print the recovery command"

unset FAKE_TMUX_DROP_SESSION
output="$(run_wrapper watch-only avocado-test-watcherfail)"
grep -Fq 'watcher started: job=avocado-test-watcherfail' <<< "${output}" || \
  fail "watch-only did not recover the missing watcher"

echo "test_mast_submit_with_wandb.sh: PASS"
