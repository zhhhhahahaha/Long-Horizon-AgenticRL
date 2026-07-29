#!/bin/bash
# Validate one staged slim checkpoint with the normal eval path, then promote it.
set -euo pipefail

SLIME=/slime-src
STAGE=/mnt/wsfuse/hhzhang01/supo-slime

: "${SLIM_RUN_NAME:?set SLIM_RUN_NAME}"
: "${SLIM_STEP:?set SLIM_STEP}"
: "${SLIM_BATCH_ID:?set SLIM_BATCH_ID}"
: "${SLIM_EVAL_ALIAS:?set a unique SLIM_EVAL_ALIAS}"
: "${EVAL_OUTPUT_DIR:?set EVAL_OUTPUT_DIR}"
: "${EVAL_CODE_ARCHIVE_SHA256:?set EVAL_CODE_ARCHIVE_SHA256}"

[[ "${SLIM_RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "ERROR: unsafe SLIM_RUN_NAME=${SLIM_RUN_NAME}" >&2
  exit 1
}
[[ "${SLIM_BATCH_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "ERROR: unsafe SLIM_BATCH_ID=${SLIM_BATCH_ID}" >&2
  exit 1
}
[[ "${SLIM_STEP}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: invalid SLIM_STEP=${SLIM_STEP}" >&2
  exit 1
}
[[ "${SLIM_EVAL_ALIAS}" =~ ^__checkpoint_slim_canary_[A-Za-z0-9._-]+$ ]] || {
  echo "ERROR: unsafe SLIM_EVAL_ALIAS=${SLIM_EVAL_ALIAS}" >&2
  exit 1
}

SOURCE_ROOT="${STAGE}/checkpoints/${SLIM_RUN_NAME}"
STAGING_ROOT="${STAGE}/checkpoint-slim/${SLIM_BATCH_ID}/staging/${SLIM_RUN_NAME}"
STATE_DIR="${STAGE}/checkpoint-slim/${SLIM_BATCH_ID}"
ALIAS_ROOT="${STAGE}/checkpoints/${SLIM_EVAL_ALIAS}"
ITER_DIR="${STAGING_ROOT}/iter_$(printf '%07d' "${SLIM_STEP}")"

[[ -f "${ITER_DIR}/.metadata" ]] || {
  echo "ERROR: staged checkpoint is incomplete: ${ITER_DIR}" >&2
  exit 1
}
[[ ! -e "${ALIAS_ROOT}" && ! -L "${ALIAS_ROOT}" ]] || {
  echo "ERROR: canary alias already exists: ${ALIAS_ROOT}" >&2
  exit 1
}

cleanup_alias() {
  if [[ -L "${ALIAS_ROOT}" ]]; then
    unlink "${ALIAS_ROOT}"
  elif [[ -e "${ALIAS_ROOT}" ]]; then
    echo "ERROR: refusing to remove non-symlink alias path: ${ALIAS_ROOT}" >&2
    return 1
  fi
}
trap cleanup_alias EXIT

ln -s "${STAGING_ROOT}" "${ALIAS_ROOT}"
export EVAL_RUN_NAME="${SLIM_EVAL_ALIAS}"
EVAL_POINT="iter$(printf '%02d' "${SLIM_STEP}")"
export EVAL_POINT
export EVAL_REQUESTED_STEP="${SLIM_STEP}"
export EVAL_N=1

bash "${SLIME}/examples/supo_browsecomp/mast/eval/run_eval.sh"

cleanup_alias
trap - EXIT
python3 "${SLIME}/examples/supo_browsecomp/mast/checkpoint_slim/checkpoint_slim.py" promote \
  --source-root "${SOURCE_ROOT}" \
  --staging-root "${STAGING_ROOT}" \
  --state-dir "${STATE_DIR}" \
  --step "${SLIM_STEP}"

echo "[slim-canary] eval passed and iter ${SLIM_STEP} was promoted"
