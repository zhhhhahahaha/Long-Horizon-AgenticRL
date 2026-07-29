#!/bin/bash
# Rewrite selected intermediate checkpoints for one completed 4B run without optimizer/RNG state.
set -euo pipefail

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=""
export NCCL_CUMEM_ENABLE=0
export SGLANG_NUMA_BIND_V2=0
unset TRITON_CACHE_MANAGER
export TRITON_CACHE_DIR=/tmp/triton_cache_slime_ckpt

SLIME=/slime-src
D=/mnt/wsfuse/hhzhang01/supo-data
STAGE=/mnt/wsfuse/hhzhang01/supo-slime
export PYTHONPATH="/root/Megatron-LM:${SLIME}${PYTHONPATH:+:${PYTHONPATH}}"

: "${SLIM_RUN_NAME:?set SLIM_RUN_NAME}"
: "${SLIM_STEPS:?set SLIM_STEPS as comma-separated iterations}"
: "${SLIM_BATCH_ID:?set SLIM_BATCH_ID}"

[[ "${SLIM_RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "ERROR: unsafe SLIM_RUN_NAME=${SLIM_RUN_NAME}" >&2
  exit 1
}
[[ "${SLIM_BATCH_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "ERROR: unsafe SLIM_BATCH_ID=${SLIM_BATCH_ID}" >&2
  exit 1
}
[[ "${SLIM_STEPS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
  echo "ERROR: invalid SLIM_STEPS=${SLIM_STEPS}" >&2
  exit 1
}
[[ "${SLIM_PROMOTE:-0}" =~ ^[01]$ ]] || {
  echo "ERROR: SLIM_PROMOTE must be 0 or 1" >&2
  exit 1
}

SOURCE_ROOT="${STAGE}/checkpoints/${SLIM_RUN_NAME}"
STAGING_ROOT="${STAGE}/checkpoint-slim/${SLIM_BATCH_ID}/staging/${SLIM_RUN_NAME}"
STATE_DIR="${STAGE}/checkpoint-slim/${SLIM_BATCH_ID}"
HF_CHECKPOINT="${D}/Qwen3.5-4B"
SLIM_TOOL="${SLIME}/examples/supo_browsecomp/mast/checkpoint_slim/checkpoint_slim.py"

[[ -f "${SOURCE_ROOT}/latest_checkpointed_iteration.txt" ]] || {
  echo "ERROR: missing source tracker: ${SOURCE_ROOT}" >&2
  exit 1
}
python3 "${SLIM_TOOL}" plan \
  --checkpoint-root "${STAGE}/checkpoints" \
  --run "${SLIM_RUN_NAME}"

mkdir -p "${STAGING_ROOT}" "${STATE_DIR}/logs"
cd "${SLIME}"
# shellcheck source=/dev/null
source scripts/models/qwen3.5-4B.sh

FIRST_STEP="${SLIM_STEPS%%,*}"
PROMOTE_ARG=()
if [[ "${SLIM_PROMOTE:-0}" == "1" ]]; then
  PROMOTE_ARG=(--slim-promote)
fi

LOG_PATH="${STATE_DIR}/logs/${SLIM_RUN_NAME}-$(printf '%s' "${SLIM_STEPS}" | tr ',' '_').log"

torchrun --standalone --nproc-per-node=8 \
  examples/supo_browsecomp/mast/checkpoint_slim/checkpoint_slim.py worker \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --load "${SOURCE_ROOT}" \
  --save "${STAGING_ROOT}" \
  --ckpt-step "${FIRST_STEP}" \
  --slim-run-name "${SLIM_RUN_NAME}" \
  --slim-steps "${SLIM_STEPS}" \
  --slim-state-dir "${STATE_DIR}" \
  "${PROMOTE_ARG[@]}" \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 2 \
  --sequence-parallel \
  --micro-batch-size 1 \
  --global-batch-size 1 \
  --seq-length 4096 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --attention-softmax-in-fp32 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-backend flash \
  --no-load-optim \
  --no-load-rng \
  --no-save-optim \
  --no-save-rng \
  --ckpt-format torch_dist \
  2>&1 | tee "${LOG_PATH}"

echo "[slim] complete run=${SLIM_RUN_NAME} steps=${SLIM_STEPS} promote=${SLIM_PROMOTE:-0}"
