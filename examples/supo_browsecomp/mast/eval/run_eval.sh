#!/bin/bash
# One-checkpoint BC+ eval runner for a single 8-GPU MAST host.
set -euo pipefail

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RAY_AUTH_MODE=disabled
export SGLANG_NUMA_BIND_V2=0
export PYTORCH_CUDA_ALLOC_CONF=""
# Eight independent TP1 engines are all NCCL rank 0 and otherwise race on the
# default timeout-dump pipe. Normal NCCL error reporting remains enabled.
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-0}"
unset TRITON_CACHE_MANAGER
export TRITON_CACHE_DIR=/tmp/triton_cache_slime_eval

unset http_proxy HTTP_PROXY
export https_proxy="http://127.0.0.1:9080" HTTPS_PROXY="http://127.0.0.1:9080"
export no_proxy="127.0.0.1,localhost,::1" NO_PROXY="127.0.0.1,localhost,::1"

SLIME=/slime-src
D=/mnt/wsfuse/hhzhang01/supo-data
STAGE=/mnt/wsfuse/hhzhang01/supo-slime

: "${EVAL_RUN_NAME:?set EVAL_RUN_NAME (or shared-base)}"
: "${EVAL_POINT:?set EVAL_POINT (base or iterNN)}"
: "${EVAL_REQUESTED_STEP:?set EVAL_REQUESTED_STEP (base or integer)}"
: "${EVAL_OUTPUT_DIR:?set EVAL_OUTPUT_DIR}"
: "${EVAL_CODE_ARCHIVE_SHA256:?set EVAL_CODE_ARCHIVE_SHA256}"

EVAL_N="${EVAL_N:-4}"
EVAL_SEED="${EVAL_SEED:-42}"
EVAL_EXPECTED_QUESTIONS="${EVAL_EXPECTED_QUESTIONS:-150}"
BC_MAX_RESPONSE_LEN="${BC_MAX_RESPONSE_LEN:-32768}"
BC_MAX_CONTEXT_LEN="${BC_MAX_CONTEXT_LEN:-65536}"
BCPLUS_MAX_TURNS="${BCPLUS_MAX_TURNS:-64}"
BCPLUS_MAX_SUB_TRAJS="${BCPLUS_MAX_SUB_TRAJS:-5}"
BCPLUS_COMPRESS_THRESH="${BCPLUS_COMPRESS_THRESH:-0.85}"
BCPLUS_SEARCH_CONCURRENCY="${BCPLUS_SEARCH_CONCURRENCY:-64}"
BCPLUS_JUDGE_CONCURRENCY="${BCPLUS_JUDGE_CONCURRENCY:-16}"
BCPLUS_SGLANG_REQUEST_TIMEOUT_SECS="${BCPLUS_SGLANG_REQUEST_TIMEOUT_SECS:-5400}"
BCPLUS_JUDGE_MODEL="${BCPLUS_JUDGE_MODEL:-gpt-5-4-genai-dss4}"
BCPLUS_JUDGE_BASE_URL="${BCPLUS_JUDGE_BASE_URL:-https://api.llama.com/compat/v1/}"
BC_MODEL_SIZE="${BC_MODEL_SIZE:-4B}"

if [[ "${MAST_HPC_TASK_GROUP_SIZE:-1}" != "1" ]]; then
  echo "ERROR: eval runner requires exactly one MAST task, got ${MAST_HPC_TASK_GROUP_SIZE}" >&2
  exit 1
fi
if [[ -e "${EVAL_OUTPUT_DIR}/_SUCCESS" ]]; then
  echo "[eval] already complete: ${EVAL_OUTPUT_DIR}"
  exit 0
fi
if [[ -e "${EVAL_OUTPUT_DIR}/rollout_data/eval_0.pt" ]]; then
  echo "ERROR: stale dump exists without _SUCCESS: ${EVAL_OUTPUT_DIR}" >&2
  exit 1
fi
mkdir -p "${EVAL_OUTPUT_DIR}"

KEY_FILE="${LLAMA_KEY_FILE:-${STAGE}/.llama_key}"
if [[ -z "${LLAMA_API_KEY:-}" && -f "${KEY_FILE}" ]]; then
  export LLAMA_API_KEY="$(tr -d ' \t\r\n' < "${KEY_FILE}")"
fi
: "${LLAMA_API_KEY:?LLAMA_API_KEY is unavailable}"

ADDR_FILE="${SEARCH_ADDR_FILE:-${STAGE}/search-server.addr}"
if [[ -z "${LOCAL_SEARCH_URL:-}" ]]; then
  [[ -f "${ADDR_FILE}" ]] || { echo "ERROR: missing ${ADDR_FILE}" >&2; exit 1; }
  export LOCAL_SEARCH_URL="http://$(tr -d ' \t\r\n' < "${ADDR_FILE}")"
fi
echo "[eval] search=${LOCAL_SEARCH_URL}; waiting for health"
healthy=0
for _ in $(seq 1 72); do
  if curl -g -sf --noproxy '*' --max-time 5 "${LOCAL_SEARCH_URL}/health" >/dev/null; then
    healthy=1
    break
  fi
  sleep 10
done
[[ "${healthy}" = "1" ]] || { echo "ERROR: search server is unhealthy" >&2; exit 1; }

export MASTER_ADDR=127.0.0.1
export SLIME_HOST_IP=127.0.0.1
export CUDA_DEVICE_MAX_CONNECTIONS=1

pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 2

cd "${SLIME}"
case "${BC_MODEL_SIZE,,}" in
  4b)
    MODEL_NAME=Qwen3.5-4B
    MODEL_CONFIG=scripts/models/qwen3.5-4B.sh
    MAX_TOKENS_PER_GPU=49152
    DEFAULT_SGLANG_SERVER_CONCURRENCY=36
    ;;
  9b)
    MODEL_NAME=Qwen3.5-9B
    MODEL_CONFIG=scripts/models/qwen3.5-9B.sh
    MAX_TOKENS_PER_GPU=32768
    DEFAULT_SGLANG_SERVER_CONCURRENCY=32
    ;;
  *)
    echo "ERROR: unsupported BC_MODEL_SIZE=${BC_MODEL_SIZE}; expected 4B or 9B" >&2
    exit 1
    ;;
esac
BCPLUS_SGLANG_SERVER_CONCURRENCY="${BCPLUS_SGLANG_SERVER_CONCURRENCY:-${DEFAULT_SGLANG_SERVER_CONCURRENCY}}"
source "${MODEL_CONFIG}"

HF_CHECKPOINT="${D}/${MODEL_NAME}"
BASE_CHECKPOINT="${D}/${MODEL_NAME}_torch_dist"
for required_path in "${HF_CHECKPOINT}" "${BASE_CHECKPOINT}"; do
  [[ -e "${required_path}" ]] || { echo "ERROR: missing model path ${required_path}" >&2; exit 1; }
done
TRAIN_DATA="${D}/BC+/bc_train.parquet"
TEST_DATA="${D}/BC+/bc_test.parquet"
CHECKPOINT_ROOT="${STAGE}/checkpoints/${EVAL_RUN_NAME}"

CKPT_ARGS=(--hf-checkpoint "${HF_CHECKPOINT}" --no-load-optim --no-load-rng)
if [[ "${EVAL_REQUESTED_STEP}" == "base" ]]; then
  [[ "${EVAL_POINT}" == "base" ]] || { echo "ERROR: base step requires point=base" >&2; exit 1; }
  CHECKPOINT_ROOT="${BASE_CHECKPOINT}"
  CHECKPOINT_METADATA="${BASE_CHECKPOINT}/release/.metadata"
  CKPT_ARGS+=(--load "${BASE_CHECKPOINT}")
else
  [[ "${EVAL_REQUESTED_STEP}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid step ${EVAL_REQUESTED_STEP}" >&2; exit 1; }
  ITER_DIR="${CHECKPOINT_ROOT}/iter_$(printf '%07d' "${EVAL_REQUESTED_STEP}")"
  [[ -f "${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt" ]] || {
    echo "ERROR: checkpoint root lacks tracker: ${CHECKPOINT_ROOT}" >&2
    exit 1
  }
  [[ -f "${ITER_DIR}/.metadata" ]] || { echo "ERROR: missing checkpoint ${ITER_DIR}" >&2; exit 1; }
  CHECKPOINT_METADATA="${ITER_DIR}/.metadata"
  CKPT_ARGS+=(--load "${CHECKPOINT_ROOT}" --ckpt-step "${EVAL_REQUESTED_STEP}")
fi

CHECKPOINT_METADATA_SHA256="$(sha256sum "${CHECKPOINT_METADATA}" | awk '{print $1}')"
DATASET_SHA256="$(sha256sum "${TEST_DATA}" | awk '{print $1}')"
LOCAL_LOG="/tmp/bcplus-eval-${MAST_HPC_JOB_NAME:-local}.log"

ROLLOUT_ARGS=(
  --prompt-data "${TRAIN_DATA}"
  --input-key prompt --label-key answer --metadata-key extra_info
  --num-rollout 0
  --rollout-batch-size 32 --n-samples-per-prompt 4 --global-batch-size 128
  --rollout-max-response-len "${BC_MAX_RESPONSE_LEN}"
  --rollout-max-context-len "${BC_MAX_CONTEXT_LEN}"
  --rollout-temperature 1.0 --rollout-seed "${EVAL_SEED}"
  --sglang-enable-deterministic-inference
)
EVAL_ARGS=(
  --eval-interval 1
  --eval-prompt-data bcplus_test "${TEST_DATA}"
  --n-samples-per-eval-prompt "${EVAL_N}"
  --dump-details "${EVAL_OUTPUT_DIR}"
  --lr-decay-iters 1
)
PERF_ARGS=(
  --tensor-model-parallel-size 4 --sequence-parallel
  --pipeline-model-parallel-size 1 --context-parallel-size 2
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 2
  --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)
OPTIMIZER_ARGS=(
  --optimizer adam --lr 1e-6 --lr-decay-style constant
  --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98
)
SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static 0.8
  --sglang-server-concurrency "${BCPLUS_SGLANG_SERVER_CONCURRENCY}"
  --sglang-router-request-timeout-secs "${BCPLUS_SGLANG_REQUEST_TIMEOUT_SECS}"
  --sglang-disable-custom-all-reduce
)
MISC_ARGS=(
  --attention-dropout 0.0 --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
  --attention-backend flash --log-multi-turn
)
CUSTOM_ARGS=(
  --custom-generate-function-path examples.supo_browsecomp.generate_with_bcplus.generate
  --custom-rm-path examples.supo_browsecomp.generate_with_bcplus.reward_func
  --reward-key score
  --custom-reward-post-process-path examples.supo_browsecomp.generate_with_bcplus.reward_post_process
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SLIME}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"MASTER_ADDR\": \"127.0.0.1\",
    \"SLIME_HOST_IP\": \"127.0.0.1\",
    \"SGLANG_NUMA_BIND_V2\": \"0\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"\",
    \"TORCH_NCCL_DUMP_ON_TIMEOUT\": \"${TORCH_NCCL_DUMP_ON_TIMEOUT}\",
    \"TRITON_CACHE_DIR\": \"/tmp/triton_cache_slime_eval\",
    \"HF_HUB_OFFLINE\": \"1\",
    \"TRANSFORMERS_OFFLINE\": \"1\",
    \"LOCAL_SEARCH_URL\": \"${LOCAL_SEARCH_URL}\",
    \"LLAMA_API_KEY\": \"${LLAMA_API_KEY}\",
    \"http_proxy\": \"\",
    \"HTTP_PROXY\": \"\",
    \"https_proxy\": \"http://127.0.0.1:9080\",
    \"HTTPS_PROXY\": \"http://127.0.0.1:9080\",
    \"no_proxy\": \"127.0.0.1,localhost,::1\",
    \"NO_PROXY\": \"127.0.0.1,localhost,::1\",
    \"BCPLUS_MAX_TURNS\": \"${BCPLUS_MAX_TURNS}\",
    \"BCPLUS_COMPRESS_THRESH\": \"${BCPLUS_COMPRESS_THRESH}\",
    \"BCPLUS_MAX_SUB_TRAJS\": \"${BCPLUS_MAX_SUB_TRAJS}\",
    \"BCPLUS_SEARCH_CONCURRENCY\": \"${BCPLUS_SEARCH_CONCURRENCY}\",
    \"BCPLUS_JUDGE_CONCURRENCY\": \"${BCPLUS_JUDGE_CONCURRENCY}\",
    \"BCPLUS_JUDGE_MODEL\": \"${BCPLUS_JUDGE_MODEL}\",
    \"BCPLUS_JUDGE_BASE_URL\": \"${BCPLUS_JUDGE_BASE_URL}\"
  }
}"

echo "[eval] model=${MODEL_NAME} run=${EVAL_RUN_NAME} point=${EVAL_POINT} step=${EVAL_REQUESTED_STEP} output=${EVAL_OUTPUT_DIR}"
echo "[eval] sglang_tp=1 server_concurrency=${BCPLUS_SGLANG_SERVER_CONCURRENCY} request_timeout_secs=${BCPLUS_SGLANG_REQUEST_TIMEOUT_SECS}"
echo "[eval] torch_nccl_dump_on_timeout=${TORCH_NCCL_DUMP_ON_TIMEOUT}"
ray start --head --node-ip-address=127.0.0.1 --num-gpus 8 \
  --disable-usage-stats --dashboard-host=127.0.0.1 --dashboard-port=8265

set +e
ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 --actor-num-gpus-per-node 8 \
  "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" --advantage-estimator grpo \
  "${PERF_ARGS[@]}" "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}" "${EVAL_ARGS[@]}" --colocate \
  2>&1 | tee "${LOCAL_LOG}"
RC=${PIPESTATUS[0]}
set -e

cp "${LOCAL_LOG}" "${EVAL_OUTPUT_DIR}/eval.log"
ray stop --force 2>/dev/null || true
if [[ "${RC}" != "0" ]]; then
  echo "ERROR: ray eval job failed with rc=${RC}" >&2
  exit "${RC}"
fi

python3 examples/supo_browsecomp/eval/eval_pipeline.py point \
  --dump "${EVAL_OUTPUT_DIR}/rollout_data/eval_0.pt" \
  --output-dir "${EVAL_OUTPUT_DIR}" --load-log "${EVAL_OUTPUT_DIR}/eval.log" \
  --run-name "${EVAL_RUN_NAME}" --point "${EVAL_POINT}" --requested-step "${EVAL_REQUESTED_STEP}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" --checkpoint-metadata-sha256 "${CHECKPOINT_METADATA_SHA256}" \
  --code-archive-sha256 "${EVAL_CODE_ARCHIVE_SHA256}" --dataset-sha256 "${DATASET_SHA256}" \
  --mast-job-name "${MAST_HPC_JOB_NAME:-local}" --model-name "${MODEL_NAME}" \
  --judge-model "${BCPLUS_JUDGE_MODEL}" \
  --search-url "${LOCAL_SEARCH_URL}" --expected-questions "${EVAL_EXPECTED_QUESTIONS}" \
  --samples-per-question "${EVAL_N}" --rollout-seed "${EVAL_SEED}" \
  --max-response-len "${BC_MAX_RESPONSE_LEN}" --max-context-len "${BC_MAX_CONTEXT_LEN}" \
  --max-turns "${BCPLUS_MAX_TURNS}" --max-sub-trajs "${BCPLUS_MAX_SUB_TRAJS}" \
  --compression-threshold "${BCPLUS_COMPRESS_THRESH}"

echo "[eval] complete: ${EVAL_OUTPUT_DIR}"
