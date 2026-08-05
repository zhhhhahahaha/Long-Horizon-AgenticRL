#!/bin/bash
# SUPO / BrowseComp-Plus RL trainer — docker-on-MAST (Path B), colocate.
#
# Runs INSIDE the slime container (launcher extracts slime-code.tgz to /slime-src
# then invokes this file). Merges everything proven on MAST:
#   * the 7 sanity env fixes (RAY_AUTH, numactl, triton, alloc-conf, ...) EXCEPT
#     the blanket proxy-clear — SUPO needs judge egress, so instead:
#   * proxy split: http_proxy EMPTY (search + sglang health go direct over the
#     backend net) + https_proxy=RELAY (judge https://api.llama.com via the
#     host-side CONNECT relay auto-started by --docker_host_cmd on 127.0.0.1:9080)
#   * search discovery: read the search server's [ipv6]:port from OILFS, wait /health
#   * LLAMA_API_KEY from an OILFS file (see note on secret hygiene)
#   * wandb ONLINE or OFFLINE → node-local disk → atomic OILFS snapshots
#   * Ray head/worker election from MAST_HPC_TASK_GROUP_HOSTNAMES (multi-node)
#
# Defaults are a 1-node (8 GPU) 4B SMOKE: TP=4, CP=2 → DP=1, tiny batch,
# 2 rollouts. Set BC_MODEL_SIZE=9B for the 9B topology; scale the remaining
# BC_* knobs once the pipeline is green.
set -uo pipefail

# --------------------------- env fixes (MAST vs image) ----------------------
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RAY_AUTH_MODE=disabled                 # Ray 2.55 token auth off (before ray start)
export SGLANG_NUMA_BIND_V2=0                  # broken bind-mounted numactl
export PYTORCH_CUDA_ALLOC_CONF=""             # torch_memory_saver(colocate) vs expandable_segments
export WANDB_X_FLUSH_INTERVAL_SECONDS="${WANDB_X_FLUSH_INTERVAL_SECONDS:-30}"
unset TRITON_CACHE_MANAGER                    # msl_tools.* unimportable here
export TRITON_CACHE_DIR=/tmp/triton_cache_slime

# --------------------------- proxy: judge via relay, rest direct ------------
# http_proxy EMPTY → all HTTP (search server, sglang _wait_server_healthy) direct.
# https_proxy = host relay → only HTTPS (judge api.llama.com) is proxied out.
unset http_proxy HTTP_PROXY
export https_proxy="http://127.0.0.1:9080" HTTPS_PROXY="http://127.0.0.1:9080"
export no_proxy="127.0.0.1,localhost,::1" NO_PROXY="127.0.0.1,localhost,::1"

SLIME=/slime-src
D=/mnt/wsfuse/hhzhang01/supo-data
STAGE=/mnt/wsfuse/hhzhang01/supo-slime
DEFAULT_TRAIN_DATA="${D}/BC+/bc_train_exclude_stable91_20260730.parquet"
TRAIN_DATA="${BC_TRAIN_DATA:-${DEFAULT_TRAIN_DATA}}"
if [[ "${TRAIN_DATA}" != /* ]]; then
  echo "ERROR: BC_TRAIN_DATA must be an absolute container path: ${TRAIN_DATA}" >&2
  exit 1
fi
if [[ ! -r "${TRAIN_DATA}" ]]; then
  echo "ERROR: training data is not readable: ${TRAIN_DATA}" >&2
  exit 1
fi
echo "[trainer] training data=${TRAIN_DATA}"
RUN_NAME="${BC_RUN_NAME:-${RUN_NAME:-${MAST_HPC_JOB_NAME:-supo-bcplus-mast-local}}}"
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ || "${RUN_NAME}" == "." || "${RUN_NAME}" == ".." ]]; then
  echo "ERROR: invalid checkpoint run name: ${RUN_NAME}" >&2
  exit 1
fi
BCPLUS_FIXED_SEARCH_TOPK="${BCPLUS_FIXED_SEARCH_TOPK:-}"
BCPLUS_DOC_WORDS_FULL="${BCPLUS_DOC_WORDS_FULL:-4096}"
if [[ -n "${BCPLUS_FIXED_SEARCH_TOPK}" && ! "${BCPLUS_FIXED_SEARCH_TOPK}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: BCPLUS_FIXED_SEARCH_TOPK must be a positive integer or empty" >&2
  exit 2
fi
if [[ ! "${BCPLUS_DOC_WORDS_FULL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: BCPLUS_DOC_WORDS_FULL must be a positive integer" >&2
  exit 2
fi
export BCPLUS_FIXED_SEARCH_TOPK BCPLUS_DOC_WORDS_FULL
if [[ -n "${BC_RUN_NAME:-}" ]]; then
  RESUME_TRACKER="${STAGE}/checkpoints/${RUN_NAME}/latest_checkpointed_iteration.txt"
  if [[ ! -f "${RESUME_TRACKER}" ]]; then
    echo "ERROR: BC_RUN_NAME requests resume from ${RUN_NAME}, but ${RESUME_TRACKER} does not exist" >&2
    exit 1
  fi
  RESUME_ITERATION="$(tr -d '[:space:]' < "${RESUME_TRACKER}")"
  if [[ ! "${RESUME_ITERATION}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid checkpoint tracker in ${RESUME_TRACKER}: ${RESUME_ITERATION}" >&2
    exit 1
  fi
  echo "[trainer] explicit resume run=${RUN_NAME} tracker=${RESUME_ITERATION}"
fi
cd "${SLIME}"

# --------------------------- LLAMA_API_KEY (judge) --------------------------
KEY_FILE="${LLAMA_KEY_FILE:-${STAGE}/.llama_key}"
if [[ -z "${LLAMA_API_KEY:-}" && -f "${KEY_FILE}" ]]; then
  export LLAMA_API_KEY="$(tr -d ' \t\r\n' < "${KEY_FILE}")"
fi
if [[ -z "${LLAMA_API_KEY:-}" ]]; then
  echo "ERROR: LLAMA_API_KEY not set and ${KEY_FILE} missing." >&2
  echo "       Stage the key (chmod 600) then resubmit. See memory q6j1v1sc note." >&2
  exit 1
fi

# --------------------------- search server discovery ------------------------
ADDR_FILE="${SEARCH_ADDR_FILE:-${STAGE}/search-server.addr}"
if [[ -z "${LOCAL_SEARCH_URL:-}" ]]; then
  if [[ ! -f "${ADDR_FILE}" ]]; then
    echo "ERROR: ${ADDR_FILE} missing — start the search server job first." >&2
    exit 1
  fi
  SEARCH_TARGET="$(tr -d ' \t\r\n' < "${ADDR_FILE}")"      # e.g. [2401:db00:..]:8000
  export LOCAL_SEARCH_URL="http://${SEARCH_TARGET}"
fi
echo "[trainer] LOCAL_SEARCH_URL=${LOCAL_SEARCH_URL}"
echo "[trainer] waiting for search /health (up to 12 min)..."
ok=0
for i in $(seq 1 72); do
  if curl -sf --noproxy '*' --max-time 5 "${LOCAL_SEARCH_URL}/health" >/dev/null 2>&1; then
    echo "[trainer] search healthy after ~$((i*10))s"; ok=1; break
  fi
  sleep 10
done
[[ "${ok}" = "1" ]] || { echo "ERROR: search server never became healthy" >&2; exit 1; }

# --------------------------- Ray head/worker election -----------------------
HOSTS="${MAST_HPC_TASK_GROUP_HOSTNAMES:-$(hostname)}"
HEAD_HOST="$(echo "${HOSTS}" | cut -d, -f1)"
NNODES="$(echo "${HOSTS}" | tr ',' '\n' | grep -c .)"
MYHOST="$(hostname)"
IS_HEAD=0
if [[ "${TW_TASK_ID:-0}" = "0" || "${MYHOST}" = "${HEAD_HOST}" ]]; then IS_HEAD=1; fi
echo "[trainer] nnodes=${NNODES} host=${MYHOST} head=${HEAD_HOST} is_head=${IS_HEAD}"
if [[ "${IS_HEAD}" == "1" ]]; then
  echo "[trainer] tool protocol: search_topk=${BCPLUS_FIXED_SEARCH_TOPK:-model} open_words=${BCPLUS_DOC_WORDS_FULL}"
fi
if [[ -n "${BC_EXPECTED_NUM_NODES:-}" ]]; then
  [[ "${BC_EXPECTED_NUM_NODES}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: BC_EXPECTED_NUM_NODES must be a positive integer, got ${BC_EXPECTED_NUM_NODES}" >&2
    exit 1
  }
  [[ "${NNODES}" == "${BC_EXPECTED_NUM_NODES}" ]] || {
    echo "ERROR: MAST assigned ${NNODES} trainer nodes; experiment requires ${BC_EXPECTED_NUM_NODES}" >&2
    exit 1
  }
fi

pkill -9 sglang 2>/dev/null || true; sleep 2
ray stop --force 2>/dev/null || true; pkill -9 python 2>/dev/null || true; sleep 2

TRAINER_ATTEMPT_ID="${MAST_HPC_JOB_ATTEMPT_INDEX:-0}-${MAST_HPC_TASK_GROUP_ATTEMPT_EPOCH:-0}"
# A task-group retry keeps MAST_HPC_JOB_NAME but starts a different Ray head.
# Isolate coordination by attempt so an early worker cannot read the previous
# attempt's head.ip or DONE marker while the new head is still starting.
COORD_DIR="${STAGE}/ray-coord/${MAST_HPC_JOB_NAME:-supo-local}/attempt-${TRAINER_ATTEMPT_ID}"
HEAD_IP_FILE="${COORD_DIR}/head.ip"
DONE_FILE="${COORD_DIR}/done"

if [[ "${NNODES}" = "1" ]]; then
  MASTER_ADDR=127.0.0.1
else
  MASTER_ADDR="$(hostname -i | tr ' ' '\n' | grep ':' | grep -vE '^(::1|fe80)' | head -1)"
fi
export MASTER_ADDR
# PER-NODE routable IP for slime's get_host_info() override. On MAST the hostname
# resolves to loopback, so without this get_host_info() returns 127.0.0.1 and the
# sgl-router binds to 127.0.0.1 on the head — sglang engines on OTHER nodes then
# fail to reach it (multinode: "Connection refused to 127.0.0.1:<router_port>").
# Must be each node's OWN IP (engines use it for their server_host too), so it is
# set here per-node and deliberately NOT propagated via RUNTIME_ENV_JSON (which
# would globalize the head's IP to every node). Harmless at 1 node (=127.0.0.1).
export SLIME_HOST_IP="${MASTER_ADDR}"

# Cross-node NCCL OOB bootstrap fix (Tupperware Netns / IP-per-task reservations).
# MAST hardcodes NCCL_SOCKET_IFNAME=beth0 for every job (nccl_env.py) — that is
# the BACKEND RoCE NIC, whose per-task `…bace…` address is NOT routable container-
# to-container on a Netns-onboarded reservation, so NCCL's TCP OOB bootstrap times
# out (socketPollConnect ... Connection timed out). The RoCE DATA plane (mlx5
# verbs) is unaffected. Fix: move the OOB bootstrap onto the FRONTEND task NIC that
# MAST assigns — its name is in $TW_TASK_ASSIGNED_IFNAMES (the same NIC Ray joins
# on cross-node) — and drop the beth-oriented NCCL_SOCKET_IPADDR_PREFIX=2401 hint.
NCCL_OOB_IFNAME="${TW_TASK_ASSIGNED_IFNAMES:-eth0}"
export NCCL_SOCKET_IFNAME="${NCCL_OOB_IFNAME}"
export NCCL_CLIENT_SOCKET_IFNAME="${NCCL_OOB_IFNAME}"   # keep == NCCL_SOCKET_IFNAME (mismatch → "CVAR incompatible")
unset NCCL_SOCKET_IPADDR_PREFIX                          # beth-oriented 2803/2401 prefix hint would mis-match in NetNS
# torch/c10d also needs the frontend NIC: gloo PGs (CPU coordination, e.g. the
# checkpoint-save barrier) default to beth0 too and would hang cross-node in NetNS.
# (TCPStore is fine — it binds MASTER_ADDR, which we already set to the eth0/face IP.)
export GLOO_SOCKET_IFNAME="${NCCL_OOB_IFNAME}"
echo "[trainer] NCCL/GLOO OOB iface -> ${NCCL_OOB_IFNAME} (frontend; TW_TASK_ASSIGNED_IFNAMES='${TW_TASK_ASSIGNED_IFNAMES:-<unset>}'); data plane stays on mlx5 RoCE"

# W&B's transaction log cannot be written safely to OILFS (wandb-core can fail
# close(2) with EOVERFLOW and leave an empty run). Online mode still writes a
# local transaction log, so both modes publish immutable recovery snapshots to
# OILFS. The online path uploads live and keeps snapshots for disaster recovery;
# the offline path is uploaded by the devserver watcher.
BC_WANDB_MODE="${BC_WANDB_MODE:-offline}"
case "${BC_WANDB_MODE}" in
  online|offline) ;;
  *)
    echo "ERROR: BC_WANDB_MODE must be online or offline, got ${BC_WANDB_MODE}" >&2
    exit 2
    ;;
esac
WANDB_BASE_URL="${BC_WANDB_HOST:-https://meta-3.wandb.io}"
WANDB_HTTPS_PROXY="${MAST_WANDB_HTTPS_PROXY:-http://fwdproxy:8080}"
export WANDB_BASE_URL WANDB_HTTPS_PROXY

WANDB_LOCAL_KEY_FILE=""
if [[ "${BC_WANDB_MODE}" == "online" ]]; then
  WANDB_SHARED_KEY_FILE="${MAST_WANDB_KEY_FILE:-${STAGE}/.wandb-online-key}"
  WANDB_LOCAL_KEY_FILE="/tmp/slime-wandb-key-${RUN_NAME}"
  wandb_key=""
  for attempt in $(seq 1 6); do
    if [[ -r "${WANDB_SHARED_KEY_FILE}" ]]; then
      wandb_key="$(tr -d ' \t\r\n' < "${WANDB_SHARED_KEY_FILE}")"
      [[ -n "${wandb_key}" ]] && break
    fi
    echo "[trainer] waiting for W&B key on OILFS (${attempt}/6): ${WANDB_SHARED_KEY_FILE}" >&2
    sleep 5
  done
  if [[ -z "${wandb_key}" ]]; then
    echo "ERROR: W&B online mode could not read a non-empty key from ${WANDB_SHARED_KEY_FILE}" >&2
    exit 1
  fi
  key_tmp="${WANDB_LOCAL_KEY_FILE}.tmp-$$"
  (umask 077; printf '%s\n' "${wandb_key}" > "${key_tmp}")
  chmod 600 "${key_tmp}"
  mv -f "${key_tmp}" "${WANDB_LOCAL_KEY_FILE}"
  unset wandb_key key_tmp
  echo "[trainer] W&B mode=online host=${WANDB_BASE_URL} proxy=${WANDB_HTTPS_PROXY} key=node-local"
else
  echo "[trainer] W&B mode=offline"
fi

WANDB_ATTEMPT_ID="${TRAINER_ATTEMPT_ID}"
WANDB_DIR="${MAST_WANDB_LOCAL_DIR:-/tmp/slime-wandb/${RUN_NAME}/attempt-${WANDB_ATTEMPT_ID}}"
WANDB_PUBLISHER_DIR="${STAGE}/wandb-snapshots/${RUN_NAME}/attempt-${WANDB_ATTEMPT_ID}-task-${TW_TASK_ID:-0}"
WANDB_SNAPSHOT_SCRIPT="${SLIME}/examples/supo_browsecomp/mast/wandb/wandb_snapshot.sh"
rm -rf "${WANDB_DIR}"
mkdir -p "${WANDB_DIR}"
bash "${WANDB_SNAPSHOT_SCRIPT}" watch "${WANDB_DIR}" "${WANDB_PUBLISHER_DIR}" \
  "${MAST_WANDB_SNAPSHOT_INTERVAL_SEC:-60}" &
WANDB_SNAPSHOT_PID=$!
echo "[trainer] W&B local=${WANDB_DIR} snapshots=${WANDB_PUBLISHER_DIR} mode=${BC_WANDB_MODE}"

RAY_LOG_COPY_TIMEOUT_SEC="${MAST_RAY_LOG_COPY_TIMEOUT_SEC:-120}"
if ! [[ "${RAY_LOG_COPY_TIMEOUT_SEC}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[trainer] WARN: invalid MAST_RAY_LOG_COPY_TIMEOUT_SEC=${RAY_LOG_COPY_TIMEOUT_SEC}; using 120" >&2
  RAY_LOG_COPY_TIMEOUT_SEC=120
fi

on_trainer_exit() {
  local rc=$?
  trap - EXIT INT TERM

  if [[ "${IS_HEAD}" = "1" ]]; then
    echo "[head] EXIT trap: touch DONE"
    touch "${DONE_FILE}" 2>/dev/null || true
  fi

  kill "${WANDB_SNAPSHOT_PID}" 2>/dev/null || true
  wait "${WANDB_SNAPSHOT_PID}" 2>/dev/null || true
  if ! bash "${WANDB_SNAPSHOT_SCRIPT}" once "${WANDB_DIR}" "${WANDB_PUBLISHER_DIR}"; then
    echo "[trainer] WARN: final W&B snapshot failed" >&2
  fi
  [[ -z "${WANDB_LOCAL_KEY_FILE}" ]] || rm -f "${WANDB_LOCAL_KEY_FILE}"

  if [[ "${IS_HEAD}" = "1" && "${MAST_PERSIST_RAY_LOGS:-1}" != "0" ]]; then
    echo "[head] persisting Ray logs (timeout=${RAY_LOG_COPY_TIMEOUT_SEC}s)"
    if ! timeout --signal=TERM --kill-after=10s "${RAY_LOG_COPY_TIMEOUT_SEC}s" \
      bash -c '
        set -e
        dest=$1
        rm -rf "${dest}"
        mkdir -p "${dest}"
        cp -rL /var/tmp/ray/session_*/logs "${dest}/"
      ' _ "${STAGE}/raylogs/${RUN_NAME}"; then
      echo "[head] WARN: Ray log persistence timed out or failed; training result is unchanged" >&2
    fi
  fi
  return "${rc}"
}
trap on_trainer_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ----- worker branch (multi-node only): join ray, wait for DONE -----
# COORD_DIR is attempt-specific, and the worker still re-reads head.ip and retries
# transient join failures. Keep a timeout around ray start: without it, an
# unreachable GCS can wait forever and the retry loop never advances.
if [[ "${IS_HEAD}" = "0" ]]; then
  joined=0
  for attempt in $(seq 1 120); do          # up to ~20 min of pull-skew + retries
    if [[ ! -f "${HEAD_IP_FILE}" ]]; then sleep 10; continue; fi
    HEAD_IP="$(tr -d ' \t\r\n' < "${HEAD_IP_FILE}")"
    [[ -z "${HEAD_IP}" ]] && { sleep 10; continue; }
    echo "[worker ${MYHOST}] attempt ${attempt}: joining ray at [${HEAD_IP}]:6379"
    ray stop --force 2>/dev/null || true
    if timeout --signal=TERM --kill-after=10s 90s \
         ray start --address="[${HEAD_IP}]:6379" --num-gpus 8 \
         --node-ip-address "${MASTER_ADDR}" --disable-usage-stats; then
      joined=1; break
    fi
    echo "[worker ${MYHOST}] join failed or timed out; retrying with current head.ip"
    sleep 10
  done
  [[ "${joined}" = "1" ]] || { echo "ERROR: worker never joined ray" >&2; exit 1; }
  echo "[worker ${MYHOST}] joined ray; waiting for DONE"
  while [[ ! -f "${DONE_FILE}" ]]; do sleep 30; done
  echo "[worker ${MYHOST}] saw DONE, exiting"; ray stop --force || true; exit 0
fi

# --------------------------- head branch: config + submit -------------------
# Clean this attempt's namespace before starting Ray and publishing head.ip.
rm -rf "${COORD_DIR}" 2>/dev/null || true
mkdir -p "${COORD_DIR}"

BC_MODEL_SIZE="${BC_MODEL_SIZE:-4B}"
case "${BC_MODEL_SIZE,,}" in
  4b)
    BC_MODEL_SIZE=4B
    MODEL_NAME=Qwen3.5-4B
    MODEL_CONFIG=scripts/models/qwen3.5-4B.sh
    DEFAULT_MAX_TOKENS_PER_GPU=49152
    DEFAULT_SGLANG_TP=2
    ;;
  9b)
    BC_MODEL_SIZE=9B
    MODEL_NAME=Qwen3.5-9B
    MODEL_CONFIG=scripts/models/qwen3.5-9B.sh
    DEFAULT_MAX_TOKENS_PER_GPU=32768
    DEFAULT_SGLANG_TP=4
    ;;
  *)
    echo "ERROR: unsupported BC_MODEL_SIZE=${BC_MODEL_SIZE}; expected 4B or 9B" >&2
    exit 1
    ;;
esac

HF_CHECKPOINT="${D}/${MODEL_NAME}"
REF_CHECKPOINT="${D}/${MODEL_NAME}_torch_dist"
for required_path in "${MODEL_CONFIG}" "${HF_CHECKPOINT}" "${REF_CHECKPOINT}"; do
  [[ -e "${required_path}" ]] || { echo "ERROR: required model path missing: ${required_path}" >&2; exit 1; }
done

TRAIN_TP="${BC_TP:-4}"
TRAIN_CP="${BC_CP:-2}"
MAX_TOKENS_PER_GPU="${BC_MAX_TOKENS_PER_GPU:-${DEFAULT_MAX_TOKENS_PER_GPU}}"
SGLANG_TP="${BC_SGLANG_TP:-${DEFAULT_SGLANG_TP}}"
LOG_PROBS_CHUNK_SIZE="${BC_LOG_PROBS_CHUNK_SIZE:-}"
SGLANG_MEM_FRACTION_STATIC="${BC_SGLANG_MEM_FRACTION_STATIC:-0.7}"
for value_name in TRAIN_TP TRAIN_CP SGLANG_TP MAX_TOKENS_PER_GPU; do
  value="${!value_name}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: ${value_name} must be a positive integer, got ${value}" >&2
    exit 1
  }
done
if [[ -n "${LOG_PROBS_CHUNK_SIZE}" && ! "${LOG_PROBS_CHUNK_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: LOG_PROBS_CHUNK_SIZE must be a positive integer, got ${LOG_PROBS_CHUNK_SIZE}" >&2
  exit 1
fi
if [[ ! "${SGLANG_MEM_FRACTION_STATIC}" =~ ^(0\.[0-9]*[1-9][0-9]*|1(\.0+)?)$ ]]; then
  echo "ERROR: SGLANG_MEM_FRACTION_STATIC must be in (0, 1], got ${SGLANG_MEM_FRACTION_STATIC}" >&2
  exit 1
fi
TOTAL_GPUS=$((NNODES * 8))
MODEL_PARALLEL_SIZE=$((TRAIN_TP * TRAIN_CP))
(( TOTAL_GPUS % MODEL_PARALLEL_SIZE == 0 )) || {
  echo "ERROR: total GPUs ${TOTAL_GPUS} must be divisible by TP*CP=${MODEL_PARALLEL_SIZE}" >&2
  exit 1
}
(( TOTAL_GPUS % SGLANG_TP == 0 )) || {
  echo "ERROR: total GPUs ${TOTAL_GPUS} must be divisible by SGLANG_TP=${SGLANG_TP}" >&2
  exit 1
}
if [[ "${BC_MODEL_SIZE}" == "9B" && "${TRAIN_TP}" != "4" ]]; then
  echo "[head] WARN: 9B TP=${TRAIN_TP} is experimental; TP=8 previously hit the Qwen3.5 output-gate/KV-replication bug" >&2
fi
if [[ "${BC_MODEL_SIZE}" == "9B" && ( "${TRAIN_CP}" != "2" || "${SGLANG_TP}" != "4" ) ]]; then
  echo "[head] WARN: non-default 9B topology CP=${TRAIN_CP} SGLANG_TP=${SGLANG_TP}; validate memory and throughput before a full run" >&2
fi
echo "[head] model=${BC_MODEL_SIZE} candidate topology: config=${MODEL_CONFIG} TP=${TRAIN_TP} CP=${TRAIN_CP} DP=$((TOTAL_GPUS / MODEL_PARALLEL_SIZE)) SGLANG_TP=${SGLANG_TP} max_tokens_per_gpu=${MAX_TOKENS_PER_GPU} log_probs_chunk_size=${LOG_PROBS_CHUNK_SIZE:-default} sglang_mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}"
source "${MODEL_CONFIG}"

CKPT_SAVE_DIR="${STAGE}/checkpoints/${RUN_NAME}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load      "${REF_CHECKPOINT}"
)
case "$(printf '%s' "${BC_SLIM_INTERMEDIATE_CHECKPOINTS:-0}" | tr '[:upper:]' '[:lower:]')" in
   1|true) SLIM_INTERMEDIATE_CHECKPOINTS=1 ;;
   ""|0|false) SLIM_INTERMEDIATE_CHECKPOINTS=0 ;;
   *)
      echo "ERROR: BC_SLIM_INTERMEDIATE_CHECKPOINTS must be one of: 1, true, 0, false" >&2
      exit 2
      ;;
esac
if [[ "${BC_SAVE_INTERVAL:-5}" == "0" ]]; then
   if [[ "${SLIM_INTERMEDIATE_CHECKPOINTS}" == "1" ]]; then
      echo "ERROR: rolling checkpoint slimming requires BC_SAVE_INTERVAL > 0" >&2
      exit 2
   fi
   echo "[head] checkpoint saving disabled (BC_SAVE_INTERVAL=0)"
else
   mkdir -p "${CKPT_SAVE_DIR}"
   CKPT_ARGS+=(--save "${CKPT_SAVE_DIR}" --save-interval "${BC_SAVE_INTERVAL:-5}")
   if [[ "${SLIM_INTERMEDIATE_CHECKPOINTS}" == "1" ]]; then
      CKPT_ARGS+=(--slim-intermediate-checkpoints)
      echo "[head] rolling checkpoint slimming enabled"
   fi
   if [[ -f "${CKPT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
      echo "[head] resuming from ${CKPT_SAVE_DIR}"
      CKPT_ARGS+=(--load "${CKPT_SAVE_DIR}")
      case "$(printf '%s' "${BC_OVERRIDE_OPT_PARAM_SCHEDULER:-0}" | tr '[:upper:]' '[:lower:]')" in
         1|true)
            CKPT_ARGS+=(--override-opt-param-scheduler)
            echo "[head] resume will override checkpoint optimizer scheduler configuration"
            ;;
         ""|0|false) : ;;
         *)
            echo "ERROR: BC_OVERRIDE_OPT_PARAM_SCHEDULER must be one of: 1, true, 0, false" >&2
            exit 2
            ;;
      esac
   fi
fi

ROLLOUT_ARGS=(
   --prompt-data "${TRAIN_DATA}"
   --input-key prompt
   --label-key answer
   --metadata-key extra_info
   --rollout-shuffle
   --num-rollout           "${BC_NUM_ROLLOUT:-2}"
   --rollout-batch-size    "${BC_ROLLOUT_BATCH_SIZE:-8}"
   --n-samples-per-prompt  "${BC_N_SAMPLES:-4}"
   --rollout-max-response-len "${BC_MAX_RESPONSE_LEN:-8192}"
   --rollout-max-context-len  "${BC_MAX_CONTEXT_LEN:-16384}"
   --rollout-temperature 1.0
   --global-batch-size     "${BC_GLOBAL_BATCH_SIZE:-32}"
   --balance-data
)

case "$(printf '%s' "${BCPLUS_DYNAMIC_SAMPLING:-0}" | tr '[:upper:]' '[:lower:]')" in
   1|true)
      DYNAMIC_FIRST_POOL_SIZE=$((2 * ${BC_ROLLOUT_BATCH_SIZE:-8}))
      ROLLOUT_ARGS+=(
         --rollout-function-path examples.supo_browsecomp.dynamic_sampling.generate_rollout
         --over-sampling-batch-size "${DYNAMIC_FIRST_POOL_SIZE}"
      )
      echo "[head] adaptive dynamic sampling enabled: first_pool=${DYNAMIC_FIRST_POOL_SIZE}"
      ;;
   ""|0|false) : ;;
   *)
      echo "ERROR: BCPLUS_DYNAMIC_SAMPLING must be one of: 1, true, 0, false" >&2
      exit 2
      ;;
esac

PERF_ARGS=(
   --tensor-model-parallel-size "${TRAIN_TP}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${TRAIN_CP}"
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 2
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)
if [[ -n "${LOG_PROBS_CHUNK_SIZE}" ]]; then
   PERF_ARGS+=(--log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}")
fi

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl
   --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28
   --use-tis --tis-clip 2.0 --tis-clip-low 0.0
)

OPTIMIZER_ARGS=(
   --optimizer adam --lr 1e-6 --lr-decay-style constant
   --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98
)

SGLANG_ARGS=(
   # SGLang TP is independent of Megatron TP: 4B uses 2 for more engines,
   # while 9B uses 4 for model/KV-cache headroom.
   --rollout-num-gpus-per-engine "${SGLANG_TP}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
   --sglang-disable-custom-all-reduce
)

MISC_ARGS=(
   --attention-dropout 0.0 --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
   --attention-backend flash --log-multi-turn
)

CUSTOM_ARGS=(
   --custom-generate-function-path examples.supo_browsecomp.generate_with_bcplus.generate
   --custom-rm-path                 examples.supo_browsecomp.generate_with_bcplus.reward_func
   --reward-key score
   --custom-reward-post-process-path examples.supo_browsecomp.generate_with_bcplus.reward_post_process
   --custom-advantage-function-path examples.supo_browsecomp.summary_advantage.compute_summary_aware_advantages
   --custom-rollout-log-function-path examples.supo_browsecomp.generate_with_bcplus.log_bcplus
   --rollout-data-postprocess-path   examples.supo_browsecomp.generate_with_bcplus.dump_rollout_data_postprocess
)

# Offline rollout-state dump (per-iter parquet per DP rank → OILFS) for debugging.
# ON by default (BC_DUMP_ROLLOUT=1); BCPLUS_DUMP_DIR must reach the ray actors via
# RUNTIME_ENV_JSON (below). train_old log_probs are OPT-IN via BCPLUS_DUMP_TRAIN_OLD
# (same knob as run_qwen3p5_4B_colocate.sh): only when truthy do we add
# --dump-train-old-log-prob (an extra pre-train forward every iter); the env is
# passed through to the actors as-is (below) so the dump actually writes the
# column. Flag and env stay in lockstep — the flag alone pays the forward cost
# but the dump skips train_old (dump_train_old defaults off); the env alone trips
# the "train_old missing" assert. See generate_with_bcplus.py:128,1698.
if [[ "${BC_DUMP_ROLLOUT:-1}" == "1" ]]; then
   export BCPLUS_DUMP_DIR="${STAGE}/rollout_dumps/${RUN_NAME}"
   mkdir -p "${BCPLUS_DUMP_DIR}"
   case "$(printf '%s' "${BCPLUS_DUMP_TRAIN_OLD:-}" | tr '[:upper:]' '[:lower:]')" in
      ""|0|false) _train_old_label=off ;;
      *) _train_old_label=on; CUSTOM_ARGS+=(--dump-train-old-log-prob) ;;
   esac
   echo "[head] rollout dump ENABLED -> ${BCPLUS_DUMP_DIR} (train_old=${_train_old_label})"
else
   export BCPLUS_DUMP_DIR=""
   echo "[head] rollout dump disabled"
fi

WANDB_ARGS=(
   --use-wandb --wandb-mode "${BC_WANDB_MODE}"
   --wandb-explicit-teardown
   --wandb-project "${BC_WANDB_PROJECT:-supo-bcplus-mast}"
   --wandb-group "${RUN_NAME}" --wandb-dir "${WANDB_DIR}"
)
if [[ "${BC_WANDB_MODE}" == "online" ]]; then
   WANDB_ARGS+=(
      --wandb-host "${WANDB_BASE_URL}"
      --wandb-team "${BC_WANDB_ENTITY:-hhzhang01}"
      --wandb-key-file "${WANDB_LOCAL_KEY_FILE}"
      --wandb-online-fallback-offline
   )
fi

COLOCATE_ARGS=( --colocate )

# ----- start ray head, publish IP, wait for workers -----
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 8 \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

if [[ "${NNODES}" != "1" ]]; then
  echo "${MASTER_ADDR}" > "${HEAD_IP_FILE}"
  echo "[head] wrote ${HEAD_IP_FILE}=${MASTER_ADDR}; waiting 40s for ${NNODES} workers"
  sleep 40; ray status || true
fi

# Propagate the NCCL OOB interface override to all ray actors (the frontend NIC
# name is identical across the homogeneous nodes, so one value is cluster-wide
# correct). Also blank NCCL_SOCKET_IPADDR_PREFIX so a leftover beth-oriented
# prefix hint can't steer OOB back onto the backend NIC.
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"NCCL_SOCKET_IFNAME\": \"${NCCL_OOB_IFNAME}\",
    \"NCCL_CLIENT_SOCKET_IFNAME\": \"${NCCL_OOB_IFNAME}\",
    \"NCCL_SOCKET_IPADDR_PREFIX\": \"\",
    \"GLOO_SOCKET_IFNAME\": \"${NCCL_OOB_IFNAME}\",
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SLIME}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"SGLANG_NUMA_BIND_V2\": \"0\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"\",
    \"TRITON_CACHE_DIR\": \"/tmp/triton_cache_slime\",
    \"HF_HUB_OFFLINE\": \"1\",
    \"TRANSFORMERS_OFFLINE\": \"1\",
    \"WANDB_X_FLUSH_INTERVAL_SECONDS\": \"${WANDB_X_FLUSH_INTERVAL_SECONDS}\",
    \"WANDB_MODE\": \"${BC_WANDB_MODE}\",
    \"WANDB_BASE_URL\": \"${WANDB_BASE_URL}\",
    \"WANDB_HTTPS_PROXY\": \"${WANDB_HTTPS_PROXY}\",
    \"LOCAL_SEARCH_URL\": \"${LOCAL_SEARCH_URL}\",
    \"LLAMA_API_KEY\": \"${LLAMA_API_KEY}\",
    \"http_proxy\": \"\",
    \"HTTP_PROXY\": \"\",
    \"https_proxy\": \"http://127.0.0.1:9080\",
    \"HTTPS_PROXY\": \"http://127.0.0.1:9080\",
    \"no_proxy\": \"127.0.0.1,localhost,::1\",
    \"NO_PROXY\": \"127.0.0.1,localhost,::1\",
    \"BCPLUS_MAX_TURNS\": \"${BCPLUS_MAX_TURNS:-64}\",
    \"BCPLUS_COMPRESS_THRESH\": \"${BCPLUS_COMPRESS_THRESH:-0.85}\",
    \"BCPLUS_MAX_SUB_TRAJS\": \"${BCPLUS_MAX_SUB_TRAJS:-5}\",
    \"BCPLUS_COMPRESS_PENALTY\": \"${BCPLUS_COMPRESS_PENALTY:-0.5}\",
    \"BCPLUS_FIXED_SEARCH_TOPK\": \"${BCPLUS_FIXED_SEARCH_TOPK}\",
    \"BCPLUS_DOC_WORDS_FULL\": \"${BCPLUS_DOC_WORDS_FULL}\",
    \"BCPLUS_DUMP_DIR\": \"${BCPLUS_DUMP_DIR:-}\",
    \"BCPLUS_DUMP_TRAIN_OLD\": \"${BCPLUS_DUMP_TRAIN_OLD:-}\",
    \"BCPLUS_JUDGE_MODEL\": \"${BCPLUS_JUDGE_MODEL:-gpt-5-4-genai-dss4}\",
    \"BCPLUS_JUDGE_BASE_URL\": \"${BCPLUS_JUDGE_BASE_URL:-https://api.llama.com/compat/v1/}\",
    \"BCPLUS_JUDGE_CONCURRENCY\": \"${BCPLUS_JUDGE_CONCURRENCY:-64}\",
    \"BCPLUS_SEARCH_CONCURRENCY\": \"${BCPLUS_SEARCH_CONCURRENCY:-128}\",
    \"BCPLUS_DYNAMIC_SAMPLING\": \"${BCPLUS_DYNAMIC_SAMPLING:-0}\"
  }
}"

set -x
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes "${NNODES}" \
   --actor-num-gpus-per-node 8 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}" \
   "${COLOCATE_ARGS[@]}"
RC=$?
set +x
echo "[head] ray job submit returned ${RC}"
touch "${DONE_FILE}" 2>/dev/null || true
sleep 20
ray stop --force 2>/dev/null || true
echo "SUPO_RUN_DONE exit=${RC}"
exit ${RC}
