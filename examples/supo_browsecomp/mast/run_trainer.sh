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
#   * wandb OFFLINE → node-local disk → atomic OILFS snapshots for live sync
#   * Ray head/worker election from MAST_HPC_TASK_GROUP_HOSTNAMES (multi-node)
#
# Defaults are a 1-node (8 GPU) SMOKE: TP=4, CP=2 → DP=1, tiny batch, 2 rollouts.
# Scale via the BC_* / NUM env overrides once the pipeline is green.
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
RUN_NAME="${RUN_NAME:-${MAST_HPC_JOB_NAME:-supo-bcplus-mast-local}}"
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

pkill -9 sglang 2>/dev/null || true; sleep 2
ray stop --force 2>/dev/null || true; pkill -9 python 2>/dev/null || true; sleep 2

COORD_DIR="${STAGE}/ray-coord/${MAST_HPC_JOB_NAME:-supo-local}"
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
# close(2) with EOVERFLOW and leave an empty run). Every Ray actor writes to the
# same path on its own host-local /tmp. Each task publishes immutable tar
# snapshots into a task-specific OILFS directory; the devserver watcher extracts
# those snapshots to local disk before running `wandb sync`.
WANDB_ATTEMPT_ID="${MAST_HPC_JOB_ATTEMPT_INDEX:-0}-${MAST_HPC_TASK_GROUP_ATTEMPT_EPOCH:-0}"
WANDB_DIR="${MAST_WANDB_LOCAL_DIR:-/tmp/slime-wandb/${RUN_NAME}/attempt-${WANDB_ATTEMPT_ID}}"
WANDB_PUBLISHER_DIR="${STAGE}/wandb-snapshots/${RUN_NAME}/attempt-${WANDB_ATTEMPT_ID}-task-${TW_TASK_ID:-0}"
WANDB_SNAPSHOT_SCRIPT="${SLIME}/examples/supo_browsecomp/mast/wandb_snapshot.sh"
rm -rf "${WANDB_DIR}"
mkdir -p "${WANDB_DIR}"
bash "${WANDB_SNAPSHOT_SCRIPT}" watch "${WANDB_DIR}" "${WANDB_PUBLISHER_DIR}" \
  "${MAST_WANDB_SNAPSHOT_INTERVAL_SEC:-60}" &
WANDB_SNAPSHOT_PID=$!
echo "[trainer] W&B local=${WANDB_DIR} snapshots=${WANDB_PUBLISHER_DIR}"

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
# Robust against COORD_DIR staleness across retries (same MAST_HPC_JOB_NAME reuses
# COORD_DIR): the head clears COORD_DIR + rewrites head.ip fresh each attempt, and
# this worker RE-READS head.ip and RETRIES the join if it fails (e.g. it read a
# stale head.ip pointing at a dead head, or the head's ray isn't up yet).
if [[ "${IS_HEAD}" = "0" ]]; then
  joined=0
  for attempt in $(seq 1 120); do          # up to ~20 min of pull-skew + retries
    if [[ ! -f "${HEAD_IP_FILE}" ]]; then sleep 10; continue; fi
    HEAD_IP="$(tr -d ' \t\r\n' < "${HEAD_IP_FILE}")"
    [[ -z "${HEAD_IP}" ]] && { sleep 10; continue; }
    echo "[worker ${MYHOST}] attempt ${attempt}: joining ray at [${HEAD_IP}]:6379"
    ray stop --force 2>/dev/null || true
    if ray start --address="[${HEAD_IP}]:6379" --num-gpus 8 \
         --node-ip-address "${MASTER_ADDR}" --disable-usage-stats; then
      joined=1; break
    fi
    echo "[worker ${MYHOST}] join failed (stale head.ip or head not up yet); retrying"
    sleep 10
  done
  [[ "${joined}" = "1" ]] || { echo "ERROR: worker never joined ray" >&2; exit 1; }
  echo "[worker ${MYHOST}] joined ray; waiting for DONE"
  while [[ ! -f "${DONE_FILE}" ]]; do sleep 30; done
  echo "[worker ${MYHOST}] saw DONE, exiting"; ray stop --force || true; exit 0
fi

# --------------------------- head branch: config + submit -------------------
# Clear any stale coord state from a previous attempt (retries reuse COORD_DIR)
# BEFORE starting ray/writing a fresh head.ip, so workers never latch onto a dead
# head or an old DONE. Only the head touches COORD_DIR after this point.
rm -rf "${COORD_DIR}" 2>/dev/null || true
mkdir -p "${COORD_DIR}"
source scripts/models/qwen3.5-4B.sh

CKPT_SAVE_DIR="${STAGE}/checkpoints/${RUN_NAME}"

CKPT_ARGS=(
   --hf-checkpoint "${D}/Qwen3.5-4B"
   --ref-load      "${D}/Qwen3.5-4B_torch_dist"
)
if [[ "${BC_SAVE_INTERVAL:-5}" == "0" ]]; then
   echo "[head] checkpoint saving disabled (BC_SAVE_INTERVAL=0)"
else
   mkdir -p "${CKPT_SAVE_DIR}"
   CKPT_ARGS+=(--save "${CKPT_SAVE_DIR}" --save-interval "${BC_SAVE_INTERVAL:-5}")
   if [[ -f "${CKPT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
      echo "[head] resuming from ${CKPT_SAVE_DIR}"
      CKPT_ARGS+=(--load "${CKPT_SAVE_DIR}")
   fi
fi

ROLLOUT_ARGS=(
   --prompt-data "${D}/BC+/bc_train.parquet"
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

PERF_ARGS=(
   --tensor-model-parallel-size "${BC_TP:-4}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${BC_CP:-2}"
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 2
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${BC_MAX_TOKENS_PER_GPU:-49152}"
)

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
   # sglang engine TP is INDEPENDENT of megatron TP (canonical: megatron TP=4 but
   # sglang engine TP=2). Do NOT tie it to BC_TP.
   --rollout-num-gpus-per-engine "${BC_SGLANG_TP:-2}"
   --sglang-mem-fraction-static 0.7
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
   --use-wandb --wandb-mode offline
   --wandb-explicit-teardown
   --wandb-project "${BC_WANDB_PROJECT:-supo-bcplus-mast}"
   --wandb-group "${RUN_NAME}" --wandb-dir "${WANDB_DIR}"
)

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
    \"BCPLUS_DUMP_DIR\": \"${BCPLUS_DUMP_DIR:-}\",
    \"BCPLUS_DUMP_TRAIN_OLD\": \"${BCPLUS_DUMP_TRAIN_OLD:-}\",
    \"BCPLUS_JUDGE_MODEL\": \"${BCPLUS_JUDGE_MODEL:-gpt-5-4-genai-dss4}\",
    \"BCPLUS_JUDGE_BASE_URL\": \"${BCPLUS_JUDGE_BASE_URL:-https://api.llama.com/compat/v1/}\",
    \"BCPLUS_JUDGE_CONCURRENCY\": \"${BCPLUS_JUDGE_CONCURRENCY:-64}\",
    \"BCPLUS_SEARCH_CONCURRENCY\": \"${BCPLUS_SEARCH_CONCURRENCY:-128}\"
  }
}"

set -x
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes "${NNODES}" \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${COLOCATE_ARGS[@]}
RC=$?
set +x
echo "[head] ray job submit returned ${RC}"
touch "${DONE_FILE}" 2>/dev/null || true
sleep 20
ray stop --force 2>/dev/null || true
echo "SUPO_RUN_DONE exit=${RC}"
exit ${RC}
