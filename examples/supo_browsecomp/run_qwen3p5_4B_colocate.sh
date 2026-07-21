#!/bin/bash
# BrowseComp-Plus RL — Qwen3.5-4B on 8 A100 nodes (colocate mode).
#
# Fork of run_qwen3p5_9B_colocate.sh with model swapped to 4B. Same colocate
# machinery (per-node /dev/shm rootfs staging, COORD_DIR host->container path
# split, EXIT trap, --sglang-disable-custom-all-reduce).
#
# Why 4B here: Qwen3.5-9B triggers a megatron _apply_output_gate shape bug
# at TP=8 (num_query_groups=4 → KV replication path that the gate code
# doesn't handle). TP=4 on 9B OOMs during actor backward on vocab head
# activation at 65k context. 4B has hidden=2560 (vs 4096) so activations are
# smaller and TP=4 fits comfortably — no gate bug (TP=4 == num_query_groups=4
# means no KV replication), no OOM.
#
# Physical layout at 8 nodes: TP=4 × CP=1 × PP=1 = 4 GPU per model group,
# 64 GPU / 4 = DP=16. Global batch 256 / DP=16 = 16 samples per DP per iter.
#
# Two-part launcher:
#   * Outer part (login pod): auto-discovers search server, exports RUN_NAME,
#     submits ONE srun spanning 8 nodes, each running enroot + this same script.
#   * Inner part (in-container, per-node): head node starts Ray head + submits
#     training job; worker nodes join Ray cluster and wait for the run to end.
#
# Before running:
#   1. Search server running (see launch_search_server.sh).
#   2. LLAMA_API_KEY set on the login pod (judge routes via Llama API).
#   3. Qwen3.5-4B HF + torch_dist checkpoints on FSx.
#
# Debug-only overrides (env vars — leave unset for canonical config):
#   BC_NUM_ROLLOUT              default 20   (2 EPOCHS ≈ 20 iter × 32 prompts)
#   BC_ROLLOUT_BATCH_SIZE       default 32   (prompts per iter)
#   BC_N_SAMPLES                default 8    (rollouts per prompt)
#   BC_GLOBAL_BATCH_SIZE        default 256  (= batch × samples, 1 grad step per iter)
#   BC_MAX_RESPONSE_LEN         default 32768 (sglang per-call max_new_tokens)
#   BC_MAX_CONTEXT_LEN          default 65536 (per-sample total context budget)
#   BCPLUS_MAX_TURNS            default 64
#   BCPLUS_COMPRESS_THRESH      default 0.85
#   BCPLUS_MAX_SUB_TRAJS        default 5
#   BCPLUS_COMPRESS_PENALTY     default 0.5
#   BCPLUS_DUMP_DIR             default "" (empty = disabled)

set -euo pipefail

# ---------------------------------------------------------------------------
# Outer part: login pod. Submit ONE srun that spans 8 nodes.
# ---------------------------------------------------------------------------
if [[ "${SLIME_INNER:-0}" != "1" ]]; then
    : "${LLAMA_API_KEY:?LLAMA_API_KEY must be set on the login pod (LLM|... key with entitlement)}"

    # Auto-discover the search server if LOCAL_SEARCH_URL wasn't passed in.
    if [[ -z "${LOCAL_SEARCH_URL:-}" ]]; then
        HOST_FILE=/genai/fsx-project/hhzhang01/logs/search-server.hostname
        if [[ -f "${HOST_FILE}" ]]; then
            SEARCH_TARGET=$(cat "${HOST_FILE}")
            if curl -sf --max-time 5 "http://${SEARCH_TARGET}/health" > /dev/null; then
                export LOCAL_SEARCH_URL="http://${SEARCH_TARGET}"
                echo "auto-discovered LOCAL_SEARCH_URL=${LOCAL_SEARCH_URL}"
            else
                echo "ERROR: search server at ${SEARCH_TARGET} not responding." >&2
                echo "       Run examples/supo_browsecomp/launch_search_server.sh first." >&2
                exit 1
            fi
        else
            echo "ERROR: LOCAL_SEARCH_URL not set and ${HOST_FILE} missing." >&2
            echo "       Run examples/supo_browsecomp/launch_search_server.sh first." >&2
            exit 1
        fi
    fi

    export RUN_NAME="${RUN_NAME:-supo-bcplus-qwen3p5-4b-$(date +%Y%m%d-%H%M)}"
    echo "RUN_NAME=${RUN_NAME}"

    SLIME_HOST_DIR=/home/hhzhang01/slime
    ENROOT_ROOTFS="${ENROOT_ROOTFS:-slime-test}"
    SLURM_ACCOUNT="${SLURM_ACCOUNT:-genai_interns}"
    QOS="${QOS:-a100_genai_shared}"
    TRAIN_WALLTIME="${TRAIN_WALLTIME:-24:00:00}"
    NUM_NODES="${NUM_NODES:-8}"
    TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-/genai/fsx-project/hhzhang01/logs/${RUN_NAME}.log}"
    mkdir -p "$(dirname "${TRAIN_LOG_PATH}")"

    # Coordination file on FSx: head node writes its IP here; workers poll for it.
    # DONE file (written when head's ray job returns) tells workers they can exit.
    # mkdir on host path; the container sees this same dir via the
    # --mount /genai/fsx-project/hhzhang01:/genai_hh remap, so we pass the
    # container-visible path as COORD_DIR env for scripts inside enroot.
    COORD_DIR_HOST=/genai/fsx-project/hhzhang01/logs/ray-coord/${RUN_NAME}
    COORD_DIR=/genai_hh/logs/ray-coord/${RUN_NAME}
    mkdir -p "${COORD_DIR_HOST}"
    # Clean stale coord files from a prior run of the same RUN_NAME. Without
    # this, resume submits will hang forever: the prior run's EXIT trap
    # (`touch DONE`) leaves `done` around, and workers on this new job's poll
    # loop see DONE at startup and immediately exit — only the head node
    # joins Ray, placement group waits forever for the missing 7 nodes.
    # Observed on 293413 (6 h wasted before we noticed).
    rm -f "${COORD_DIR_HOST}/done" "${COORD_DIR_HOST}/head.ip"
    echo "coord dir host: ${COORD_DIR_HOST}"
    echo "coord dir container: ${COORD_DIR}"

    # One srun spanning all N nodes. `--ntasks-per-node=1` → one enroot per node.
    # `--exclusive` reserves the full node so nothing else lands on our GPUs.
    #
    # We background srun and drive a wandb-sync poll loop from this login-pod
    # shell while training runs — mirrors the launch_all.sh orchestration
    # pattern used by the 4B external-sglang canonical. This lets you watch
    # loss curves on wandb.ai mid-run without waiting for the whole job to
    # finish. Sync interval defaults to 5 min (override via WANDB_SYNC_INTERVAL_SEC).
    WANDB_SYNC_INTERVAL_SEC="${WANDB_SYNC_INTERVAL_SEC:-300}"
    SYNC_SCRIPT=/home/hhzhang01/slime/aws-cluster/wandb-sync.sh

    srun \
        --nodes=${NUM_NODES} --gpus-per-node=8 --ntasks-per-node=1 --exclusive \
        --cpus-per-task=64 --mem=0 \
        --account="${SLURM_ACCOUNT}" --qos="${QOS}" \
        --time="${TRAIN_WALLTIME}" \
        --mpi=none \
        --job-name="${RUN_NAME}" \
        --output="${TRAIN_LOG_PATH}" \
        bash -c "
            # Pre-stage rootfs to per-node local /dev/shm to avoid flock()
            # failures on shared FSx (NFS4). enroot's runtime.sh:243 does
            # 'flock -w 30' on \${rootfs}/.enroot.lock; on NFS4 flock is
            # unreliable and with 8 nodes racing, 6/8 immediately fail with
            # 'Could not acquire rootfs lock'. Local tmpfs sidesteps this
            # entirely. cp -a is ~30-60s from FSx to tmpfs; each subsequent
            # enroot start reuses the local copy.
            LOCAL_ENROOT_DATA=/dev/shm/enroot-\${USER}-\${SLURM_JOB_ID}
            LOCAL_ROOTFS=\${LOCAL_ENROOT_DATA}/${ENROOT_ROOTFS}
            if [[ ! -d \${LOCAL_ROOTFS} ]]; then
                mkdir -p \${LOCAL_ENROOT_DATA}
                echo \"[node \${SLURM_NODEID:-0}] copying rootfs FSx -> \${LOCAL_ROOTFS} ...\"
                time cp -a /storage/home/hhzhang01/.local/share/enroot/${ENROOT_ROOTFS} \${LOCAL_ENROOT_DATA}/
                echo \"[node \${SLURM_NODEID:-0}] rootfs staged\"
            fi

            ENROOT_TEMP_PATH=/dev/shm \
            ENROOT_DATA_PATH=\${LOCAL_ENROOT_DATA} \
            ENROOT_MOUNT_HOME=false \
            enroot start \
                --mount ${SLIME_HOST_DIR}:/slime \
                --mount ${SLIME_HOST_DIR}/aws-cluster:/aws-cluster \
                --mount /genai/fsx-project/hhzhang01:/genai_hh \
                --mount /genai/fsx-project/hhzhang01/wandb:/data/wandb \
                --env RUN_NAME='${RUN_NAME}' \
                --env SLIME_INNER=1 \
                --env NUM_NODES='${NUM_NODES}' \
                --env COORD_DIR='${COORD_DIR}' \
                --env LOCAL_SEARCH_URL='${LOCAL_SEARCH_URL}' \
                --env LLAMA_API_KEY='${LLAMA_API_KEY}' \
                --env SLURM_NODEID=\${SLURM_NODEID:-0} \
                --env SLURM_JOB_NODELIST=\${SLURM_JOB_NODELIST} \
                --env SLURM_JOB_ID=\${SLURM_JOB_ID} \
                --env BCPLUS_COMPRESS_THRESH='${BCPLUS_COMPRESS_THRESH:-}' \
                --env BCPLUS_MAX_SUB_TRAJS='${BCPLUS_MAX_SUB_TRAJS:-}' \
                --env BCPLUS_MAX_TURNS='${BCPLUS_MAX_TURNS:-}' \
                --env BCPLUS_COMPRESS_PENALTY='${BCPLUS_COMPRESS_PENALTY:-}' \
                --env BCPLUS_DUMP_DIR='${BCPLUS_DUMP_DIR:-}' \
                --env BC_NUM_ROLLOUT='${BC_NUM_ROLLOUT:-}' \
                --env BC_ROLLOUT_BATCH_SIZE='${BC_ROLLOUT_BATCH_SIZE:-}' \
                --env BC_N_SAMPLES='${BC_N_SAMPLES:-}' \
                --env BC_GLOBAL_BATCH_SIZE='${BC_GLOBAL_BATCH_SIZE:-}' \
                --env BC_MAX_RESPONSE_LEN='${BC_MAX_RESPONSE_LEN:-}' \
                --env BC_MAX_CONTEXT_LEN='${BC_MAX_CONTEXT_LEN:-}' \
                --env WANDB_X_FLUSH_INTERVAL_SECONDS='${WANDB_X_FLUSH_INTERVAL_SECONDS:-30}' \
                ${ENROOT_ROOTFS} \
                bash /slime/examples/supo_browsecomp/run_qwen3p5_4B_colocate.sh
        " &
    TRAIN_PID=$!
    echo "[outer] srun backgrounded, pid=${TRAIN_PID}; wandb-sync every ${WANDB_SYNC_INTERVAL_SEC}s"

    # Ctrl-C in this shell -> scancel the slurm job so we don't leak.
    trap 'echo "[outer] SIGINT: scancel ${RUN_NAME}"; scancel --jobname="${RUN_NAME}" 2>/dev/null; exit 130' INT TERM

    # Poll wandb-sync while training alive. sync failures are non-fatal
    # (wandb service on login pod occasionally has hiccups; keep going).
    while kill -0 "${TRAIN_PID}" 2>/dev/null; do
        sleep "${WANDB_SYNC_INTERVAL_SEC}"
        if ! bash "${SYNC_SCRIPT}" "${RUN_NAME}" 2>&1 | sed 's/^/[wandb-sync] /'; then
            echo "[outer] mid-run wandb-sync returned non-zero; continuing"
        fi
    done

    wait "${TRAIN_PID}"
    TRAIN_STATUS=$?
    echo "[outer] training finished with status ${TRAIN_STATUS}"

    # Final sync — catch anything the last periodic sync missed.
    echo "[outer] final wandb-sync for ${RUN_NAME}"
    bash "${SYNC_SCRIPT}" "${RUN_NAME}" 2>&1 | sed 's/^/[wandb-sync] /' || true

    exit "${TRAIN_STATUS}"
fi

# ---------------------------------------------------------------------------
# Inner part: in-container, ONE per node. Head vs worker branch.
# ---------------------------------------------------------------------------
: "${RUN_NAME:?RUN_NAME must be set (populated by outer part)}"
: "${LOCAL_SEARCH_URL:?LOCAL_SEARCH_URL must be forwarded into the container}"
: "${LLAMA_API_KEY:?LLAMA_API_KEY must be forwarded into the container}"
: "${COORD_DIR:?COORD_DIR must be forwarded into the container}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID must be forwarded into the container}"

HEAD_IP_FILE="${COORD_DIR}/head.ip"
DONE_FILE="${COORD_DIR}/done"
NODEID="${SLURM_NODEID:-0}"
NUM_NODES="${NUM_NODES:-8}"
# `hostname -i` returns 127.0.0.1 when the hostname resolves to loopback in
# /etc/hosts. `hostname -I` returns all non-loopback addresses; the first is
# the ethernet interface ray needs for cross-node control-plane traffic.
# NCCL handles the EFA/IB data plane separately.
MY_IP=$(hostname -I | awk '{print $1}')

pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3

set -x
export PYTHONUNBUFFERED=1

cd /slime

# ---------- Worker branch: join ray cluster, wait for head to signal done ----
if [[ "${NODEID}" != "0" ]]; then
    echo "[worker node ${NODEID}] my_ip=${MY_IP}, waiting for head.ip"
    for i in $(seq 1 60); do
        [[ -f "${HEAD_IP_FILE}" ]] && break
        sleep 5
    done
    if [[ ! -f "${HEAD_IP_FILE}" ]]; then
        echo "[worker node ${NODEID}] head.ip never appeared after 5min, giving up" >&2
        exit 1
    fi
    HEAD_IP=$(cat "${HEAD_IP_FILE}")
    echo "[worker node ${NODEID}] connecting to head at ${HEAD_IP}:6379"
    ray start --address="${HEAD_IP}:6379" --num-gpus 8 \
        --node-ip-address "${MY_IP}" --disable-usage-stats

    # Wait for head to signal training is done, then exit cleanly. srun's
    # --exclusive keeps the allocation up until all tasks return; if we sleep
    # forever here, the head's exit doesn't tear us down.
    echo "[worker node ${NODEID}] joined ray, waiting for DONE"
    while [[ ! -f "${DONE_FILE}" ]]; do
        sleep 30
    done
    echo "[worker node ${NODEID}] saw DONE, exiting"
    ray stop --force || true
    exit 0
fi

# ---------- Head branch: start ray head, launch training, signal DONE --------
echo "[head node] my_ip=${MY_IP}, num_nodes=${NUM_NODES}"
source /aws-cluster/wandb-args.sh
source scripts/models/qwen3.5-4B.sh

HF_CKPT_HOST=/genai_hh/models/Qwen3.5-4B
REF_LOAD_HOST=/genai_hh/models/Qwen3.5-4B_torch_dist
TRAIN_DATA=/genai_hh/datasets/BC+/bc_train.parquet
TEST_DATA=/genai_hh/datasets/BC+/bc_test.parquet
CKPT_SAVE_DIR=/genai_hh/checkpoints/${RUN_NAME}
mkdir -p "${CKPT_SAVE_DIR}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT_HOST}"
   --ref-load "${REF_LOAD_HOST}"
   # Save mcore torch_dist checkpoints every 5 iters (see convert_torch_dist_to_hf
   # to convert individual checkpoints for offline eval).
   --save "${CKPT_SAVE_DIR}"
   --save-interval 5
)

# Conditional --load for resume. First run: no latest_checkpointed_iteration.txt
# exists, so we cold-init from --hf-checkpoint. Resume: file exists, add --load
# so megatron picks up from the last saved iter AND slime's data_source.load()
# restores sample_offset + epoch_id (data order stays identical because
# rollout_shuffle is deterministic on `seed + epoch_id`, seed=42 default).
#
# To resume a specific run:
#   export RUN_NAME=<the-original-run-name>   # same name as first submission
#   bash run_qwen3p5_4B_colocate.sh
# Without the RUN_NAME export the outer part generates a fresh timestamp,
# CKPT_SAVE_DIR points to a new empty dir, and the branch below no-ops.
if [[ -f "${CKPT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
    LOADED_ITER=$(cat "${CKPT_SAVE_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null || echo "?")
    echo "[head] resuming from ${CKPT_SAVE_DIR} (last iter: ${LOADED_ITER})"
    CKPT_ARGS+=(--load "${CKPT_SAVE_DIR}")
else
    echo "[head] first run — cold init from ${HF_CKPT_HOST}"
fi

ROLLOUT_ARGS=(
   --prompt-data "${TRAIN_DATA}"
   --input-key prompt
   --label-key answer
   --metadata-key extra_info
   # NOTE: no --apply-chat-template flag. Our generate() function calls
   # apply_chat_template(tools=TOOLS) itself so Qwen3.5's <tools> schema block
   # + <tool_call> format instructions get injected into the system message.
   --rollout-shuffle
   # Batch: 32 prompts × 8 samples = 256 rollouts per iter. global_batch_size
   # equals num_rollouts so we do exactly 1 gradient step per iter (SUPO-style
   # fully on-policy). 20 iter × 32 prompts = 640 prompts ≈ 1 epoch of the
   # 680-prompt training set.
   --num-rollout ${BC_NUM_ROLLOUT:-20}
   --rollout-batch-size ${BC_ROLLOUT_BATCH_SIZE:-32}
   --n-samples-per-prompt ${BC_N_SAMPLES:-8}
   # Per-sglang-call max_new_tokens (see notes/CONTEXT_LENGTH_LAYERS.md L4).
   --rollout-max-response-len ${BC_MAX_RESPONSE_LEN:-32768}
   # Per-sample total context budget (prompt + accumulated response). Drives
   # SUPO compression trigger: compress fires at BCPLUS_COMPRESS_THRESH ×
   # rollout-max-context-len = 0.85 × 64k ≈ 55.7k tokens.
   --rollout-max-context-len ${BC_MAX_CONTEXT_LEN:-65536}
   --rollout-temperature 1.0
   --global-batch-size ${BC_GLOBAL_BATCH_SIZE:-256}
   --balance-data
)

PERF_ARGS=(
   # 8 nodes × 8 GPU = 64 GPUs. TP=4 × CP=2 × PP=1 × DP=8 = 64.
   # TP=4 == num_query_groups=4: each rank gets exactly 1 KV head, no
   # replication. Avoids the megatron _apply_output_gate shape bug that
   # trips at TP > num_query_groups (which is what killed our TP=8 9B run).
   # CP=2 (was 1): the previous CP=1 config OOM'd on run 292745 at iter 10
   # (actor train's vocab_parallel_softmax needed 15 GB for a 55834-token
   # sample × vocab_size/TP=4 × fp32 logits; PyTorch had 25 GB reserved-
   # but-unallocated i.e. fragmented, couldn't find a contiguous 15 GB
   # block). --recompute-granularity full doesn't help this because it
   # only recomputes transformer layers, not the loss-side vocab head.
   # CP=2 shards along seq dim, halving per-rank loss compute to
   # 27917 tokens × vocab_size/TP=4 × 4 bytes ≈ 4.2 GB (fits comfortably).
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 2
   --use-dynamic-batch-size
   # max-tokens-per-gpu: 32k tokens per microbatch per DP rank (was 65k).
   # Combined with CP=2 halving per-rank seq, one microbatch's peak
   # vocab-head alloc is ~4 GB, well within 80 GB HBM after model shard,
   # activation, optimizer state, and torch_memory_saver overhead.
   --max-tokens-per-gpu 32768
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   # Truncated Importance Sampling. See run_qwen3p5_4B.sh for rationale.
   --use-tis
   --tis-clip 2.0
   --tis-clip-low 0.0
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   # Colocate: sglang engines run on the same 64 GPUs as the training actor.
   # Slime offloads actor weights to CPU during rollout and re-onloads before
   # training. Engine TP=2 → 64/2 = 32 sglang engines running concurrently.
   # 4B model at 64k context: KV cache per max-length request ~4.7 GB, so per
   # engine (2 GPU, ~126 GB free HBM) can hold ~50 concurrent max-len requests
   # — vast headroom vs BC+ per-engine load of ~8 concurrent. Prior TP=4 was
   # over-sharded: same aggregate bandwidth but half the engine count, so
   # per-engine concurrency capacity was under-utilized and rollout was the
   # main bottleneck (wait_time_ratio ~48%). Actor keeps TP=4 (Megatron topo
   # unchanged); slime's colocate weight update all-gathers actor's TP=4
   # shards to full HF tensor, then IPC-distributes to sglang's TP=2 engines
   # — no manual reshape needed. Precedent: examples/retool/retool_qwen3_4b_rl.sh
   # (same model size, uses TP=2), examples/on_policy_distillation/run-qwen3-8B-opd.sh
   # (actor_tp=2 vs sglang_tp=1).
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   # Disable custom all-reduce. In colocate mode, torch_memory_saver's CUDA
   # VMM allocations are incompatible with sglang's custom_all_reduce.cuh
   # (which relies on cudaIpcGetMemHandle for cross-rank shared memory). NCCL
   # fallback is slightly slower on small reductions but works. This applies
   # at any TP > 1.
   --sglang-disable-custom-all-reduce
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   # Safe with SUPO compression (sub-traj list) — per-sample aggregation only.
   --log-multi-turn
)

CUSTOM_ARGS=(
   --custom-generate-function-path examples.supo_browsecomp.generate_with_bcplus.generate
   --custom-rm-path                 examples.supo_browsecomp.generate_with_bcplus.reward_func
   --reward-key score
   --custom-reward-post-process-path examples.supo_browsecomp.generate_with_bcplus.reward_post_process
   --custom-rollout-log-function-path examples.supo_browsecomp.generate_with_bcplus.log_bcplus
   --rollout-data-postprocess-path   examples.supo_browsecomp.generate_with_bcplus.dump_rollout_data_postprocess
)

# When dumping is enabled, force slime to run the pre-training forward pass
# that populates rollout_data["log_probs"] (train_old). See run_qwen3p5_4B.sh
# comments for why can_reuse_log_probs_in_loss otherwise skips it.
if [[ -n "${BCPLUS_DUMP_DIR:-}" ]]; then
    CUSTOM_ARGS+=(--dump-train-old-log-prob)
    if [[ "${BCPLUS_DUMP_DIR}" == /genai/fsx-project/hhzhang01/* ]]; then
        BCPLUS_DUMP_DIR_CONTAINER="${BCPLUS_DUMP_DIR/#\/genai\/fsx-project\/hhzhang01/\/genai_hh}"
        echo "[BCPLUS] auto-translated BCPLUS_DUMP_DIR host=${BCPLUS_DUMP_DIR} -> container=${BCPLUS_DUMP_DIR_CONTAINER}"
        export BCPLUS_DUMP_DIR="${BCPLUS_DUMP_DIR_CONTAINER}"
    fi
fi

# Colocate + offload: sglang engines and training actor share the same 64
# GPUs. --colocate implies --offload-train and --offload-rollout.
COLOCATE_ARGS=(
   --colocate
)

# ---- Start ray head, publish IP, wait for workers ----
export MASTER_ADDR="${MY_IP}"

# EXIT trap: no matter how head exits (training crash triggering set -e, kill
# signal, normal completion), always touch DONE so workers can leave their
# poll loop. Without this, `ray job submit` failure trips set -e and workers
# hang until srun walltime.
trap 'echo "[head] EXIT trap: touching ${DONE_FILE}"; touch "${DONE_FILE}" 2>/dev/null || true' EXIT

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 8 \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Publish head IP so workers can join.
echo "${MY_IP}" > "${HEAD_IP_FILE}"
echo "[head] wrote ${HEAD_IP_FILE}=${MY_IP}, waiting 30s for workers to join"
sleep 30
ray status || true

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"LOCAL_SEARCH_URL\": \"${LOCAL_SEARCH_URL}\",
    \"LLAMA_API_KEY\": \"${LLAMA_API_KEY}\",
    \"BCPLUS_MAX_TURNS\": \"${BCPLUS_MAX_TURNS:-64}\",
    \"BCPLUS_COMPRESS_THRESH\": \"${BCPLUS_COMPRESS_THRESH:-0.85}\",
    \"BCPLUS_MAX_SUB_TRAJS\": \"${BCPLUS_MAX_SUB_TRAJS:-5}\",
    \"BCPLUS_COMPRESS_PENALTY\": \"${BCPLUS_COMPRESS_PENALTY:-0.5}\",
    \"BCPLUS_DUMP_DIR\": \"${BCPLUS_DUMP_DIR:-}\",
    \"WANDB_X_FLUSH_INTERVAL_SECONDS\": \"${WANDB_X_FLUSH_INTERVAL_SECONDS:-30}\"
  }
}"
# NOTE: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is DELIBERATELY NOT
# set. torch_memory_saver (used by --colocate for offload/onload) refuses to
# run with expandable_segments, throwing:
#   RuntimeError: TorchMemorySaver is disabled for the current process
#   because expandable_segments is not supported yet.
# The 4B canonical script sets expandable_segments because it uses external
# sglang and does not activate torch_memory_saver. In colocate mode the two
# are mutually exclusive.

# Submit training job. Blocks until run finishes (num_rollout iters done).
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes ${NUM_NODES} \
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

TRAIN_STATUS=$?
echo "[head] ray job submit returned status=${TRAIN_STATUS}"

# Disable errexit for the tail cleanup. Long-running training on FSx-mounted
# /slime can leave the NFS session stale; when bash tries to read the next
# script line after ray job submit returns, it can fail with:
#   /slime/examples/.../run_qwen3p5_4B_colocate.sh: error reading input file:
#     Stale file handle
# Under `set -e` that immediately trips exit 2 even though training itself
# succeeded (ray job returned status=0). Downgrading errexit here lets the
# cleanup lines and final exit fall through so srun reflects TRAIN_STATUS.
set +e

# Signal workers to exit so srun can complete.
touch "${DONE_FILE}" 2>/dev/null || true
echo "[head] wrote DONE, waiting for workers to exit"
sleep 30 || true
ray stop --force 2>/dev/null || true

exit ${TRAIN_STATUS}
