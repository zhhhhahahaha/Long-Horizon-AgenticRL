#!/bin/bash
# BrowseComp-Plus RL — pipeline smoke test with Qwen3.5-4B on 8 GPUs.
#
# Two-part launcher:
#   * Outer part (login pod): allocates the Slurm node, exports RUN_NAME,
#     and re-execs this same script inside the enroot container.
#   * Inner part (in-container): starts Ray, submits the training job.
#
# Before running:
#   1. Search server running (see launch_search_server.sh). If
#      LOCAL_SEARCH_URL is not set, this script auto-reads the hostname
#      written by that script at $HOST_FILE.
#   2. LLAMA_API_KEY must be set on the login pod (judge routes through
#      MetaGen via the Llama API OpenAI-compat endpoint; see README).
#   3. HF checkpoints for Qwen3.5-4B (both hf and torch_dist mcore forms)
#      exist at HF_CKPT_HOST / REF_LOAD_HOST below.

set -euo pipefail

# ---------------------------------------------------------------------------
# Outer part: login pod
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

    export RUN_NAME="${RUN_NAME:-supo-bcplus-qwen3p5-4b-smoke-$(date +%Y%m%d-%H%M)}"
    echo "RUN_NAME=${RUN_NAME}"

    SLIME_HOST_DIR=/home/hhzhang01/slime
    ENROOT_ROOTFS="${ENROOT_ROOTFS:-slime-test}"
    SLURM_ACCOUNT="${SLURM_ACCOUNT:-genai_interns}"
    QOS="${QOS:-a100_genai_shared}"
    TRAIN_WALLTIME="${TRAIN_WALLTIME:-24:00:00}"

    exec srun \
        --nodes=1 --gpus-per-node=8 --ntasks-per-node=1 --exclusive \
        --cpus-per-task=64 --mem=0 \
        --account="${SLURM_ACCOUNT}" --qos="${QOS}" \
        --time="${TRAIN_WALLTIME}" \
        --mpi=none \
        --job-name="${RUN_NAME}" \
        --output="/genai/fsx-project/hhzhang01/logs/${RUN_NAME}.log" \
        bash -c "
            ENROOT_TEMP_PATH=/dev/shm \
            ENROOT_DATA_PATH=/storage/home/hhzhang01/.local/share/enroot \
            ENROOT_MOUNT_HOME=false \
            enroot start \
                --mount ${SLIME_HOST_DIR}:/slime \
                --mount ${SLIME_HOST_DIR}/aws-cluster:/aws-cluster \
                --mount /genai/fsx-project/hhzhang01:/genai_hh \
                --mount /genai/fsx-project/hhzhang01/wandb:/data/wandb \
                --env RUN_NAME='${RUN_NAME}' \
                --env SLIME_INNER=1 \
                --env LOCAL_SEARCH_URL='${LOCAL_SEARCH_URL}' \
                --env LLAMA_API_KEY='${LLAMA_API_KEY}' \
                ${ENROOT_ROOTFS} \
                bash /slime/examples/supo_browsecomp/run_qwen3p5_4B.sh
        "
fi

# ---------------------------------------------------------------------------
# Inner part: in-container
# ---------------------------------------------------------------------------
: "${RUN_NAME:?RUN_NAME must be set (populated by outer part)}"
: "${LOCAL_SEARCH_URL:?LOCAL_SEARCH_URL must be forwarded into the container}"
: "${LLAMA_API_KEY:?LLAMA_API_KEY must be forwarded into the container}"

pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3

set -x
export PYTHONUNBUFFERED=1

cd /slime

source /aws-cluster/wandb-args.sh
source scripts/models/qwen3.5-4B.sh

HF_CKPT_HOST=/genai_hh/models/Qwen3.5-4B
REF_LOAD_HOST=/genai_hh/models/Qwen3.5-4B_torch_dist
TRAIN_DATA=/genai_hh/datasets/BC+/bc_train.parquet
TEST_DATA=/genai_hh/datasets/BC+/bc_test.parquet

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT_HOST}"
   --ref-load "${REF_LOAD_HOST}"
)

ROLLOUT_ARGS=(
   --prompt-data "${TRAIN_DATA}"
   --input-key prompt
   --label-key answer
   --metadata-key extra_info
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 20
   --rollout-batch-size 8
   --n-samples-per-prompt 4
   --rollout-max-response-len 32768
   --rollout-temperature 1.0
   --global-batch-size 32
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 2
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
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
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.85
   # Custom all-reduce fails on this cluster's A100 topology at tp=4
   # ("custom_all_reduce.cuh:37: CUDA error: invalid argument" during CUDA
   # graph capture). Fall back to NCCL all-reduce.
   --sglang-disable-custom-all-reduce
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path examples.supo_browsecomp.generate_with_bcplus.generate
   --custom-rm-path                 examples.supo_browsecomp.generate_with_bcplus.reward_func
   # reward_func returns a dict; `score` is the training signal, other keys
   # (n_turns, n_search, n_open, finished, truncated) are rollout diagnostics.
   --reward-key score
   # reward_post_process aggregates the diagnostic keys into bcplus/* wandb
   # metrics per rollout batch, then falls through to slime's default GRPO
   # group-normalized reward math.
   --custom-reward-post-process-path examples.supo_browsecomp.generate_with_bcplus.reward_post_process
)

MISC_ARGS+=(
   # log_multi_turn_data reads sample.metadata["round_number"] we set in generate()
   # and produces multi_turn_metric/round_number_* plus raw/observed response
   # length stats.
   --log-multi-turn
   # log pass@k across the group of n_samples_per_prompt trajectories per prompt
   # (with n=4 we get pass@1..pass@4 → useful to see if the model ever solves
   # a hard question even if only 1 of 4 samples works).
   --log-passrate
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"LOCAL_SEARCH_URL\": \"${LOCAL_SEARCH_URL}\",
    \"LLAMA_API_KEY\": \"${LLAMA_API_KEY}\",
    \"BCPLUS_MAX_TURNS\": \"5\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
