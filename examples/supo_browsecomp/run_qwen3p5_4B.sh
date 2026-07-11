#!/bin/bash
# BrowseComp-Plus RL — canonical training launcher for Qwen3.5-4B on 8 GPUs.
#
# This file is the make-sense config for real training runs. Every rollout /
# training knob defaulted here is what we want in production. Debug scripts
# live under debug_scripts/ and only override a small set of BC_* env vars
# (rollout size, context budget, threshold) — the pipeline logic itself
# (compression, tools schema, reward hooks) is shared with this file.
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
#
# Debug-only overrides (env vars — leave unset for canonical config):
#   BC_NUM_ROLLOUT              default 20
#   BC_ROLLOUT_BATCH_SIZE       default 8
#   BC_N_SAMPLES                default 4
#   BC_GLOBAL_BATCH_SIZE        default 32
#   BC_MAX_RESPONSE_LEN         default 32768 (sglang per-call max_new_tokens)
#   BC_MAX_CONTEXT_LEN          default 131072 (per-sample total context budget)
#   BCPLUS_MAX_TURNS            default 64 (SUPO react_agent max turn count)
#   BCPLUS_COMPRESS_THRESH      default 0.85 (compression trigger fraction)
#   BCPLUS_MAX_SUB_TRAJS        default 5 (SUPO max_session)

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

    # Auto-discover the sglang rollout server (external). launch_all.sh calls
    # launch_sglang_server.sh first, which writes hostname:port to HOST_FILE.
    if [[ -z "${SGLANG_SERVER_URL:-}" ]]; then
        HOST_FILE=/genai/fsx-project/hhzhang01/logs/sglang-server.hostname
        if [[ -f "${HOST_FILE}" ]]; then
            SGL_TARGET=$(cat "${HOST_FILE}")
            if curl -sf --max-time 5 "http://${SGL_TARGET}/health" > /dev/null; then
                export SGLANG_SERVER_URL="http://${SGL_TARGET}"
                echo "auto-discovered SGLANG_SERVER_URL=${SGLANG_SERVER_URL}"
            else
                echo "ERROR: sglang server at ${SGL_TARGET} not responding." >&2
                echo "       Run examples/supo_browsecomp/launch_sglang_server.sh first." >&2
                exit 1
            fi
        else
            echo "ERROR: SGLANG_SERVER_URL not set and ${HOST_FILE} missing." >&2
            echo "       Run examples/supo_browsecomp/launch_sglang_server.sh first." >&2
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
    # TRAIN_LOG_PATH lets launch_all.sh route this into the per-run log dir.
    # Fall back to the legacy top-level path when not set.
    TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-/genai/fsx-project/hhzhang01/logs/${RUN_NAME}.log}"
    mkdir -p "$(dirname "${TRAIN_LOG_PATH}")"

    exec srun \
        --nodes=1 --gpus-per-node=8 --ntasks-per-node=1 --exclusive \
        --cpus-per-task=64 --mem=0 \
        --account="${SLURM_ACCOUNT}" --qos="${QOS}" \
        --time="${TRAIN_WALLTIME}" \
        --mpi=none \
        --job-name="${RUN_NAME}" \
        --output="${TRAIN_LOG_PATH}" \
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
                --env SGLANG_SERVER_URL='${SGLANG_SERVER_URL}' \
                --env LLAMA_API_KEY='${LLAMA_API_KEY}' \
                --env BCPLUS_COMPRESS_THRESH='${BCPLUS_COMPRESS_THRESH:-}' \
                --env BCPLUS_MAX_SUB_TRAJS='${BCPLUS_MAX_SUB_TRAJS:-}' \
                --env BCPLUS_MAX_TURNS='${BCPLUS_MAX_TURNS:-}' \
                --env BC_NUM_ROLLOUT='${BC_NUM_ROLLOUT:-}' \
                --env BC_ROLLOUT_BATCH_SIZE='${BC_ROLLOUT_BATCH_SIZE:-}' \
                --env BC_N_SAMPLES='${BC_N_SAMPLES:-}' \
                --env BC_GLOBAL_BATCH_SIZE='${BC_GLOBAL_BATCH_SIZE:-}' \
                --env BC_MAX_RESPONSE_LEN='${BC_MAX_RESPONSE_LEN:-}' \
                --env BC_MAX_CONTEXT_LEN='${BC_MAX_CONTEXT_LEN:-}' \
                ${ENROOT_ROOTFS} \
                bash /slime/examples/supo_browsecomp/run_qwen3p5_4B.sh
        "
fi

# ---------------------------------------------------------------------------
# Inner part: in-container
# ---------------------------------------------------------------------------
: "${RUN_NAME:?RUN_NAME must be set (populated by outer part)}"
: "${LOCAL_SEARCH_URL:?LOCAL_SEARCH_URL must be forwarded into the container}"
: "${SGLANG_SERVER_URL:?SGLANG_SERVER_URL must be forwarded into the container}"
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
   # NOTE: no --apply-chat-template flag. Our generate() function calls
   # apply_chat_template(tools=TOOLS) itself so Qwen3.5's <tools> schema block
   # + <tool_call> format instructions get injected into the system message.
   # If slime pre-renders, the tools argument is dropped and the model won't
   # know the tool schemas.
   --rollout-shuffle
   # Rollout sizing. Defaults here are the make-sense config for a real
   # training run. The debug wrapper (debug_scripts/launch_all_debug.sh)
   # shrinks these via the BC_* env vars declared in the module header.
   --num-rollout ${BC_NUM_ROLLOUT:-20}
   --rollout-batch-size ${BC_ROLLOUT_BATCH_SIZE:-8}
   --n-samples-per-prompt ${BC_N_SAMPLES:-4}
   # Per-turn sglang max_new_tokens. See notes/CONTEXT_LENGTH_LAYERS.md.
   --rollout-max-response-len ${BC_MAX_RESPONSE_LEN:-32768}
   # Per-sample total context budget (prompt + accumulated response). Governs
   # when SUPO compression fires: trigger = BCPLUS_COMPRESS_THRESH *
   # rollout-max-context-len. 128k for real training so a sample can grow to
   # ~5 sub-trajectories worth of tokens before capping. Debug scripts shrink
   # this (via BC_MAX_CONTEXT_LEN) to force early compression triggers. Must
   # stay <= SGLang server's --context-length (set in launch_sglang_server.sh).
   --rollout-max-context-len ${BC_MAX_CONTEXT_LEN:-131072}
   --rollout-temperature 1.0
   --global-batch-size ${BC_GLOBAL_BATCH_SIZE:-32}
   --balance-data
)

PERF_ARGS=(
   # Actor is 8 GPUs (external SGLang lives on a separate slurm job/node).
   # TP=4 × CP=2 × PP=1 × DP=1 = 8. CP splits each sample along the seq
   # dimension across 2 ranks (ring attention keeps context correct), which
   # halves activation memory per rank — necessary because rollouts can
   # reach the total context budget set by --rollout-max-context-len (32k
   # during debug, targeting 128k for real runs).
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
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
   # Truncated Importance Sampling (Yao et al., https://fengyao.notion.site/off-policy-rl).
   # Corrects for train/rollout distribution mismatch when the Megatron actor
   # and the external SGLang sampler produce different token-level probabilities
   # for the same context. Per-token IS ratio exp(log π_train - log π_rollout)
   # is clamped to [tis-clip-low, tis-clip] and multiplied onto pg_loss.
   # Both clip values below are the argparse defaults — pinned here so the
   # training config is self-documenting.
   # Adds train/tis, train/tis_abs, train/tis_clipfrac, train/ois to wandb.
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
   # External SGLang: rollout runs on a separate slurm job (see
   # launch_sglang_server.sh). Slime auto-discovers tp size etc. from the
   # server's /get_server_info. Weight sync goes via NCCL over InfiniBand.
   --rollout-external-engine-addrs "${SGLANG_SERVER_URL}"
   # Server-side we set --disable-custom-all-reduce; slime's external-engine
   # sanity check compares this field, so we must set it here too.
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
   # reward_post_process runs slime's default GRPO group-normalized advantage
   # math. It does NOT log wandb — that's done in log_bcplus below, which is
   # the only hook that receives slime's driver-side rollout_id.
   --custom-reward-post-process-path examples.supo_browsecomp.generate_with_bcplus.reward_post_process
   # log_bcplus aggregates the reward-dict diagnostic keys into bcplus/*
   # wandb metrics, using the driver's rollout_id as rollout/step. Called
   # uniformly for sync and async training paths.
   --custom-rollout-log-function-path examples.supo_browsecomp.generate_with_bcplus.log_bcplus
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
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"LOCAL_SEARCH_URL\": \"${LOCAL_SEARCH_URL}\",
    \"SGLANG_SERVER_URL\": \"${SGLANG_SERVER_URL}\",
    \"LLAMA_API_KEY\": \"${LLAMA_API_KEY}\",
    \"BCPLUS_MAX_TURNS\": \"${BCPLUS_MAX_TURNS:-64}\",
    \"BCPLUS_COMPRESS_THRESH\": \"${BCPLUS_COMPRESS_THRESH:-0.85}\",
    \"BCPLUS_MAX_SUB_TRAJS\": \"${BCPLUS_MAX_SUB_TRAJS:-5}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
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
   ${CUSTOM_ARGS[@]}
