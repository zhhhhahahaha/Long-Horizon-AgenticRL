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
#   BCPLUS_COMPRESS_PENALTY     default 0.5 (per-sub-traj penalty when a
#                                compression turn failed to emit a real
#                                <summary> block; set to 0 to disable)
#   BCPLUS_DUMP_DIR             default "" (empty = disabled). When set,
#                                per-iter parquet dumps of prompt_ids,
#                                response_ids, loss_mask, rollout_logps,
#                                train_old_logps, advantage go to
#                                $BCPLUS_DUMP_DIR/rollouts_iter_NNNNN_dp*.parquet
#                                for offline TIS drift analysis.

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
                --env BCPLUS_COMPRESS_PENALTY='${BCPLUS_COMPRESS_PENALTY:-}' \
                --env BCPLUS_DUMP_DIR='${BCPLUS_DUMP_DIR:-}' \
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
   # dump_rollout_data_postprocess runs in the trainer after
   # rollout_log_probs + train_old log_probs + advantages are computed but
   # before training starts. Writes per-iter parquet of rollout + train-old
   # state for offline TIS drift / advantage attribution analysis. Only fires
   # when BCPLUS_DUMP_DIR is non-empty (default disabled).
   --rollout-data-postprocess-path examples.supo_browsecomp.generate_with_bcplus.dump_rollout_data_postprocess
)

# When dumping is enabled, force slime to run the pre-training forward pass
# that populates rollout_data["log_probs"] (train_old). Without this, our
# canonical config trips can_reuse_log_probs_in_loss=True (num_rollouts ==
# global_batch_size → 1 training step per iter) and slime skips the forward,
# leaving train_old_logps empty in the parquet. See dp_schedule.py comments
# and actor.py:439-451 for the reuse logic.
if [[ -n "${BCPLUS_DUMP_DIR:-}" ]]; then
    CUSTOM_ARGS+=(--dump-train-old-log-prob)
    # Users typically pass the login-pod-visible path (/genai/fsx-project/
    # hhzhang01/...) since that's what they see with ls / tail. But inside
    # the container that mount is remapped to /genai_hh (see enroot start
    # --mount above). Auto-translate so dumps land on the actually-mounted
    # FSx volume instead of the ephemeral container rootfs (where they
    # vanish when the srun ends).
    if [[ "${BCPLUS_DUMP_DIR}" == /genai/fsx-project/hhzhang01/* ]]; then
        BCPLUS_DUMP_DIR_CONTAINER="${BCPLUS_DUMP_DIR/#\/genai\/fsx-project\/hhzhang01/\/genai_hh}"
        echo "[BCPLUS] auto-translated BCPLUS_DUMP_DIR host=${BCPLUS_DUMP_DIR} -> container=${BCPLUS_DUMP_DIR_CONTAINER}"
        export BCPLUS_DUMP_DIR="${BCPLUS_DUMP_DIR_CONTAINER}"
    fi
fi

MISC_ARGS+=(
   # log_multi_turn_data reads sample.metadata["round_number"] we set in generate()
   # and produces multi_turn_metric/round_number_* plus raw/observed response
   # length stats. Safe with SUPO compression (sub-traj list) because it does
   # per-sample mean/max/min aggregation without any group-size assertion.
   --log-multi-turn
   # NOTE: --log-passrate is intentionally OFF. Unlike --log-multi-turn,
   # slime's compute_pass_rate asserts len(flat_rewards) ==
   # rollout_batch_size * n_samples_per_prompt, but our generate() returns
   # list[Sample] (one per SUPO sub-trajectory) so flat_rewards grows with
   # compression. The pass@k intent is already covered by bcplus/score_max
   # + bcplus/score_hits in log_bcplus (which dedupes by rollout_id, so
   # they reflect per-parent-rollout hit counts).
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
    \"BCPLUS_MAX_SUB_TRAJS\": \"${BCPLUS_MAX_SUB_TRAJS:-5}\",
    \"BCPLUS_COMPRESS_PENALTY\": \"${BCPLUS_COMPRESS_PENALTY:-0.5}\",
    \"BCPLUS_DUMP_DIR\": \"${BCPLUS_DUMP_DIR:-}\"
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
