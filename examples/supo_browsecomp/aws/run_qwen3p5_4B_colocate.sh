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
# Physical layout at 8 nodes: TP=4 × CP=2 × PP=1 = 8 GPU per model group,
# 64 GPU / 8 = DP=8. Global batch 256 / DP=8 = 32 trajectories per DP rank.
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
#   BC_NUM_ROLLOUT              default 20   (20 iter × 32 prompts ≈ 1 epoch)
#   BC_ROLLOUT_BATCH_SIZE       default 32   (prompts per iter)
#   BC_N_SAMPLES                default 8    (rollouts per prompt)
#   BC_GLOBAL_BATCH_SIZE        default 256  (= batch × samples, 1 grad step per iter)
#   BC_MAX_RESPONSE_LEN         default 32768 (sglang per-call max_new_tokens)
#   BC_MAX_CONTEXT_LEN          default 65536 (per-sample total context budget)
#   BCPLUS_MAX_TURNS            default 64
#   BCPLUS_COMPRESS_THRESH      default 0.85
#   BCPLUS_MAX_SUB_TRAJS        default 5
#   BCPLUS_COMPRESS_PENALTY     default 0.5
#   BCPLUS_FIXED_SEARCH_TOPK    default empty (model-controlled topk, default
#                               10 and cap 20). Set to 5 for fixed top-5; the
#                               model-facing schema then hides the topk arg.
#   BCPLUS_DOC_WORDS_FULL       default 4096. Set to 10000 to raise the
#                               open_page full-text word cap independently.
#   BCPLUS_DUMP_DIR             default /genai/fsx-project/hhzhang01/dumps/${RUN_NAME}
#                               — per-iter rollout parquet dump is ON by default
#                               so training rollouts can be inspected offline.
#                               Set BCPLUS_DUMP_DIR="" to disable.
#   BCPLUS_DUMP_TRAIN_OLD       default "" (empty/0/false = off). Only meaningful
#                               when BCPLUS_DUMP_DIR is set. When truthy, adds
#                               --dump-train-old-log-prob (extra pre-training
#                               forward pass every iter) and fills the
#                               train_old_logps column. Leave off to dump
#                               trajectories cheaply for inspection.
#   BCPLUS_RAW_ROLLOUT_DIR      default empty (off). When set, writes one full
#                               Sample dump per iteration using slime's native
#                               evaluation-compatible .pt format. This includes
#                               decoded prompt/response, label, reward, tool and
#                               compression metadata, tokens, and loss masks.
#   BCPLUS_CHECKPOINT_ROOT      default /genai/fsx-llm/interns/hhzhang01/checkpoints.
#                               Override with the old FSx root when resuming a
#                               run stored under /genai/fsx-project.
#   BCPLUS_JUDGE_MODEL          default "gpt-5-4-genai-dss4" (MetaGen judge id)
#   BCPLUS_JUDGE_BASE_URL       default "https://api.llama.com/compat/v1/"
#   BCPLUS_JUDGE_CONCURRENCY    default 64  (judge call semaphore)
#   BCPLUS_SEARCH_CONCURRENCY   default 128 (search call semaphore)
#   BCPLUS_DYNAMIC_SAMPLING     default 0. Set to 1/true to sample an initial
#                               pool of 2 x rollout_batch_size prompt groups,
#                               then use a 95% Beta-Binomial top-up policy.
#   SEARCH_BUFFER_HOURS         default 4. Extra runway beyond TRAIN_WALLTIME the
#                               search server must have to be reused; else it is
#                               scancel'd + relaunched (10s Ctrl-C grace).
#   SEARCH_QOS                  default a100_dev. QOS for the search server job
#                               (decoupled from training QOS; keeps it off the
#                               a100_*_high quota).
#   QOS                         default a100_genai_interns_high (fast/high-prio
#                               for the 8-node training srun). Override for a
#                               different training queue.
#   DEV_ALLOCATION_JOB_ID       default empty. Set to an existing persistent
#                               Slurm allocation job ID to run training as a
#                               job step and reuse the imported rootfs.
#   MIN_HOURS_REMAINING         overrides the reuse threshold directly (default
#                               = TRAIN_WALLTIME hours + SEARCH_BUFFER_HOURS).
#   LOCAL_SEARCH_URL            set to skip the ensure step and use this server.

set -euo pipefail

# ---------------------------------------------------------------------------
# Outer part: login pod. Submit ONE srun that spans 8 nodes.
# ---------------------------------------------------------------------------
if [[ "${SLIME_INNER:-0}" != "1" ]]; then
    : "${LLAMA_API_KEY:?LLAMA_API_KEY must be set on the login pod (LLM|... key with entitlement)}"

    # Search server is ensured after the config block below (that logic needs
    # TRAIN_WALLTIME / QOS / SLURM_ACCOUNT / SLIME_HOST_DIR).

    SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
    SLIME_HOST_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

    FIXED_SEARCH_TOPK="${BCPLUS_FIXED_SEARCH_TOPK:-}"
    DOC_WORDS_FULL="${BCPLUS_DOC_WORDS_FULL:-4096}"
    if [[ -n "${FIXED_SEARCH_TOPK}" && ! "${FIXED_SEARCH_TOPK}" =~ ^[1-9][0-9]*$ ]]; then
        echo "BCPLUS_FIXED_SEARCH_TOPK must be a positive integer or empty" >&2
        exit 2
    fi
    if [[ ! "${DOC_WORDS_FULL}" =~ ^[1-9][0-9]*$ ]]; then
        echo "BCPLUS_DOC_WORDS_FULL must be a positive integer" >&2
        exit 2
    fi
    if [[ -n "${FIXED_SEARCH_TOPK}" ]]; then
        SEARCH_CONFIG_TAG="fixedtopk${FIXED_SEARCH_TOPK}"
    else
        SEARCH_CONFIG_TAG="modeltopk"
    fi
    CONFIG_TAG="${SEARCH_CONFIG_TAG}-open${DOC_WORDS_FULL}w"

    export RUN_NAME="${RUN_NAME:-supo-bcplus-${CONFIG_TAG}-qwen3p5-4b-$(date +%Y%m%d-%H%M)}"
    echo "RUN_NAME=${RUN_NAME}"
    echo "BC+ tool config: ${CONFIG_TAG}"

    ENROOT_ROOTFS="${ENROOT_ROOTFS:-slime-test}"
    SLURM_ACCOUNT="${SLURM_ACCOUNT:-genai_interns}"
    QOS="${QOS:-a100_genai_interns_high}"
    TRAIN_WALLTIME="${TRAIN_WALLTIME:-24:00:00}"
    NUM_NODES="${NUM_NODES:-8}"
    DEV_ALLOCATION_JOB_ID="${DEV_ALLOCATION_JOB_ID:-}"
    if [[ -n "${DEV_ALLOCATION_JOB_ID}" && ! "${DEV_ALLOCATION_JOB_ID}" =~ ^[0-9]+$ ]]; then
        echo "DEV_ALLOCATION_JOB_ID must be a numeric Slurm job ID or empty" >&2
        exit 2
    fi
    TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-/genai/fsx-project/hhzhang01/logs/${RUN_NAME}.log}"
    mkdir -p "$(dirname "${TRAIN_LOG_PATH}")"
    export BCPLUS_CHECKPOINT_ROOT="${BCPLUS_CHECKPOINT_ROOT:-/genai/fsx-llm/interns/hhzhang01/checkpoints}"

    # Dump per-iter rollout parquet BY DEFAULT so training rollouts can be
    # inspected offline (token ids + loss_mask + rollout_logps + advantage +
    # outcome/score, one file per DP rank per iter, written to fast Lustre).
    # Uses `-` (not `:-`): unset -> default path (on); explicit BCPLUS_DUMP_DIR=""
    # -> off. train_old stays off (no extra forward pass) unless
    # BCPLUS_DUMP_TRAIN_OLD=1.
    export BCPLUS_DUMP_DIR="${BCPLUS_DUMP_DIR-/genai/fsx-project/hhzhang01/dumps/${RUN_NAME}}"

    # Ensure a healthy retrieval search server with enough runway for this run.
    # Idempotent (launch_search_server.sh): reuses a RUNNING server that has
    # >= MIN_HOURS_REMAINING left, else scancel + resubmit a fresh 7-day job
    # (with a 10s Ctrl-C grace — abort if another experiment relies on it).
    # This is the search-server ensure the retired external-sglang orchestrator
    # used to do. Set LOCAL_SEARCH_URL yourself to skip this and use a specific server.
    if [[ -z "${LOCAL_SEARCH_URL:-}" ]]; then
        # Default runway = this run's walltime (hours) + buffer, so the server
        # outlives training. Falls back to launch_search_server.sh's own default
        # if TRAIN_WALLTIME isn't a plain HH:MM:SS.
        _wt_hours="${TRAIN_WALLTIME%%:*}"
        _min_hours="${MIN_HOURS_REMAINING:-}"
        if [[ -z "${_min_hours}" && "${_wt_hours}" =~ ^[0-9]+$ ]]; then
            _min_hours="$((_wt_hours + ${SEARCH_BUFFER_HOURS:-4}))"
        fi
        # The search server runs on its OWN qos (SEARCH_QOS, default a100_dev):
        # decoupled from the training QOS so it never eats the a100_*_high quota,
        # gets high priority, and can hold the full 7-day walltime. Passed inline
        # so this run's ${QOS} (used for the training srun below) is untouched.
        echo "[colocate] ensuring search server on qos=${SEARCH_QOS:-a100_dev} (>= ${_min_hours:-48}h runway)"
        MIN_HOURS_REMAINING="${_min_hours}" QOS="${SEARCH_QOS:-a100_dev}" SLURM_ACCOUNT="${SLURM_ACCOUNT}" \
            bash "${SLIME_HOST_DIR}/examples/supo_browsecomp/aws/search/launch_search_server.sh"
        SEARCH_HOST_FILE=/genai/fsx-project/hhzhang01/logs/search-server.hostname
        [[ -f "${SEARCH_HOST_FILE}" ]] || {
            echo "ERROR: ${SEARCH_HOST_FILE} missing after launch_search_server.sh" >&2
            exit 1
        }
        export LOCAL_SEARCH_URL="http://$(cat "${SEARCH_HOST_FILE}")"
    fi
    echo "[colocate] LOCAL_SEARCH_URL=${LOCAL_SEARCH_URL}"

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

    # One srun spanning all N nodes. `--ntasks-per-node=1` -> one enroot per node.
    # Normally this creates an exclusive allocation. For iterative development,
    # DEV_ALLOCATION_JOB_ID attaches the step to a persistent sbatch allocation;
    # --overlap lets it coexist with that allocation's lightweight holder loop.
    #
    # We background srun and drive a wandb-sync poll loop from this login-pod
    # shell while training runs. This lets you watch
    # loss curves on wandb.ai mid-run without waiting for the whole job to
    # finish. Sync interval defaults to 5 min (override via WANDB_SYNC_INTERVAL_SEC).
    WANDB_SYNC_INTERVAL_SEC="${WANDB_SYNC_INTERVAL_SEC:-300}"
    SYNC_SCRIPT="${SLIME_HOST_DIR}/aws-cluster/wandb-sync.sh"

    SRUN_ARGS=(
        --nodes="${NUM_NODES}"
        --gpus-per-node=8
        --ntasks-per-node=1
        --cpus-per-task=64
        --mem=0
        --mpi=none
        --job-name="${RUN_NAME}"
        --output="${TRAIN_LOG_PATH}"
    )
    if [[ -n "${DEV_ALLOCATION_JOB_ID}" ]]; then
        SRUN_ARGS+=(--jobid="${DEV_ALLOCATION_JOB_ID}" --overlap)
        echo "[outer] attaching to persistent Slurm allocation ${DEV_ALLOCATION_JOB_ID}"
        if [[ "${NUM_NODES}" == "1" ]]; then
            ENROOT_DATA_MODE=shared
        else
            ENROOT_DATA_MODE=staged
        fi
    else
        ENROOT_DATA_MODE=staged
        SRUN_ARGS+=(
            --exclusive
            --account="${SLURM_ACCOUNT}"
            --qos="${QOS}"
            --time="${TRAIN_WALLTIME}"
        )
    fi

    srun "${SRUN_ARGS[@]}" \
        bash -c "
            if [[ '${ENROOT_DATA_MODE}' == shared ]]; then
                # Persistent dev allocations run one enroot step at a time, so
                # they can reuse the shared imported rootfs without the
                # multi-node flock race or a 24 GB copy on every retry.
                LOCAL_ENROOT_DATA=/storage/home/hhzhang01/.local/share/enroot
                echo \"[node \${SLURM_NODEID:-0}] using shared enroot rootfs for dev allocation\"
            else
                # Pre-stage rootfs to per-node local /dev/shm to avoid flock()
                # failures when multiple nodes start concurrently on NFS4.
                LOCAL_ENROOT_DATA=/dev/shm/enroot-\${USER}-\${SLURM_JOB_ID}
                LOCAL_ROOTFS=\${LOCAL_ENROOT_DATA}/${ENROOT_ROOTFS}
                ROOTFS_READY=\${LOCAL_ROOTFS}/.slime-stage-complete
                if [[ ! -f \${ROOTFS_READY} ]]; then
                    mkdir -p \${LOCAL_ROOTFS}
                    echo \"[node \${SLURM_NODEID:-0}] staging rootfs FSx -> \${LOCAL_ROOTFS} ...\"
                    # Imported rootfs directories can retain host/Slurm runtime
                    # mounts that are unreadable and are recreated at launch.
                    # Exclude only those transient paths; a completion marker
                    # prevents a partial copy from being reused after failure.
                    if ! time rsync -a \
                        --exclude='/tmp_host/' \
                        --exclude='/var/spool/slurmd/pmix.*' \
                        /storage/home/hhzhang01/.local/share/enroot/${ENROOT_ROOTFS}/ \
                        \${LOCAL_ROOTFS}/; then
                        echo \"[node \${SLURM_NODEID:-0}] rootfs staging failed\" >&2
                        exit 1
                    fi
                    touch \${ROOTFS_READY}
                    echo \"[node \${SLURM_NODEID:-0}] rootfs staged\"
                fi
            fi

            ENROOT_TEMP_PATH=/dev/shm \
            ENROOT_DATA_PATH=\${LOCAL_ENROOT_DATA} \
            ENROOT_MOUNT_HOME=false \
            enroot start \
                --mount ${SLIME_HOST_DIR}:/slime \
                --mount ${SLIME_HOST_DIR}/aws-cluster:/aws-cluster \
                --mount /genai/fsx-project/hhzhang01:/genai_hh \
                --mount /genai/fsx-llm/interns/hhzhang01:/genai_llm \
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
                --env BCPLUS_FIXED_SEARCH_TOPK='${BCPLUS_FIXED_SEARCH_TOPK:-}' \
                --env BCPLUS_DOC_WORDS_FULL='${BCPLUS_DOC_WORDS_FULL:-}' \
                --env BCPLUS_DUMP_DIR='${BCPLUS_DUMP_DIR:-}' \
                --env BCPLUS_DUMP_TRAIN_OLD='${BCPLUS_DUMP_TRAIN_OLD:-}' \
                --env BCPLUS_RAW_ROLLOUT_DIR='${BCPLUS_RAW_ROLLOUT_DIR:-}' \
                --env BCPLUS_CHECKPOINT_ROOT='${BCPLUS_CHECKPOINT_ROOT}' \
                --env BCPLUS_JUDGE_MODEL='${BCPLUS_JUDGE_MODEL:-}' \
                --env BCPLUS_JUDGE_BASE_URL='${BCPLUS_JUDGE_BASE_URL:-}' \
                --env BCPLUS_JUDGE_CONCURRENCY='${BCPLUS_JUDGE_CONCURRENCY:-}' \
                --env BCPLUS_SEARCH_CONCURRENCY='${BCPLUS_SEARCH_CONCURRENCY:-}' \
                --env BCPLUS_DYNAMIC_SAMPLING='${BCPLUS_DYNAMIC_SAMPLING:-}' \
                --env BC_NUM_ROLLOUT='${BC_NUM_ROLLOUT:-}' \
                --env BC_ROLLOUT_BATCH_SIZE='${BC_ROLLOUT_BATCH_SIZE:-}' \
                --env BC_N_SAMPLES='${BC_N_SAMPLES:-}' \
                --env BC_GLOBAL_BATCH_SIZE='${BC_GLOBAL_BATCH_SIZE:-}' \
                --env BC_MAX_RESPONSE_LEN='${BC_MAX_RESPONSE_LEN:-}' \
                --env BC_MAX_CONTEXT_LEN='${BC_MAX_CONTEXT_LEN:-}' \
                --env WANDB_X_FLUSH_INTERVAL_SECONDS='${WANDB_X_FLUSH_INTERVAL_SECONDS:-30}' \
                ${ENROOT_ROOTFS} \
                bash /slime/examples/supo_browsecomp/aws/run_qwen3p5_4B_colocate.sh
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
TEST_DATA="${BC_TEST_DATA:-/genai_hh/datasets/BC+/bc_test.parquet}"
CHECKPOINT_ROOT="${BCPLUS_CHECKPOINT_ROOT:-/genai/fsx-llm/interns/hhzhang01/checkpoints}"
case "${CHECKPOINT_ROOT}" in
    /genai/fsx-llm/interns/hhzhang01*)
        CHECKPOINT_ROOT="/genai_llm${CHECKPOINT_ROOT#/genai/fsx-llm/interns/hhzhang01}"
        ;;
    /genai/fsx-project/hhzhang01*)
        CHECKPOINT_ROOT="/genai_hh${CHECKPOINT_ROOT#/genai/fsx-project/hhzhang01}"
        ;;
esac
CKPT_SAVE_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}"
mkdir -p "${CKPT_SAVE_DIR}"
echo "[head] checkpoint directory: ${CKPT_SAVE_DIR}"

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
   # fully on-policy). Without dynamic sampling, 20 iter × 32 prompts = 640
   # prompts ≈ 1 epoch of the 680-prompt training set.
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

case "$(printf '%s' "${BCPLUS_DYNAMIC_SAMPLING:-0}" | tr '[:upper:]' '[:lower:]')" in
    1|true)
        DYNAMIC_FIRST_POOL_SIZE=$((2 * ${BC_ROLLOUT_BATCH_SIZE:-32}))
        ROLLOUT_ARGS+=(
            --rollout-function-path examples.supo_browsecomp.dynamic_sampling.generate_rollout
            --over-sampling-batch-size "${DYNAMIC_FIRST_POOL_SIZE}"
        )
        echo "[BCPLUS] adaptive dynamic sampling enabled: first_pool=${DYNAMIC_FIRST_POOL_SIZE}"
        ;;
    ""|0|false) : ;;
    *)
        echo "BCPLUS_DYNAMIC_SAMPLING must be one of: 1, true, 0, false" >&2
        exit 2
        ;;
esac

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
   # max-tokens-per-gpu: 48k tokens per microbatch per DP rank (was 32k after
   # the CP=1→CP=2 fix). Empirical peak used_GB on 293928 was 22.58 GB out of
   # 79.25 GB (28.5% util) with 32k; ~56 GB free. 48k = 1.5x should push
   # peak to ~30-35 GB, still well within budget. Larger microbatch = fewer
   # microbatch iterations per training step = less framework overhead.
   --max-tokens-per-gpu 49152
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
   --custom-advantage-function-path examples.supo_browsecomp.summary_advantage.compute_summary_aware_advantages
   --custom-rollout-log-function-path examples.supo_browsecomp.generate_with_bcplus.log_bcplus
   --rollout-data-postprocess-path   examples.supo_browsecomp.generate_with_bcplus.dump_rollout_data_postprocess
)

# When dumping is enabled, force slime to run the pre-training forward pass
# that populates rollout_data["log_probs"] (train_old). See run_qwen3p5_4B.sh
# comments for why can_reuse_log_probs_in_loss otherwise skips it.
if [[ -n "${BCPLUS_DUMP_DIR:-}" ]]; then
    # train_old log_probs are opt-in via BCPLUS_DUMP_TRAIN_OLD (empty/0/false =
    # off): only then add --dump-train-old-log-prob to force the pre-training
    # forward pass. Otherwise trajectories dump without train_old (null column).
    case "$(printf '%s' "${BCPLUS_DUMP_TRAIN_OLD:-}" | tr '[:upper:]' '[:lower:]')" in
        ""|0|false) : ;;
        *) CUSTOM_ARGS+=(--dump-train-old-log-prob) ;;
    esac
    if [[ "${BCPLUS_DUMP_DIR}" == /genai/fsx-project/hhzhang01/* ]]; then
        BCPLUS_DUMP_DIR_CONTAINER="${BCPLUS_DUMP_DIR/#\/genai\/fsx-project\/hhzhang01/\/genai_hh}"
        echo "[BCPLUS] auto-translated BCPLUS_DUMP_DIR host=${BCPLUS_DUMP_DIR} -> container=${BCPLUS_DUMP_DIR_CONTAINER}"
        export BCPLUS_DUMP_DIR="${BCPLUS_DUMP_DIR_CONTAINER}"
    fi
fi

# Optional full-Sample trajectory dump. This is the same raw format used by
# evaluation's eval_0.pt and can be consumed by eval/eval_pipeline.py helpers.
# Keep it separate from the parquet dump above: the .pt owns decoded research
# trajectories and metadata, while parquet owns training-time logprob tensors.
if [[ -n "${BCPLUS_RAW_ROLLOUT_DIR:-}" ]]; then
    if [[ "${BCPLUS_RAW_ROLLOUT_DIR}" == /genai/fsx-project/hhzhang01/* ]]; then
        BCPLUS_RAW_ROLLOUT_DIR="${BCPLUS_RAW_ROLLOUT_DIR/#\/genai\/fsx-project\/hhzhang01/\/genai_hh}"
    fi
    CUSTOM_ARGS+=(
        --save-debug-rollout-data
        "${BCPLUS_RAW_ROLLOUT_DIR}/rollout_{rollout_id}.pt"
    )
    echo "[BCPLUS] full Sample trajectories: ${BCPLUS_RAW_ROLLOUT_DIR}/rollout_{rollout_id}.pt"
fi

# Colocate + offload: sglang engines and training actor share the same 64
# GPUs. --colocate implies --offload-train and --offload-rollout.
COLOCATE_ARGS=(
   --colocate
)

# ----------------------------------------------------------------------------
# Eval-only mode (opt-in via BC_EVAL_MODE=1; no-op otherwise).
#
# Reuses the ENTIRE training config above (model / sglang / custom generate +
# reward + judge / colocate), and only flips to slime's eval-only path:
#   * --num-rollout 0 + --eval-interval 1 -> train.py runs ONE eval at rollout 0
#     then exits (train.py:36-37). argparse "last wins" makes the appended
#     --num-rollout 0 override ROLLOUT_ARGS' --num-rollout ${BC_NUM_ROLLOUT}.
#   * --eval-prompt-data feeds the 150-question BC+ test set. eval inherits
#     input/label/metadata keys + temperature + max_response_len + custom_rm
#     from the training args via slime/utils/eval_config.py fallbacks, and the
#     eval path dispatches to args.custom_generate_function_path (our multi-turn
#     ReAct generate) via sglang_rollout.py:251 — same search + judge as train.
#   * --dump-details writes <dir>/rollout_data/eval_0.pt with every eval Sample
#     (token ids + decoded response + reward + _bcplus metadata) — no HF
#     conversion needed: actor.update_weights() (train.py:27) syncs the loaded
#     megatron checkpoint into sglang BEFORE the eval runs.
#
# Legacy Slurm eval is driven by eval/legacy/slurm/run_qwen3p5_4B_eval.sh:
#   BC_EVAL_LOAD  = <run>/iter_0000NN (host path) to eval that iter, or
#                 = "base"/"" -> no --load (base weights from --hf-checkpoint)
#   BC_EVAL_N     = --n-samples-per-eval-prompt (default 4)
#   EVAL_DUMP_DIR = host dir for the eval dump (auto host->/genai_hh remapped)
#
# Checkpoint selection is ELEGANT: point --load at the RUN dir (it has the
# latest_checkpointed_iteration.txt tracker) and pass --ckpt-step N — megatron
# overrides the tracker's iteration with args.ckpt_step (checkpointing.py:215,
# 1188). No per-iter tracker files / symlink dirs needed. Two consequences we
# handle here:
#   * --ckpt-step is applied to EVERY checkpoint load (actor AND ref), so eval
#     runs WITHOUT a reference model — drop --use-kl-loss (=> with_ref=False,
#     placement_group.py:178) and don't pass --ref-load. Ref/KL are unused in eval.
#   * a trained checkpoint carries optimizer+scheduler state whose total-iters
#     (e.g. 5120) would clash with our dummy --lr-decay-iters 1 and trip
#     `OptimizerParamScheduler ... do not match`. --no-load-optim/--no-load-rng
#     load ONLY model weights (eval needs no optimizer), sidestepping that.
# ----------------------------------------------------------------------------
EVAL_ARGS=()
if [[ "${BC_EVAL_MODE:-}" == "1" ]]; then
    : "${EVAL_DUMP_DIR:?BC_EVAL_MODE=1 requires EVAL_DUMP_DIR}"
    # Host -> container path remap (same idiom as BCPLUS_DUMP_DIR above).
    if [[ "${EVAL_DUMP_DIR}" == /genai/fsx-project/hhzhang01/* ]]; then
        EVAL_DUMP_DIR="${EVAL_DUMP_DIR/#\/genai\/fsx-project\/hhzhang01/\/genai_hh}"
    fi

    EVAL_ARGS=(
        --num-rollout 0
        --eval-interval 1
        --eval-prompt-data bcplus_test "${TEST_DATA}"
        --n-samples-per-eval-prompt "${BC_EVAL_N:-4}"
        --dump-details "${EVAL_DUMP_DIR}"
        # num-rollout 0 makes train_iters=0 (model.py:204), which would zero the
        # LR-decay schedule and trip `assert lr_decay_steps > 0` in megatron's
        # OptimizerParamScheduler when the (unused) optimizer is still BUILT.
        --lr-decay-iters 1
    )

    # Actor checkpoint: --load <run> + --ckpt-step N selects the iter (no ref, no
    # symlink view dirs). base = --hf-checkpoint only (no --load / --ckpt-step).
    NEW_CKPT_ARGS=(--hf-checkpoint "${HF_CKPT_HOST}")
    if [[ -n "${BC_EVAL_LOAD:-}" && "${BC_EVAL_LOAD}" != "base" ]]; then
        RUN_DIR="$(dirname "${BC_EVAL_LOAD%/}")"
        ITER_NAME="$(basename "${BC_EVAL_LOAD%/}")"
        if [[ ! "${ITER_NAME}" =~ ^iter_[0-9]{7}$ ]]; then
            echo "[eval] ERROR: BC_EVAL_LOAD must end in iter_NNNNNNN or be 'base', got ${BC_EVAL_LOAD}" >&2
            exit 1
        fi
        CKPT_STEP=$((10#${ITER_NAME#iter_}))
        if [[ "${RUN_DIR}" == /genai/fsx-project/hhzhang01/* ]]; then
            RUN_DIR="${RUN_DIR/#\/genai\/fsx-project\/hhzhang01/\/genai_hh}"
        fi
        # --no-load-optim/--no-load-rng: load ONLY model weights (a trained
        # checkpoint's optimizer/scheduler total-iters would clash with the dummy
        # --lr-decay-iters 1). Only in the iter branch: base has no --load, and
        # --no-load-optim without --load trips `assert args.load is not None ...`.
        NEW_CKPT_ARGS+=(--load "${RUN_DIR}" --ckpt-step "${CKPT_STEP}" --no-load-optim --no-load-rng)
        echo "[eval] loading iter ${CKPT_STEP}: --load ${RUN_DIR} --ckpt-step ${CKPT_STEP}"
    else
        # base = untrained model. It still needs a --load: slime asserts
        # args.load/pretrained_checkpoint in setup_model_and_optimizer (model.py:292),
        # and with no ref model there is otherwise no checkpoint source. Load the
        # base torch_dist (a release checkpoint = same base weights as --hf-checkpoint).
        NEW_CKPT_ARGS+=(--load "${REF_LOAD_HOST}" --no-load-optim --no-load-rng)
        echo "[eval] base model via --load ${REF_LOAD_HOST} (base torch_dist release)"
    fi
    CKPT_ARGS=("${NEW_CKPT_ARGS[@]}")

    # No reference model in eval: args.ckpt_step is global (would also hit the ref
    # load and break it). Dropping --use-kl-loss => with_ref=False, so no ref is
    # created/loaded. KL/ref are unused in eval anyway.
    GRPO_ARGS=(--advantage-estimator grpo)

    # Drop the training-only rollout-data dump hook (fires during train() only)
    # and --dump-train-old-log-prob (no training step in eval). Keep the custom
    # generate + reward + reward-post + rollout-log hooks (eval uses all of them).
    CUSTOM_ARGS=(
        --custom-generate-function-path examples.supo_browsecomp.generate_with_bcplus.generate
        --custom-rm-path                 examples.supo_browsecomp.generate_with_bcplus.reward_func
        --reward-key score
        --custom-reward-post-process-path examples.supo_browsecomp.generate_with_bcplus.reward_post_process
        --custom-rollout-log-function-path examples.supo_browsecomp.generate_with_bcplus.log_bcplus
    )
    echo "[eval] BC_EVAL_MODE=1 -> eval-only (no ref), n=${BC_EVAL_N:-4}, dump -> ${EVAL_DUMP_DIR}"
fi

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

# RUNTIME_ENV_JSON contains LLAMA_API_KEY. Keep xtrace disabled until the Ray
# submission returns so neither the assignment nor the CLI argument leaks the
# credential into the Slurm training log.
set +x
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
    \"BCPLUS_FIXED_SEARCH_TOPK\": \"${BCPLUS_FIXED_SEARCH_TOPK:-}\",
    \"BCPLUS_DOC_WORDS_FULL\": \"${BCPLUS_DOC_WORDS_FULL:-4096}\",
    \"BCPLUS_DUMP_DIR\": \"${BCPLUS_DUMP_DIR:-}\",
    \"BCPLUS_DUMP_TRAIN_OLD\": \"${BCPLUS_DUMP_TRAIN_OLD:-}\",
    \"BCPLUS_JUDGE_MODEL\": \"${BCPLUS_JUDGE_MODEL:-gpt-5-4-genai-dss4}\",
    \"BCPLUS_JUDGE_BASE_URL\": \"${BCPLUS_JUDGE_BASE_URL:-https://api.llama.com/compat/v1/}\",
    \"BCPLUS_JUDGE_CONCURRENCY\": \"${BCPLUS_JUDGE_CONCURRENCY:-64}\",
    \"BCPLUS_SEARCH_CONCURRENCY\": \"${BCPLUS_SEARCH_CONCURRENCY:-128}\",
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
   "${EVAL_ARGS[@]}" \
   "${COLOCATE_ARGS[@]}"

TRAIN_STATUS=$?
unset RUNTIME_ENV_JSON
set -x
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
