#!/bin/bash
# Convert Qwen3.5-9B HF checkpoint to Megatron torch_dist format for slime.
#
# One-shot preflight for the 9B baseline: needs to run before you can pass
# --ref-load /genai/fsx-project/hhzhang01/models/Qwen3.5-9B_torch_dist to
# any training job.
#
# Requires:
#   * HF checkpoint downloaded at /genai/fsx-project/hhzhang01/models/Qwen3.5-9B/
#   * Enroot rootfs `slime-test` available (same as the training runs)
#
# Usage:
#   bash examples/supo_browsecomp/convert_qwen3p5_9B.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Outer part: login pod, submits a small sbatch job on 1 GPU.
# ---------------------------------------------------------------------------
if [[ "${SLIME_INNER:-0}" != "1" ]]; then
    SLIME_HOST_DIR=/home/hhzhang01/slime
    ENROOT_ROOTFS="${ENROOT_ROOTFS:-slime-test}"
    SLURM_ACCOUNT="${SLURM_ACCOUNT:-genai_interns}"
    QOS="${QOS:-a100_genai_interns_high}"
    WALLTIME="${WALLTIME:-1:00:00}"
    JOB_NAME="convert-qwen3p5-9b"
    LOG_PATH=/genai/fsx-project/hhzhang01/logs/${JOB_NAME}-$(date +%Y%m%d-%H%M).log
    mkdir -p "$(dirname "${LOG_PATH}")"
    echo "log path: ${LOG_PATH}"

    exec srun \
        --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --exclusive \
        --cpus-per-task=8 --mem=0 \
        --account="${SLURM_ACCOUNT}" --qos="${QOS}" \
        --time="${WALLTIME}" \
        --mpi=none \
        --job-name="${JOB_NAME}" \
        --output="${LOG_PATH}" \
        bash -c "
            ENROOT_TEMP_PATH=/dev/shm \
            ENROOT_DATA_PATH=/storage/home/hhzhang01/.local/share/enroot \
            ENROOT_MOUNT_HOME=false \
            enroot start \
                --mount ${SLIME_HOST_DIR}:/slime \
                --mount /genai/fsx-project/hhzhang01:/genai_hh \
                --env SLIME_INNER=1 \
                ${ENROOT_ROOTFS} \
                bash /slime/examples/supo_browsecomp/convert_qwen3p5_9B.sh
        "
fi

# ---------------------------------------------------------------------------
# Inner part: in-container, runs the actual convert.
# ---------------------------------------------------------------------------
set -x
export PYTHONUNBUFFERED=1

cd /slime
source scripts/models/qwen3.5-9B.sh

HF_CKPT=/genai_hh/models/Qwen3.5-9B
SAVE_CKPT=/genai_hh/models/Qwen3.5-9B_torch_dist

PYTHONPATH=/root/Megatron-LM/:/slime python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint "${HF_CKPT}" \
    --save "${SAVE_CKPT}"

echo "Done. Output at: ${SAVE_CKPT}"
ls -la "${SAVE_CKPT}"
