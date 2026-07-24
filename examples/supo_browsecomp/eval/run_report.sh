#!/bin/bash
# Run build_eval_report.py inside the slime enroot container on a compute node
# (the login pod has no GPU driver, so enroot can't start there). CPU-only work
# — pickle load + parse — but the partition requires a GPU allocation.
#
# Usage (from the login pod):
#   EVAL_ROOT=/genai/fsx-project/hhzhang01/evals/<run>-sweep \
#   bash examples/supo_browsecomp/eval/run_report.sh
#
# Blocks until the builder finishes (srun, foreground) and tails its output to
# REPORT_LOG. The generated eval_summary.json / summary_table.md / failures land
# under EVAL_ROOT.

set -euo pipefail

: "${EVAL_ROOT:?set EVAL_ROOT (host path under /genai/fsx-project/hhzhang01)}"
# Host -> container remap for the builder's --eval-root arg.
EVAL_ROOT_CTR="${EVAL_ROOT/#\/genai\/fsx-project\/hhzhang01/\/genai_hh}"

SLIME_HOST_DIR=/home/hhzhang01/slime
ENROOT_ROOTFS="${ENROOT_ROOTFS:-slime-test}"
ENROOT_DIR=/storage/home/hhzhang01/.local/share/enroot
SLURM_ACCOUNT="${SLURM_ACCOUNT:-genai_interns}"
QOS="${QOS:-a100_genai_interns_high}"
REPORT_LOG="${REPORT_LOG:-/genai/fsx-project/hhzhang01/logs/eval-report-$(basename "${EVAL_ROOT}").log}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# Which analysis script to run (default: the Stage-A report builder).
REPORT_SCRIPT="${REPORT_SCRIPT:-examples/supo_browsecomp/eval/build_eval_report.py}"

echo "[report] EVAL_ROOT=${EVAL_ROOT} -> ${EVAL_ROOT_CTR}"
echo "[report] log -> ${REPORT_LOG}"

srun \
    --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 \
    --cpus-per-task=16 --mem=64G \
    --account="${SLURM_ACCOUNT}" --qos="${QOS}" \
    --time="0:20:00" --mpi=none \
    --job-name="eval-report" \
    --output="${REPORT_LOG}" \
    bash -c "
        ENROOT_TEMP_PATH=/dev/shm \
        ENROOT_DATA_PATH=${ENROOT_DIR} \
        ENROOT_MOUNT_HOME=false \
        enroot start \
            --mount ${SLIME_HOST_DIR}:/slime \
            --mount /genai/fsx-project/hhzhang01:/genai_hh \
            ${ENROOT_ROOTFS} \
            bash -c 'cd /slime && python3 ${REPORT_SCRIPT} --eval-root ${EVAL_ROOT_CTR} ${EXTRA_ARGS}'
    "
echo "[report] done; see ${REPORT_LOG} and ${EVAL_ROOT}/summary_table.md"
