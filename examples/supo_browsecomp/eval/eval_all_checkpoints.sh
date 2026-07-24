#!/bin/bash
# Orchestrator: submit one eval job per checkpoint (base + all saved iters) of a
# BC+ 4B training run, on the high-priority QOS. Slurm runs up to 8 concurrently
# (64-GPU / 8-node cap) and queues the rest.
#
# Each job writes EVAL_ROOT/<point>/rollout_data/eval_0.pt. Build the report
# afterwards with eval/build_eval_report.py once all dumps exist.
#
# Usage (from the login pod):
#   export LLAMA_API_KEY='LLM|...'
#   bash examples/supo_browsecomp/eval/eval_all_checkpoints.sh
#
# Env overrides:
#   RUN        training run to evaluate (default: the 20260720-2059 4B run)
#   BC_EVAL_N  samples per question (default 4)
#   POINTS     space-separated subset, e.g. POINTS="base iter44" (default: all)
#   ITERS      override the iter list (default: 4 9 14 19 24 29 34 39 44)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${RUN:-supo-bcplus-qwen3p5-4b-20260720-2059}"
CKPT="/genai/fsx-project/hhzhang01/checkpoints/${RUN}"
EVAL_ROOT="${EVAL_ROOT:-/genai/fsx-project/hhzhang01/evals/${RUN}-sweep}"
BC_EVAL_N="${BC_EVAL_N:-4}"
ITERS="${ITERS:-4 9 14 19 24 29 34 39 44}"

# Build the list of eval points (base + each iter) unless POINTS is given.
if [[ -z "${POINTS:-}" ]]; then
    POINTS="base"
    for it in ${ITERS}; do
        POINTS="${POINTS} iter$(printf '%02d' "${it}")"
    done
fi

echo "[sweep] RUN=${RUN}"
echo "[sweep] EVAL_ROOT=${EVAL_ROOT}"
echo "[sweep] n=${BC_EVAL_N}, points: ${POINTS}"
mkdir -p "${EVAL_ROOT}"

for pt in ${POINTS}; do
    if [[ "${pt}" == "base" ]]; then
        LOAD="base"
    else
        it="${pt#iter}"; it="$((10#${it}))"   # strip zero-pad
        LOAD="${CKPT}/iter_$(printf '%07d' "${it}")"
        if [[ ! -d "${LOAD}" ]]; then
            echo "[sweep] WARN: ${LOAD} missing, skipping ${pt}" >&2
            continue
        fi
    fi

    RUN_NAME="eval-4b-${pt}" \
    BC_EVAL_LOAD="${LOAD}" \
    EVAL_DUMP_DIR="${EVAL_ROOT}/${pt}" \
    BC_EVAL_N="${BC_EVAL_N}" \
    bash "${HERE}/run_qwen3p5_4B_eval.sh"
done

echo "[sweep] all jobs submitted. Watch: squeue -u \$USER -o '%.10i %.24j %.8T %.11L'"
echo "[sweep] dumps will land under: ${EVAL_ROOT}/<point>/rollout_data/eval_0.pt"
