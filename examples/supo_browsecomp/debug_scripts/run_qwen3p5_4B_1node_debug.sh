#!/bin/bash
# 1-node debug wrapper for run_qwen3p5_4B_colocate.sh.
#
# Purpose: validate the 4B TP=4 colocate config on 1 node before spending an
# 8-node backfill slot. Inter-node parallelism in the canonical run is pure
# DP, so 1 node with matched per-DP-rank workload reproduces the same
# per-GPU memory pressure.
#
# Workload matching:
#   Canonical 8-node: TP=4, CP=1, DP=16, global_batch=256 -> 16 samples per DP-rank
#   1-node debug:     TP=4, CP=1, DP=2,  global_batch=32  -> 16 samples per DP-rank  ← match
#   (n_samples_per_prompt=8 stays; 32/8 = 4 prompts per iter)
#
# Long context stays FULL (rollout_max_context_len=65536) so we stress the
# vocab head + backward memory path. num_rollout=2 covers iter 0 + iter 1
# (iter 1 is where our earlier 9B TP=4 hit OOM on actor backward).
#
# Usage:
#   bash examples/supo_browsecomp/debug_scripts/run_qwen3p5_4B_1node_debug.sh

set -euo pipefail

# ---- 1-node debug overrides ----
export NUM_NODES=1
# Match per-DP workload (16 samples/DP): 4 prompts × 8 samples = 32 rollouts,
# split across DP=2 groups (1 node with TP=4 CP=1 gives 2 DP groups).
export BC_ROLLOUT_BATCH_SIZE=4
export BC_N_SAMPLES=8
export BC_GLOBAL_BATCH_SIZE=32
# Full-size context — stress the memory-heavy vocab head path.
export BC_MAX_RESPONSE_LEN=32768
export BC_MAX_CONTEXT_LEN=65536
# Iter 0 + iter 1 (iter 1 is where the 9B TP=4 hit OOM on actor backward).
export BC_NUM_ROLLOUT=2

# Distinct RUN_NAME so wandb/log/ckpt don't collide.
export RUN_NAME="${RUN_NAME:-supo-bcplus-qwen3p5-4b-1node-debug-$(date +%Y%m%d-%H%M%S)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../run_qwen3p5_4B_colocate.sh" "$@"
