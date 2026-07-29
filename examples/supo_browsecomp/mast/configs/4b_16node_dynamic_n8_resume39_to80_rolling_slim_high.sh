#!/bin/bash
# This file is sourced by submit_experiment.sh.
# shellcheck disable=SC2034
# Resume the completed 4B n8 16-node run from checkpoint 39 and train rollouts 40-79.

# Keep training below the CRITICAL search service in MAST scheduling priority.
MAST_JOB_NAME=supo_4b_n8_16n_resume39_to80_rolling_slim_v4_high
MAST_JOB_PRIORITY=HIGH
MAST_NUM_NODES=16
MAST_GPUS_PER_NODE=8
MAST_DATA_PARALLEL_SIZE=16
MAST_CONTEXT_PARALLEL_SIZE=8
MAST_CONDA_DOCKER_IMAGE=588845226011.dkr.ecr.us-east-2.amazonaws.com/msl_infra/slime:hhz-20260629a
# Reuse the immutable, already-validated archive from the v3 resume attempt.
MAST_CODE_ARCHIVE=/mnt/wsfuse/hhzhang01/supo-slime/slime-code-4b-resume39-to80-rolling-slim-v3-20260728.tgz

# Deliberately retain the original checkpoint/W&B/log namespace so Megatron
# resumes the logical run instead of starting from iteration 0.
BC_RUN_NAME=supo_4b_n8_16n_40iter_dynamic-pvz5m0d5
BC_MODEL_SIZE=4B
BC_NUM_ROLLOUT=80
BC_ROLLOUT_BATCH_SIZE=32
BC_N_SAMPLES=8
BC_GLOBAL_BATCH_SIZE=256
BC_MAX_RESPONSE_LEN=32768
BC_MAX_CONTEXT_LEN=65536

# Preserve the original 128-GPU topology: 128 / (TP 4 x CP 2) = DP 16.
BC_TP=4
BC_CP=2
BC_SGLANG_TP=2
BC_MAX_TOKENS_PER_GPU=49152

BCPLUS_DYNAMIC_SAMPLING=1
BCPLUS_SEARCH_CONCURRENCY=512
BCPLUS_JUDGE_CONCURRENCY=128
BC_SAVE_INTERVAL=5
BC_SLIM_INTERMEDIATE_CHECKPOINTS=1
# Extend the original 40-rollout scheduler horizon while retaining progress.
BC_OVERRIDE_OPT_PARAM_SCHEDULER=1
BC_DUMP_ROLLOUT=0
MAST_WANDB_SNAPSHOT_INTERVAL_SEC=60
WANDB_X_FLUSH_INTERVAL_SECONDS=30
