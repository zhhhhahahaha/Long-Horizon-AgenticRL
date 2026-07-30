#!/bin/bash
# This file is sourced by submit_experiment.sh.
# shellcheck disable=SC2034
# Resume the completed 4B n8 8-node groupfix run at 39 and train rollouts 40-99.

# Keep training below the CRITICAL search service in MAST scheduling priority.
MAST_JOB_NAME=supo_4b_n8_8n_resume39_to100_dump_groupfix_high
MAST_JOB_PRIORITY=HIGH
MAST_NUM_NODES=8
MAST_GPUS_PER_NODE=8
MAST_DATA_PARALLEL_SIZE=8
MAST_CONTEXT_PARALLEL_SIZE=8
MAST_CONDA_DOCKER_IMAGE=588845226011.dkr.ecr.us-east-2.amazonaws.com/msl_infra/slime:hhz-20260629a
MAST_CODE_ARCHIVE=/mnt/wsfuse/hhzhang01/supo-slime/slime-code-4b-8n-groupfix-resume39-to100-high-20260730.tgz

# Retain the original checkpoint, W&B, rollout dump, and Ray log namespace.
BC_RUN_NAME=supo_4b_n8_8n_40iter_dump_groupfix-mj1d0qw1
BC_MODEL_SIZE=4B
BC_NUM_ROLLOUT=100
BC_ROLLOUT_BATCH_SIZE=32
BC_N_SAMPLES=8
BC_GLOBAL_BATCH_SIZE=256
BC_MAX_RESPONSE_LEN=32768
BC_MAX_CONTEXT_LEN=65536

# Preserve the original 64-GPU topology: 64 / (TP 4 x CP 2) = DP 8.
BC_TP=4
BC_CP=2
BC_SGLANG_TP=2
BC_MAX_TOKENS_PER_GPU=49152
BC_SGLANG_MEM_FRACTION_STATIC=0.7

# Preserve the original static sampling and rollout dump behavior.
BCPLUS_DYNAMIC_SAMPLING=0
BCPLUS_SEARCH_CONCURRENCY=128
BCPLUS_JUDGE_CONCURRENCY=64
BC_SAVE_INTERVAL=5
BC_SLIM_INTERMEDIATE_CHECKPOINTS=1
# Extend the original 40-rollout scheduler horizon while retaining progress.
BC_OVERRIDE_OPT_PARAM_SCHEDULER=1
BC_DUMP_ROLLOUT=1
BCPLUS_DUMP_TRAIN_OLD=0
MAST_WANDB_SNAPSHOT_INTERVAL_SEC=60
WANDB_X_FLUSH_INTERVAL_SECONDS=30
