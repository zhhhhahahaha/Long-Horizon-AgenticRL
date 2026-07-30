#!/bin/bash
# This file is sourced by submit_experiment.sh.
# shellcheck disable=SC2034
# Resume the completed 4B n16 8-node run from checkpoint 39 for 60 rollouts.

MAST_JOB_NAME=supo_4b_n16_8n_resume39_to100_rolling_slim_high
MAST_JOB_PRIORITY=HIGH
MAST_NUM_NODES=8
MAST_GPUS_PER_NODE=8
MAST_DATA_PARALLEL_SIZE=8
MAST_CONTEXT_PARALLEL_SIZE=8
MAST_CONDA_DOCKER_IMAGE=588845226011.dkr.ecr.us-east-2.amazonaws.com/msl_infra/slime:hhz-20260629a
MAST_CODE_ARCHIVE=/mnt/wsfuse/hhzhang01/supo-slime/slime-code-4b-n16-8n-resume39-to100-high-20260730.tgz

# Keep the original checkpoint, W&B, and Ray-log namespace for the resume.
BC_RUN_NAME=supo_4b_n16_8n_40iter-bt6g4q5g
BC_MODEL_SIZE=4B
# The tracker is 39; exclusive upper bound 100 runs rollouts 40-99.
BC_NUM_ROLLOUT=100
BC_ROLLOUT_BATCH_SIZE=32
BC_N_SAMPLES=16
BC_GLOBAL_BATCH_SIZE=512
BC_MAX_RESPONSE_LEN=32768
BC_MAX_CONTEXT_LEN=65536

# Preserve the original 64-GPU topology: 64 / (TP 4 x CP 2) = train DP 8.
BC_TP=4
BC_CP=2
BC_SGLANG_TP=2
BC_MAX_TOKENS_PER_GPU=49152

BC_SAVE_INTERVAL=5
BC_SLIM_INTERMEDIATE_CHECKPOINTS=1
# Extend the scheduler horizon from 40 to 100 while retaining its progress.
BC_OVERRIDE_OPT_PARAM_SCHEDULER=1
BC_DUMP_ROLLOUT=0
MAST_WANDB_SNAPSHOT_INTERVAL_SEC=60
WANDB_X_FLUSH_INTERVAL_SECONDS=30
