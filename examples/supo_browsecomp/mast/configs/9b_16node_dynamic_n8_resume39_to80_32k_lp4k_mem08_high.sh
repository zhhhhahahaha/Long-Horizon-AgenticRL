#!/bin/bash
# This file is sourced by submit_experiment.sh.
# shellcheck disable=SC2034
# Resume the completed 9B n8 run from checkpoint 39 and train rollouts 40-79.

# This is a new MAST submission, while BC_RUN_NAME deliberately retains the
# original checkpoint, W&B, and ray-log namespace for the logical training run.
MAST_JOB_NAME=supo_9b_n8_16n_resume39_to80_32k_lp4k_mem08_high
MAST_JOB_PRIORITY=HIGH
MAST_NUM_NODES=16
MAST_GPUS_PER_NODE=8
MAST_DATA_PARALLEL_SIZE=16
MAST_CONTEXT_PARALLEL_SIZE=8
MAST_CONDA_DOCKER_IMAGE=588845226011.dkr.ecr.us-east-2.amazonaws.com/msl_infra/slime:hhz-20260629a
MAST_CODE_ARCHIVE=/mnt/wsfuse/hhzhang01/supo-slime/slime-code-9b-resume39-to80-32k-lp4k-mem08-slim-high-20260729.tgz

BC_RUN_NAME=supo_9b_n8_16n_40iter_dynamic_32k_lp4k_mem08-c77ftk5w
BC_MODEL_SIZE=9B
BC_NUM_ROLLOUT=80
BC_ROLLOUT_BATCH_SIZE=32
BC_N_SAMPLES=8
BC_GLOBAL_BATCH_SIZE=256
BC_MAX_RESPONSE_LEN=32768
BC_MAX_CONTEXT_LEN=65536

# Preserve the original 128-GPU topology: TP 4 x CP 2 gives DP 16, and
# SGLang TP 2 provides 64 colocated rollout engines.
BC_TP=4
BC_CP=2
BC_SGLANG_TP=2
BC_MAX_TOKENS_PER_GPU=32768
BC_LOG_PROBS_CHUNK_SIZE=4096
BC_SGLANG_MEM_FRACTION_STATIC=0.8

BCPLUS_DYNAMIC_SAMPLING=1
BCPLUS_SEARCH_CONCURRENCY=512
BCPLUS_JUDGE_CONCURRENCY=128
BC_SAVE_INTERVAL=5
BC_SLIM_INTERMEDIATE_CHECKPOINTS=1
# The original scheduler horizon ended at 40 rollouts. Adopt the new horizon
# while retaining optimizer state and the scheduler progress stored at step 39.
BC_OVERRIDE_OPT_PARAM_SCHEDULER=1
BC_DUMP_ROLLOUT=0
MAST_WANDB_SNAPSHOT_INTERVAL_SEC=60
WANDB_X_FLUSH_INTERVAL_SECONDS=30
