#!/bin/bash
# This file is sourced by submit_experiment.sh.
# shellcheck disable=SC2034
# 4B, 16-node colocated SUPO training with adaptive dynamic sampling.

# MAST submission resources. The submission wrapper verifies that these produce
# taskCount=16 and ROLE_ASSIGNMENT_MAP=trainer_0=128 before it can submit.
MAST_JOB_NAME=supo_4b_n16_16n_40iter_dynamic
MAST_NUM_NODES=16
MAST_GPUS_PER_NODE=8
MAST_DATA_PARALLEL_SIZE=16
MAST_CONTEXT_PARALLEL_SIZE=8
MAST_CONDA_DOCKER_IMAGE=588845226011.dkr.ecr.us-east-2.amazonaws.com/msl_infra/slime:hhz-20260629a
MAST_CODE_ARCHIVE=/mnt/wsfuse/hhzhang01/supo-slime/slime-code.tgz

# Training shape. Dynamic sampling keeps 32 groups x 16 responses for each
# update; its first candidate pool is 64 groups x 16 responses.
BC_MODEL_SIZE=4B
BC_NUM_ROLLOUT=40
BC_ROLLOUT_BATCH_SIZE=32
BC_N_SAMPLES=16
BC_GLOBAL_BATCH_SIZE=512
BC_MAX_RESPONSE_LEN=32768
BC_MAX_CONTEXT_LEN=65536

# 4B Megatron/SGLang topology.
BC_TP=4
BC_CP=2
BC_SGLANG_TP=2
BC_MAX_TOKENS_PER_GPU=49152

# Experiment behavior and service concurrency.
BCPLUS_DYNAMIC_SAMPLING=1
BCPLUS_SEARCH_CONCURRENCY=512
BCPLUS_JUDGE_CONCURRENCY=128
BC_SAVE_INTERVAL=5
BC_DUMP_ROLLOUT=0
MAST_WANDB_SNAPSHOT_INTERVAL_SEC=60
WANDB_X_FLUSH_INTERVAL_SECONDS=30
