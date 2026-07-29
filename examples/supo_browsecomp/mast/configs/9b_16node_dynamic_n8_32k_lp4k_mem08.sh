#!/bin/bash
# This file is sourced by submit_experiment.sh.
# shellcheck disable=SC2034
# 9B, 16-node colocated SUPO training with group size 8 and dynamic sampling.

# MAST submission resources. The submission wrapper verifies that these produce
# taskCount=16 and ROLE_ASSIGNMENT_MAP=trainer_0=128 before it can submit.
MAST_JOB_NAME=supo_9b_n8_16n_40iter_dynamic_32k_lp4k_mem08
MAST_NUM_NODES=16
MAST_GPUS_PER_NODE=8
MAST_DATA_PARALLEL_SIZE=16
MAST_CONTEXT_PARALLEL_SIZE=8
MAST_CONDA_DOCKER_IMAGE=588845226011.dkr.ecr.us-east-2.amazonaws.com/msl_infra/slime:hhz-20260629a
MAST_CODE_ARCHIVE=/mnt/wsfuse/hhzhang01/supo-slime/slime-code-9b-32k-lp4k-mem08.tgz

# Keep the 4B n8 experiment shape: each update selects 32 groups x 8 responses,
# and dynamic sampling starts from 64 groups x 8 responses.
BC_MODEL_SIZE=9B
BC_NUM_ROLLOUT=40
BC_ROLLOUT_BATCH_SIZE=32
BC_N_SAMPLES=8
BC_GLOBAL_BATCH_SIZE=256
BC_MAX_RESPONSE_LEN=32768
BC_MAX_CONTEXT_LEN=65536

# 9B training topology: 128 GPUs / (TP 4 x CP 2) = DP 16. SGLang uses
# TP 2 so the colocated rollout phase runs 64 engines across the cluster.
BC_TP=4
BC_CP=2
BC_SGLANG_TP=2
BC_MAX_TOKENS_PER_GPU=32768
BC_LOG_PROBS_CHUNK_SIZE=4096
BC_SGLANG_MEM_FRACTION_STATIC=0.8

# Dynamic first-pool concurrency and judge API throttle.
BCPLUS_DYNAMIC_SAMPLING=1
BCPLUS_SEARCH_CONCURRENCY=512
BCPLUS_JUDGE_CONCURRENCY=128
BC_SAVE_INTERVAL=5
BC_SLIM_INTERMEDIATE_CHECKPOINTS=1
BC_DUMP_ROLLOUT=0
MAST_WANDB_SNAPSHOT_INTERVAL_SEC=60
WANDB_X_FLUSH_INTERVAL_SECONDS=30
