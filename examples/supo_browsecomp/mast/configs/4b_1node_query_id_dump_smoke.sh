#!/bin/bash
# This file is sourced by submit_experiment.sh.
# shellcheck disable=SC2034
# One-node, full-settings smoke that verifies query_id reaches the rollout dump.

MAST_JOB_NAME=supo_4b_n4_p2_1n_queryid_dump_smoke
MAST_TENANT=rhea_assistant_avocado_iterations
MAST_REGION=eag
MAST_HOST=grandteton_80g_roce
MAST_JOB_PRIORITY=HIGH
MAST_RETRIES=1
MAST_NUM_NODES=1
MAST_GPUS_PER_NODE=8
MAST_DATA_PARALLEL_SIZE=1
MAST_CONTEXT_PARALLEL_SIZE=8
MAST_CONDA_DOCKER_IMAGE=588845226011.dkr.ecr.us-east-2.amazonaws.com/msl_infra/slime:hhz-20260629a
MAST_CODE_ARCHIVE=/mnt/wsfuse/hhzhang01/supo-slime/slime-code-4b-n4-p2-1n-queryid-dump-smoke-20260803.tgz

# Two prompts with four rollouts each give two independent query ids and four
# repeated observations of each id in one optimizer step.
BC_TRAIN_DATA=/mnt/wsfuse/hhzhang01/supo-data/BC+/bc_train_exclude_stable91_20260730.parquet
SEARCH_ADDR_FILE=/mnt/wsfuse/hhzhang01/supo-slime/search-servers/bcplus-search-01.addr
BC_MODEL_SIZE=4B
BC_NUM_ROLLOUT=1
BC_ROLLOUT_BATCH_SIZE=2
BC_N_SAMPLES=4
BC_GLOBAL_BATCH_SIZE=8
BC_MAX_RESPONSE_LEN=32768
BC_MAX_CONTEXT_LEN=65536

# 8 GPUs / (Megatron TP 4 x CP 2) = train DP 1. SGLang TP 2 launches four
# colocated engines.
BC_TP=4
BC_CP=2
BC_SGLANG_TP=2
BC_MAX_TOKENS_PER_GPU=49152
BC_SGLANG_MEM_FRACTION_STATIC=0.7

# Match the full experiment's rollout and tool settings. Dynamic sampling is
# deliberately off so exactly two source prompts, rather than a larger
# candidate pool, reach generation and the dump.
BCPLUS_DYNAMIC_SAMPLING=0
BCPLUS_MAX_TURNS=64
BCPLUS_MAX_SUB_TRAJS=5
BCPLUS_COMPRESS_THRESH=0.85
BCPLUS_FIXED_SEARCH_TOPK=5
BCPLUS_DOC_WORDS_FULL=10000
BCPLUS_SEARCH_CONCURRENCY=512
BCPLUS_JUDGE_CONCURRENCY=128

# Exercise the parquet hook without checkpoint I/O or the extra train-old
# forward pass. The expected artifact is one rollouts_iter_00000_dp0.parquet
# containing two query ids and four distinct parent rollout ids per query.
BC_SAVE_INTERVAL=0
BC_SLIM_INTERMEDIATE_CHECKPOINTS=0
BC_DUMP_ROLLOUT=1
BCPLUS_DUMP_TRAIN_OLD=0
MAST_WANDB_SNAPSHOT_INTERVAL_SEC=30
WANDB_X_FLUSH_INTERVAL_SECONDS=30
