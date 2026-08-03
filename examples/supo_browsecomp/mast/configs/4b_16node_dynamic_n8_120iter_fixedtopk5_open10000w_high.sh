#!/bin/bash
# This file is sourced by submit_experiment.sh.
# shellcheck disable=SC2034
# 4B, 16-node H100, n8 BC+ training with adaptive dynamic filtering and
# fixed-top5/open-10000 tool settings.

# MAST submission resources. The tenant resolves to:
# gen_ai/msl/tbd_research/rhea/msl_tbd_rhea_friends_data/
# rhea_assistant/rhea_assistant_avocado_iterations
MAST_JOB_NAME=supo_4b_n8_16n_120iter_dynamic_fixedtopk5_open10000w_h100_high
MAST_TENANT=rhea_assistant_avocado_iterations
MAST_REGION=eag
MAST_HOST=grandteton_80g_roce
MAST_JOB_PRIORITY=HIGH
MAST_NUM_NODES=16
MAST_GPUS_PER_NODE=8
MAST_DATA_PARALLEL_SIZE=16
MAST_CONTEXT_PARALLEL_SIZE=8
MAST_CONDA_DOCKER_IMAGE=588845226011.dkr.ecr.us-east-2.amazonaws.com/msl_infra/slime:hhz-20260629a
MAST_CODE_ARCHIVE=/mnt/wsfuse/hhzhang01/supo-slime/slime-code-4b-n8-16n-120iter-dynamic-fixedtopk5-open10000w-h100-high-20260803.tgz

# Fresh training run: BC_NUM_ROLLOUT=120 executes rollout steps 0-119.
BC_MODEL_SIZE=4B
BC_NUM_ROLLOUT=120
BC_ROLLOUT_BATCH_SIZE=32
BC_N_SAMPLES=8
BC_GLOBAL_BATCH_SIZE=256
BC_MAX_RESPONSE_LEN=32768
BC_MAX_CONTEXT_LEN=65536
BC_TRAIN_DATA=/mnt/wsfuse/hhzhang01/supo-data/BC+/bc_train_exclude_stable91_20260730.parquet

# 128 GPUs / (Megatron TP 4 x CP 2) = train DP 16. SGLang TP 2 gives
# 64 colocated rollout engines.
BC_TP=4
BC_CP=2
BC_SGLANG_TP=2
BC_MAX_TOKENS_PER_GPU=49152

# Adaptive dynamic filtering starts from 64 groups and selects 32 n8 groups,
# prioritizing nonzero reward variance before its bounded zero-std fallback.
BCPLUS_DYNAMIC_SAMPLING=1
BCPLUS_FIXED_SEARCH_TOPK=5
BCPLUS_DOC_WORDS_FULL=10000
BCPLUS_SEARCH_CONCURRENCY=512
BCPLUS_JUDGE_CONCURRENCY=128

BC_SAVE_INTERVAL=5
BC_SLIM_INTERMEDIATE_CHECKPOINTS=1
BC_DUMP_ROLLOUT=0
MAST_WANDB_SNAPSHOT_INTERVAL_SEC=60
WANDB_X_FLUSH_INTERVAL_SECONDS=30
