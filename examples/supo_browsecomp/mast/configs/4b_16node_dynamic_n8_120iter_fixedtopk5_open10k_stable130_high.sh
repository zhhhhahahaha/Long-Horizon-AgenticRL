#!/bin/bash
# This file is sourced by submit_experiment.sh.
# shellcheck disable=SC2034
# 4B, 16-node H100, n8 BC+ training with adaptive dynamic filtering,
# fixed-top5/open-10000 tool settings, and per-iteration trajectory dumps.
#
# Same shape as 4b_16node_dynamic_n8_120iter_fixedtopk5_open10000w_dump_high.sh.
# Two differences:
#   1. Training data excludes the 130 queries that stayed solved at base, ckpt4,
#      and ckpt9 of supo_4b_n8_16n_120step_dyn_topk5_open10k_dump_high-wncngd6s,
#      filtered under this same fixed5/open10k setting (550 rows, down from 589
#      under the older stable91 filter computed with a different setting).
#   2. W&B runs online to meta-3.wandb.io instead of offline+snapshot sync.
#      meta.wandb.io history ingest was hours behind as of 2026-08-05 (T283503334);
#      meta-3 is a separate deployment and returns rows in under 30s. Recovery
#      snapshots to OILFS are still published in online mode -- see wandb/README.md.

# MAST submission resources. The tenant resolves to:
# gen_ai/msl/tbd_research/rhea/msl_tbd_rhea_friends_data/
# rhea_assistant/rhea_assistant_avocado_iterations
MAST_JOB_NAME=supo_4b_n8_16n_120step_stable130_dump_high
MAST_TENANT=rhea_assistant_avocado_iterations
MAST_REGION=eag
MAST_HOST=grandteton_80g_roce
MAST_JOB_PRIORITY=HIGH
MAST_NUM_NODES=16
MAST_GPUS_PER_NODE=8
MAST_DATA_PARALLEL_SIZE=16
MAST_CONTEXT_PARALLEL_SIZE=8
MAST_CONDA_DOCKER_IMAGE=588845226011.dkr.ecr.us-east-2.amazonaws.com/msl_infra/slime:hhz-20260629a
MAST_CODE_ARCHIVE=/mnt/wsfuse/hhzhang01/supo-slime/slime-code-4b-n8-16n-120step-stable130-20260805.tgz

# Fresh training run: BC_NUM_ROLLOUT=120 executes rollout steps 0-119.
BC_MODEL_SIZE=4B
BC_NUM_ROLLOUT=120
BC_ROLLOUT_BATCH_SIZE=32
BC_N_SAMPLES=8
BC_GLOBAL_BATCH_SIZE=256
BC_MAX_RESPONSE_LEN=32768
BC_MAX_CONTEXT_LEN=65536
BC_TRAIN_DATA=/mnt/wsfuse/hhzhang01/supo-data/BC+/bc_train_exclude_stable130_fixed5_open10k_20260804.parquet
SEARCH_ADDR_FILE=/mnt/wsfuse/hhzhang01/supo-slime/search-servers/bcplus-search-01.addr

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

# Dump every selected training trajectory, without the extra train_old
# log-probability forward pass.
BC_DUMP_ROLLOUT=1
BCPLUS_DUMP_TRAIN_OLD=0

# Online W&B to meta-3, with OILFS recovery snapshots retained in both modes.
BC_WANDB_MODE=online
BC_WANDB_ENTITY=hhzhang01
BC_WANDB_PROJECT=supo-bcplus-mast
BC_WANDB_HOST=https://meta-3.wandb.io
MAST_WANDB_HTTPS_PROXY=http://fwdproxy:8080
MAST_WANDB_SNAPSHOT_INTERVAL_SEC=60
MAST_WANDB_SNAPSHOT_KEEP=3
WANDB_X_FLUSH_INTERVAL_SECONDS=30
