# 在 MAST 上新建或 Resume BrowseComp/SUPO 训练任务

这是一份给后续 Codex session 使用的执行手册。用户要求在 MAST 上启动新的或
resume 已有的 BrowseComp/SUPO 训练实验时，应完整阅读本文件，然后按顺序完成
checkpoint 核对、配置、归档、dry-run、真实提交和状态核对。不要只给出命令而不
执行，除非用户明确只要方案。

## 执行原则

- 用户给出的模型、节点数、batch shape、并行度和性能参数优先于模板默认值。
- 保留工作区内已有的用户改动；先运行 `git status --short`，不要 reset、checkout
  或覆盖无关文件。
- 每个新实验新建一个 config，不要修改已经提交过的 config。
- 每个实验使用独立、描述性的代码归档名，避免后续刷新共享 archive 影响 MAST
  retry 或其他正在运行的任务。
- 真实提交只执行一次。CLI 暂时没有输出、JSON 文件暂时为 0 字节或工具调用提前
  yield，都不代表失败；只要提交进程仍在，就继续等待，绝不能再发一份。
- 不打印 `.llama_key`、W&B key 或其他 secret 的内容，只检查文件是否存在且非空。
- 完成后向用户报告 full MAST job name、MAST URL、当前状态、config 路径以及关键参数。

## 入口和文件关系

- `configs/*.sh`：单个实验的配置，只放实验和资源参数。
- `submit_experiment.sh`：读取 config，先做 MAST dry-run，校验 task count 和 rank
  map，再调用真实提交器。
- `wandb/submit_with_wandb.sh`：执行真实提交、保存结构化响应，并启动 W&B watcher。
- `run_trainer.sh`：在 MAST 容器中运行，选择模型、组装 slime 参数并启动 Ray 训练。
- `search/README.md`：提交、替换和验证独立 search server；训练提交前必须确认其真实
  `/search` 请求可用。
- `wandb/wandb_sync.sh`：在 devserver 的 tmux 中追踪 MAST 状态并同步离线 W&B 数据。

正常情况下不要绕过 `submit_experiment.sh` 手写长 MAST 命令。

## 1. 理解需求并选择最近的模板

在仓库根目录开始：

```bash
cd /home/hhzhang01/Long-Horizon-AgenticRL
git status --short
ls -1 examples/supo_browsecomp/mast/configs
```

当前可参考的模板包括：

- `configs/4b_16node_dynamic_n8.sh`：4B、16 nodes、每组 8 samples。
- `configs/4b_16node_dynamic_n16.sh`：4B、16 nodes、每组 16 samples。
- `configs/9b_16node_dynamic_n8_32k_lp4k_mem08.sh`：9B、16 nodes、n8、
  SGLang TP=2、32K tokens/GPU、log-probs chunk 4096、SGLang memory fraction 0.8。

选择 shape 和模型最接近的文件作为参考，但使用 `apply_patch` 新建一个新的 config。
文件名和 `MAST_JOB_NAME` 应能区分模型、group size、节点数和关键实验差异。例如：

```text
configs/9b_16node_dynamic_n8_<change>.sh
MAST_JOB_NAME=supo_9b_n8_16n_40iter_dynamic_<change>
```

`MAST_JOB_NAME` 只使用字母、数字、点、下划线和连字符。

在提交前检查本机是否已有同一基础名称的提交记录：

```bash
base_job_name=<MAST_JOB_NAME-from-config>
find /home/hhzhang01/.local/state/mast-wandb \
  -mindepth 1 -maxdepth 1 -type d -name "${base_job_name}-*" -print
```

如果有结果，先读取其中的 submit JSON 和当前 MAST 状态，确认用户确实要开一个新实验；
需要新实验时应换一个可区分的 `MAST_JOB_NAME`，不要无意中重复已有任务。

## 2. 填写并检查实验配置

一个 16-node config 的核心结构如下。数值只是示例，必须按用户需求调整：

```bash
#!/bin/bash
# This file is sourced by submit_experiment.sh.
# shellcheck disable=SC2034

MAST_JOB_NAME=supo_9b_n8_16n_40iter_dynamic_fixedtopk5_open10000w
MAST_TENANT=rhea_assistant_avocado_iterations
MAST_REGION=eag
MAST_HOST=grandteton_80g_roce
MAST_JOB_PRIORITY=HIGH
MAST_NUM_NODES=16
MAST_GPUS_PER_NODE=8
MAST_DATA_PARALLEL_SIZE=16
MAST_CONTEXT_PARALLEL_SIZE=8
MAST_CONDA_DOCKER_IMAGE=588845226011.dkr.ecr.us-east-2.amazonaws.com/msl_infra/slime:hhz-20260629a
MAST_CODE_ARCHIVE=/mnt/wsfuse/hhzhang01/supo-slime/slime-code-9b-example.tgz

BC_MODEL_SIZE=9B
BC_NUM_ROLLOUT=40
BC_ROLLOUT_BATCH_SIZE=32
BC_N_SAMPLES=8
BC_GLOBAL_BATCH_SIZE=256
BC_MAX_RESPONSE_LEN=32768
BC_MAX_CONTEXT_LEN=65536

BC_TP=4
BC_CP=2
BC_SGLANG_TP=2
BC_MAX_TOKENS_PER_GPU=32768
BC_LOG_PROBS_CHUNK_SIZE=4096
BC_SGLANG_MEM_FRACTION_STATIC=0.8

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
```

每个新 config 都必须显式设置 `MAST_REGION` 和 `MAST_HOST`，不能依赖
`submit_experiment.sh` 的默认 `nha` / `zionex_80g`。硬件选择会直接改变 GPU
代际、网络和 tenant capacity。例如：

| Intended hardware | Config | Dry-run capability |
|---|---|---|
| EAG Grand Teton H100 RoCE | `MAST_REGION=eag`, `MAST_HOST=grandteton_80g_roce` | `LogicalServerSubType.T20_GRAND_TETON_HBM3_ROCE` |
| NHA ZionEX A100 80GB | `MAST_REGION=nha`, `MAST_HOST=zionex_80g` | `LogicalServerSubType.T20_ZION_EX_A100_80GB` |

对于 tenant reservation，还要用 `mast get-capacity` 核对 capability 对应的实际
machine type。例如 `rhea_assistant_avocado_iterations` 的 EAG Grand Teton H100
capacity 是 `MACHINE_TYPE_T20_GRAND_TETON_HBM3_ROCE_GENAI`。只看到 8 GPUs/task
不足以证明硬件正确。

新训练默认使用
`/mnt/wsfuse/hhzhang01/supo-data/BC+/bc_train_exclude_stable91_20260730.parquet`
（589 条，排除了三个 model point 均 8/8 成功的 91 条）。需要覆盖时，在 config
中设置容器内绝对路径，例如恢复使用原始 680 条数据：

```bash
BC_TRAIN_DATA=/mnt/wsfuse/hhzhang01/supo-data/BC+/bc_train.parquet
```

Resume 旧 run 时必须沿用该 run 原先的数据集；如果旧 run 使用原始 680 条数据，需显式
设置上面的 `BC_TRAIN_DATA`，不要依赖新默认值。

检查以下不变量：

1. MAST 分配的 ranks 必须一致：
   `MAST_DATA_PARALLEL_SIZE * MAST_CONTEXT_PARALLEL_SIZE == MAST_NUM_NODES * MAST_GPUS_PER_NODE`。
   16 nodes、每节点 8 GPU 时应为 128。
2. 总 GPU 数必须能整除训练模型并行度：
   `TOTAL_GPUS % (BC_TP * BC_CP) == 0`。训练 DP 为
   `TOTAL_GPUS / (BC_TP * BC_CP)`。
3. 总 GPU 数必须能整除 `BC_SGLANG_TP`。rollout engine 数约为
   `TOTAL_GPUS / BC_SGLANG_TP`。
4. 当前 n8/n16 模板满足
   `BC_GLOBAL_BATCH_SIZE == BC_ROLLOUT_BATCH_SIZE * BC_N_SAMPLES`。
5. dynamic sampling 的 first pool group 数由 runner 设置为
   `2 * BC_ROLLOUT_BATCH_SIZE`；不要把 search concurrency 误当成轨迹总数。
6. 9B 当前验证过的训练拓扑是 TP=4/CP=2。SGLang TP 应按用户要求显式设置；
   TP=2 在 128 GPU 上产生 64 个 engines，TP=4 则产生 32 个 engines 并提供更多
   单 engine 显存余量。
7. 新的 fixed-budget 协议使用 `BCPLUS_FIXED_SEARCH_TOPK=5` 和
   `BCPLUS_DOC_WORDS_FULL=10000`。fixed top-k 沿用 SUPO 加权预算，已访问结果仍按
   `0.25` 计数。Resume 已有 logical run 时必须保持原值，不能在中途切换工具 schema。

如果用户要求的是 runner 尚不支持的新 `BC_*` 参数，需要同时：

1. 在 `submit_experiment.sh` 的 `TRAIN_ENV_VARS` 中加入该变量；
2. 在 `run_trainer.sh` 中读取、校验并传给准确的 slime CLI flag；
3. 保持旧实验的默认行为不变；
4. 在 dry-run JSON 的最终 container entrypoint 中确认变量确实被透传。

不要仅仅把变量写入 config；没有进入 `TRAIN_ENV_VARS` 的变量不会进入 MAST 容器。

## 3. Resume 已有训练 run 的完整流程

新 MAST submission 会获得新的 `MAST_HPC_JOB_NAME` 随机后缀，不能依靠它自动找到旧
checkpoint。Resume 必须使用一个新的 submission config，同时通过 `BC_RUN_NAME`
显式指向原始 checkpoint 目录。不要修改或重新提交原始 config。

> **Resume 身份不变量**：`MAST_JOB_NAME` 标识这一次新的调度任务，应使用新名称；
> `BC_RUN_NAME` 标识旧 checkpoint、dataset state 和 W&B logical group，必须逐字保留
> 原始 full run name（包括随机 suffix）。改变 `MAST_JOB_NAME` 不会破坏 resume，改变
> `BC_RUN_NAME` 才会让任务找不到原训练状态。

### 3.1 确认旧任务已经停止

不能在旧任务仍可能写 checkpoint 时启动 resume 或离线瘦身。先找到旧的 full MAST
job name，并确认它不是 `RUNNING`、`PENDING` 或正在 retry：

```bash
old_job_name=<full-MAST-job-name-with-suffix>
with-proxy mast --output json get-status "${old_job_name}"
```

如果无法确认旧任务状态，停止操作并询问用户；不要让两个任务写同一个 checkpoint
根目录。

### 3.2 只读核对 checkpoint 根目录

`BC_RUN_NAME` 是 `checkpoints/` 下面的目录 basename，不是路径，也不是新的
`MAST_JOB_NAME`。例如：

```bash
checkpoint_base=/data/users/hhzhang01/wsfuse_mnt/hhzhang01/supo-slime/checkpoints
resume_run=supo_9b_n8_16n_40iter_dynamic_32k_lp4k_mem08-c77ftk5w
resume_root="${checkpoint_base}/${resume_run}"

test -d "${resume_root}"
test -f "${resume_root}/latest_checkpointed_iteration.txt"
resume_step="$(tr -d '[:space:]' < "${resume_root}/latest_checkpointed_iteration.txt")"
[[ "${resume_step}" =~ ^[0-9]+$ ]]
test -f "${resume_root}/iter_$(printf '%07d' "${resume_step}")/.metadata"
test -f "${resume_root}/iter_$(printf '%07d' "${resume_step}")/common.pt"
printf 'resume_run=%s resume_step=%s\n' "${resume_run}" "${resume_step}"
```

再运行只读计划检查，确认 `protected_step` 等于 tracker，且没有更大的半写入目录：

```bash
python examples/supo_browsecomp/mast/checkpoint_slim/checkpoint_slim.py plan \
  --checkpoint-root "${checkpoint_base}" \
  --run "${resume_run}"
```

latest checkpoint 必须是保留 optimizer、scheduler 和 RNG 的 full checkpoint。不要从
已经瘦身的中间 checkpoint resume，也不要手动修改 tracker 指向中间 checkpoint。

### 3.3 正确理解 `BC_NUM_ROLLOUT`

checkpoint iteration 是从 0 开始的 rollout id，`BC_NUM_ROLLOUT` 是循环的 exclusive
upper bound：

- tracker 为 29、`BC_NUM_ROLLOUT=40` 时，resume 会执行 rollout 30–39；
- tracker 为 39、`BC_NUM_ROLLOUT=40` 时，不会再训练；
- tracker 为 39、`BC_NUM_ROLLOUT=41` 时，只会执行 rollout 40。

因此，要让 resume 至少执行一个新 rollout，必须确认
`BC_NUM_ROLLOUT > resume_step + 1`。仅满足 `BC_NUM_ROLLOUT > resume_step` 时，可能像
tracker 39、终点 40 一样直接结束而不训练。对于中断任务，通常保持原计划的
`BC_NUM_ROLLOUT`；只有用户明确要求延长已完成训练时才增大它，并同时核对 LR scheduler
等训练终点相关配置。

Megatron 会把 scheduler 的总 samples 数写入 full checkpoint。仅把原计划从 40 rollout
改为 80 rollout 时，新旧总数不同，默认 resume 会在加载 optimizer scheduler 时退出，
典型报错是：

```text
OptimizerParamScheduler: class input value 20480 and checkpointvalue 10240
for total number of iterations do not match
```

这不是 checkpoint 损坏。如果目标确实是延长训练，并且已经确认新 config 中的 LR、weight
decay、warmup 等 scheduler 设置符合预期，在 resume config 中显式设置：

```bash
BC_OVERRIDE_OPT_PARAM_SCHEDULER=1
```

Runner 会只在实际发现 checkpoint 并加入 `--load` 时传递 Megatron 的
`--override-opt-param-scheduler`。它保留 checkpoint 中已经完成的 scheduler 进度和 optimizer
state，但采用新 config 的 scheduler 总长度与参数。普通中断恢复、且训练终点未改变时不要
设置这个开关；也不要用它掩盖意外的 optimizer/scheduler 配置差异。

### 3.4 新建 resume 专用 config

以原实验 config 为基准，通过 `apply_patch` 新建一个 config。保持模型、TP/CP、batch
shape、optimizer 和数据设置与原实验一致，只修改明确需要变化的字段：

```bash
# 新 submission 的基础名称；必须与旧任务区分。
MAST_JOB_NAME=supo_9b_n8_16n_40iter_dynamic_32k_lp4k_mem08_resume29

# 新代码归档；不能继续引用旧任务的 archive。
MAST_CODE_ARCHIVE=/mnt/wsfuse/hhzhang01/supo-slime/slime-code-9b-resume29-rolling-slim.tgz

# 原始 checkpoint 目录 basename；必须保留旧任务的随机 suffix。
BC_RUN_NAME=supo_9b_n8_16n_40iter_dynamic_32k_lp4k_mem08-c77ftk5w

# 训练总终点，不是“再训练多少步”。
BC_NUM_ROLLOUT=40

# 只有在延长原训练终点、并核对过 scheduler 设置后才启用。
BC_OVERRIDE_OPT_PARAM_SCHEDULER=1

# 需要训练时滚动瘦身才设置；latest checkpoint 始终保持 full。
BC_SLIM_INTERMEDIATE_CHECKPOINTS=1
```

这里两个名字用途不同：

- `MAST_JOB_NAME`：这一次新 MAST submission 的名称，会获得新的随机 suffix；
- `BC_RUN_NAME`：旧 checkpoint、W&B group、ray log 等持久化目录使用的原始名称。

Submit wrapper 会自动把 `BC_RUN_NAME` 映射为 watcher 的
`MAST_WANDB_RUN_NAME`。Watcher 用新的 full MAST job name 查询任务状态，但从旧 logical
run name 下读取 W&B snapshots；两者不能混为同一个变量。

Runner 会要求
`checkpoints/${BC_RUN_NAME}/latest_checkpointed_iteration.txt` 存在且内容为整数。名称写错
时会在启动 Ray 前退出，不会创建新目录从头训练。全新实验不要设置 `BC_RUN_NAME`。

### 3.5 用新代码重新归档

Resume 不能复用旧实验 archive，因为旧 archive 不会自动包含当前 checkout 的滚动瘦身
和 `BC_RUN_NAME` 支持。按照第 5 节创建一个新的、不可变的 archive，并额外确认：

```bash
tar tzf "${stage_target}" | rg '^\./slime/utils/checkpoint_rotation\.py$'
tar xOzf "${stage_target}" \
  ./examples/supo_browsecomp/mast/run_trainer.sh | \
  rg 'BC_RUN_NAME|slim-intermediate-checkpoints'
tar xOzf "${stage_target}" ./<resume-config> | \
  rg 'MAST_JOB_NAME|MAST_CODE_ARCHIVE|BC_RUN_NAME|BC_NUM_ROLLOUT|BC_OVERRIDE_OPT_PARAM_SCHEDULER|BC_SLIM_INTERMEDIATE_CHECKPOINTS'
```

### 3.6 Dry-run 和真实提交

先按第 6 节运行 dry-run。除了 nodes/ranks，还必须确认最终 container entrypoint 包含准确
的 `BC_RUN_NAME`、新 archive 和训练终点：

```bash
jq -r '.spec.app_def.roles[] |
  select(.name=="trainer_0") | .entrypoint' "${dryrun_file}" | \
  rg 'BC_RUN_NAME|BC_NUM_ROLLOUT|BC_OVERRIDE_OPT_PARAM_SCHEDULER|BC_SLIM_INTERMEDIATE_CHECKPOINTS|slime-code-.*resume'
```

确认无误后按第 7 节只提交一次。

### 3.7 启动后必须看到的证据

不要仅凭 MAST 状态为 `RUNNING` 就认为 resume 成功。Trainer/Ray 日志至少应出现：

```text
[trainer] explicit resume run=<BC_RUN_NAME> tracker=<resume_step>
[head] resuming from /mnt/wsfuse/hhzhang01/supo-slime/checkpoints/<BC_RUN_NAME>
[head] resume will override checkpoint optimizer scheduler configuration  # 仅延长终点时
```

16 个 task 可能先用数分钟完成 GPU CDI、ECR 登录和 Docker image pull；stdout 暂时停在
`[docker] Pulling image ...` 不代表失败。先确认所有 task 仍为 `RUNNING` 且 restart 为 0，
然后读取 trainer task 0：

```bash
full_job_name=<new-full-MAST-job-name>
with-proxy mast --output json get-status "${full_job_name}" | \
  jq '{state:.data.state,restarts:.data.numRestarts,attempt:.data.latestAttempt.state}'
with-proxy mast get-logs --file-path stdout \
  --twjob '.*trainer_0.*/0' \
  --regex 'explicit resume|resuming from|successfully loaded checkpoint|rolling checkpoint|ERROR' \
  "${full_job_name}"
```

Image pull 或 trainer 初始化仍在进行时只等待并继续查询同一个 job，绝不能重新提交。

MAST task-group retry 会保留同一个 job name，但产生新的
`MAST_HPC_TASK_GROUP_ATTEMPT_EPOCH`。Runner 按 attempt 隔离 `ray-coord` 目录，并给 worker
的 `ray start` 加了 90 秒 timeout；这避免早启动的 worker 读取上一 attempt 的 `head.ip`
后永久等待旧 GCS。排查多节点 retry 时可查看：

```bash
with-proxy mast get-logs --file-path stdout \
  --twjob '.*trainer_0.*' \
  --regex 'joining ray|joined ray|join failed|Waiting for placement group' \
  "${full_job_name}"
```

如果 placement group 持续报告少于预期 GPU，先确认各 worker 是否正在 timeout 后用当前
`head.ip` 重试；不要启动另一个会写同一 checkpoint 根目录的 job。SGLang 并发初始化偶尔
也会遇到本机随机 ZMQ 端口冲突（`Address already in use`），这种发生在 checkpoint load
之前的瞬时错误应交给同一 MAST job 的 task-group retry 处理。

随后确认 Megatron 报告从同一个 iteration 成功加载，而不是从 HF/reference checkpoint 的
iteration 0 初始化。启用滚动瘦身时，旧代码产生的 latest full checkpoint 会先生成
`resume bootstrap` slim staging；到下一个 full checkpoint 成功后才替换它。更早的历史
full checkpoints 不会在 resume 时自动瘦身，需要的话使用
[`checkpoint_slim/README.md`](checkpoint_slim/README.md) 的离线流程。

如果日志中的 run 名、tracker 或加载 iteration 不一致，立即停止新任务并检查配置；不要
通过修改 tracker 强行继续。

### 3.8 用首个新 rollout 验收 resume

成功加载 checkpoint 只证明模型和 optimizer state 可以恢复，不能单独证明完整训练链路
正常。至少等 `resume_step + 1` 对应的首个 rollout 完成，再把 resume 判定为成功。日志应
同时满足：

- dynamic sampling 选出的 group 数等于 `BC_ROLLOUT_BATCH_SIZE`；
- 出现首个新 rollout 的 `perf <step>` 汇总；
- 没有 `BCPLUS search-server error`；
- MAST task 保持全员 `RUNNING`，restart 和 failed task 都为 0。

例如 tracker 为 39 时，首个 canary 是 rollout 40。可以用下面的只读检查定位证据：

```bash
full_job_name=<new-full-MAST-job-name>
with-proxy mast get-logs --file-path stdout \
  --regex 'BC\+ dynamic sampling|perf 40:|BCPLUS search-server error' \
  "${full_job_name}"
with-proxy mast --output json get-status "${full_job_name}" | \
  jq '{state:.data.state,restarts:.data.numRestarts,attempt:.data.latestAttempt.state}'
```

如果首个 rollout 中 search error 大量增加，即使 job 仍为 `RUNNING` 也不能视为成功；
应立即检查 trainer 启动时记录的 `LOCAL_SEARCH_URL` 是否仍等于当前
`search-server.addr`。

### 3.9 正确理解 tracker 和实时训练进度

`latest_checkpointed_iteration.txt` 只记录最后一个完整落盘的 checkpoint，不会在每个
rollout 后更新。两次保存之间应通过 `perf <step>` 日志或 W&B rollout step 判断实时
进度，不能因为 tracker 暂时不变就认定训练卡住或重复提交任务。

例如 `BC_SAVE_INTERVAL=5`、resume tracker 为 39 时：

- rollout 40 完成后 tracker 仍为 39，这是正常行为；
- rollout 40–44 之间都可能继续显示 tracker 39；
- iter 44 checkpoint 完整保存后，tracker 才会更新为 44。

启用滚动瘦身时，也必须等新 full checkpoint 完整保存后，旧 latest checkpoint 的
weights-only promotion 才会发生。不要在保存窗口中手动修改 tracker、删除 rolling
staging，或把暂时未更新误判为 checkpoint 逻辑失败。

## 4. 本地静态检查

把下面的 `<config>` 替换成新文件：

```bash
bash -n \
  examples/supo_browsecomp/mast/submit_experiment.sh \
  examples/supo_browsecomp/mast/run_trainer.sh \
  <config>
git diff --check
```

如果运行 `shellcheck`，应区分本次新增问题和 `run_trainer.sh` 已存在的 warning；不要为
提交一个实验顺便大范围改写无关代码。

检查依赖文件，只检查存在性，绝不输出 key：

```bash
test -s /home/hhzhang01/eag-wsf/hhzhang01/supo-slime/.llama_key
test -s /home/hhzhang01/eag-wsf/hhzhang01/supo-slime/search-server.addr
```

如果需要新建或替换 search job，完整阅读并执行
[`search/README.md`](search/README.md)。不要仅凭地址文件存在就认为服务可用。

模型文件也应存在于同一个 EAG WSF mount。9B 示例：

```bash
test -e /home/hhzhang01/eag-wsf/hhzhang01/supo-data/Qwen3.5-9B
test -e /home/hhzhang01/eag-wsf/hhzhang01/supo-data/Qwen3.5-9B_torch_dist
```

如果本机 mount 路径不同，先用 `findmnt -t fuse` 找到
`ws://ws.ai.eag0genai/genai_fair_llm` 的本地挂载点。config 中仍然使用 MAST 容器内的
`/mnt/wsfuse/hhzhang01/...` 路径。

## 5. 创建实验专用代码归档

不要覆盖一个正在被其他实验引用的共享 `slime-code.tgz`。令归档 basename 与 config
中的 `MAST_CODE_ARCHIVE` 完全一致。以下示例创建归档、验证关键文件，然后通过同目录
临时文件原子替换目标：

```bash
repo_root=/home/hhzhang01/Long-Horizon-AgenticRL
archive_name=slime-code-9b-example.tgz
archive_tmp_dir="$(mktemp -d)"
local_archive="${archive_tmp_dir}/${archive_name}"
stage_target="/home/hhzhang01/eag-wsf/hhzhang01/supo-slime/${archive_name}"
stage_tmp="${stage_target}.tmp.$$"
trap 'rm -rf -- "${archive_tmp_dir}"' EXIT

tar --exclude='./.git' -czf "${local_archive}" -C "${repo_root}" .
tar tzf "${local_archive}" | rg \
  '^\./examples/supo_browsecomp/mast/(run_trainer\.sh|submit_experiment\.sh|configs/)'
cp "${local_archive}" "${stage_tmp}"
mv -f "${stage_tmp}" "${stage_target}"
stat -c '%y %s %n' "${stage_target}"
sha256sum "${stage_target}"
```

归档会包含当前工作树中的未提交文件，所以打包前必须查看 `git status --short`，确认没有
意外的大文件、secret 或与训练无关的产物。不要把 `.git` 放进归档。

最后从归档中读取新 config 和 runner，确认打进去的是最终版本：

```bash
tar xOzf "${stage_target}" ./<config> | sed -n '1,160p'
tar xOzf "${stage_target}" \
  ./examples/supo_browsecomp/mast/run_trainer.sh | \
  rg -- '--max-tokens-per-gpu|--log-probs-chunk-size|--sglang-mem-fraction-static'
```

不要在输出中显示任何 secret 文件。

## 6. 单独执行并验证 dry-run

先执行只验证、不提交的命令：

```bash
examples/supo_browsecomp/mast/submit_experiment.sh --dry-run <config>
```

MAST CLI 可能需要一两分钟。必须等同一个进程结束。成功输出应包含：

```text
[mast-experiment] dry-run verified: taskCount=<nodes> ROLE_ASSIGNMENT_MAP=trainer_0=<ranks>
```

dry-run JSON 保存于：

```text
/home/hhzhang01/.local/state/mast-experiments/dryruns/<MAST_JOB_NAME>-*.json
```

对最新文件做二次检查：

```bash
dryrun_file=<latest-dry-run-json>
jq -r '[
  .status,
  .dryrun,
  ([.spec.hpc_job_definition.hpcTaskGroups[] |
    select(.name=="trainer_0") | .taskCount][0]),
  ([.spec.app_def.roles[] |
    select(.name=="trainer_0") | .env.ROLE_ASSIGNMENT_MAP][0]),
  ([.spec.app_def.roles[] |
    select(.name=="trainer_0") | .env.MAST_REGION][0]),
  ([.spec.app_def.roles[] |
    select(.name=="trainer_0") | .resource.tags["torchx/named_resources.name"]][0]),
  ([.spec.app_def.roles[] |
    select(.name=="trainer_0") | .resource.capabilities.server_sub_types[0]][0])
] | @tsv' "${dryrun_file}"
```

输出除了 task count/rank map 外，还必须与 config 的 region、host 和预期 server
subtype 完全一致。若请求 H100 却看到 `ZION_EX_A100`，立即停止，不得真实提交。

再检查最终 entrypoint 中的实验参数和归档名：

```bash
jq -r '.spec.app_def.roles[] |
  select(.name=="trainer_0") | .entrypoint' "${dryrun_file}" | \
  rg 'BC_RUN_NAME|BC_MODEL_SIZE|BC_TP|BC_CP|BC_SGLANG_TP|BC_MAX_TOKENS_PER_GPU|BC_LOG_PROBS_CHUNK_SIZE|BC_SGLANG_MEM_FRACTION_STATIC|BCPLUS_FIXED_SEARCH_TOPK|BCPLUS_DOC_WORDS_FULL|BC_OVERRIDE_OPT_PARAM_SCHEDULER|BC_SLIM_INTERMEDIATE_CHECKPOINTS|MAST_CODE_ARCHIVE|slime-code-'
```

只有 JSON 为 `status=ok`、`dryrun=true`，task count/rank map 正确，且所有用户指定参数
都出现在最终 entrypoint 中时，才能继续。

## 7. 只执行一次真实提交

真实提交命令如下。它还会自动再跑一次 dry-run，这是预期行为：

```bash
examples/supo_browsecomp/mast/submit_experiment.sh <config>
```

提交期间遵守以下规则：

- 命令可能先在 dry-run 阶段等待，然后在真实 MAST submit 阶段再次等待。
- `/home/hhzhang01/.local/state/mast-wandb/submissions/submit-*.json` 在 CLI 运行时可能
  暂时为 0 字节。
- 如果 Codex 的 tool call 已 yield 或界面暂时没有新输出，先用 `ps` 检查原进程：

  ```bash
  ps -eo pid,ppid,stat,etime,args | \
    rg '[s]ubmit_experiment|[s]ubmit_with_wandb|[c]li\.sh mast'
  ```

- 只要原进程仍在，就继续轮询该进程和响应文件；绝不能再次运行真实提交命令。
- 如果进程结束且临时响应文件消失，通常是提交成功后文件已移动到 job 专属目录，
  不要误判为失败。

成功后响应文件位于：

```text
/home/hhzhang01/.local/state/mast-wandb/<full-job-name>/submit-*.json
```

读取结果：

```bash
job_state_dir=/home/hhzhang01/.local/state/mast-wandb/<full-job-name>
submit_file="$(find "${job_state_dir}" -maxdepth 1 -type f \
  -name 'submit-*.json' | sort | tail -1)"
jq -r '{
  status,
  job_name:.job.job_name,
  mast_url:.job.mast_url,
  dump_dir:.job.dump_dir,
  tenant:.job.tenant,
  region:.job.region
}' "${submit_file}"
```

必须以 MAST 返回的 full job name（带随机 suffix）为准，而不是 config 中的基础
`MAST_JOB_NAME`。

## 8. 核对 MAST 和 W&B watcher

提交器应自动创建 tmux watcher：

```bash
full_job_name=<full-job-name>
watcher_session="mast-wandb-${full_job_name//./_}"
watcher_session="${watcher_session:0:180}"
tmux has-session -t "${watcher_session}"
sed -n '1,160p' \
  "/home/hhzhang01/.local/state/mast-wandb/${full_job_name}/watcher.log"
```

job name 中如果含点，watcher 的 session name 会把点替换成下划线。watcher log 中出现
`MAST state=PENDING` 是正常的，表示任务已提交、正在等待资源。

也可以直接查询：

```bash
with-proxy mast --output json get-status "${full_job_name}"
```

如果 MAST 已提交成功但 watcher 没有启动，绝不能重新提交 job。只恢复 watcher：

```bash
examples/supo_browsecomp/mast/wandb/submit_with_wandb.sh \
  watch-only "${full_job_name}"
```

## 9. 故障处理和防重复提交

- dry-run JSON 为 0 字节且 dry-run 进程仍在：继续等待。
- dry-run JSON 为 0 字节且相关进程已结束：检查 CLI stderr；只重跑 dry-run，不要直接提交。
- 真实提交 JSON 为 0 字节且 submit 进程仍在：继续等待，绝不重跑。
- 已经获得 full job name 或 MAST URL：任务已经创建，绝不重跑提交命令。
- submit wrapper 以状态 3 退出：job 已创建、watcher 启动失败；使用 `watch-only`。
- MAST 状态为 `PENDING`：正常排队，不是提交失败。
- 容器报 archive 不存在：核对本地 EAG WSF mount 与 config 内 `/mnt/wsfuse` 路径的
  basename 是否完全一致。
- trainer 报 search server 不健康：按 [`search/README.md`](search/README.md) 检查
  `search-server.addr`、execution attempt、当前 task IP 和真实 `/search`，不要因此
  重复提交 trainer。
- trainer 报模型或 torch-dist checkpoint 缺失：先修复 WSF 上的模型文件，再决定是否
  新建任务；不要修改或删除旧任务的数据。
- trainer 报 `BC_RUN_NAME requests resume ... tracker does not exist`：检查是否漏写旧 run
  的随机 suffix；不要创建空 tracker 绕过保护。
- tracker 大于等于 `BC_NUM_ROLLOUT`：这次 submission 不会产生新训练 step；确认用户
  是否确实要求延长训练，然后修改 resume config 的训练终点，不要修改 tracker。
- rolling slim 对旧 checkpoint 输出 `retained_full` warning：训练可以继续，只是该旧
  checkpoint 没有瘦身；不要为此重复提交任务。

## 10. 最终交付格式

向用户简洁报告：

1. “实验已提交”或明确的失败原因；
2. full MAST job name 和可点击的 MAST URL；
3. 当前 MAST state，以及 watcher 是否运行；
4. nodes/GPUs、模型、训练 TP/CP/DP、SGLang TP/engine 数、batch shape 和用户指定的
   特殊参数；
5. 新 config 的本地路径；
6. 代码改动是否尚未 commit。

如果只完成 dry-run 而没有真实提交，必须明确写“尚未提交真实任务”，不能让用户误以为
实验已经启动。
