# 在 MAST 上提交和维护 BrowseComp Search Server

这是一份给后续 Codex session 使用的执行手册。Search server 是一个独立的
1-node、8-GPU MAST job。每个服务通过自己显式指定的 OILFS 地址文件，向训练和
评测任务发布当前 IPv6 地址。

## 重要约束

- 每次提交都必须显式选择 tenant、region、host 和 priority；不要从地址文件名推断
  硬件。优先级只能降低被抢占概率，不能保证节点或 IP 永远不变。
- 同一个地址文件同一时间只能有一个 writer。多个 search server 可以共存，但必须
  使用不同的 `SEARCH_ADDR_FILE`；替换同一个地址文件背后的服务时，仍要先停止旧
  job，确认它不会重排，再提交新 job。
- 真实提交命令只能执行一次。MAST job 可能在 CLI 暂时没有返回、列表暂时查不到时
  已经开始创建；先等待原进程并继续查询，绝不能据此重复提交。
- 不要使用 `wandb/submit_with_wandb.sh`。Search server 不产生需要同步的 W&B run，直接
  使用结构化的 `rl/cli.sh mast --json` 命令即可。
- 地址文件更新不等于服务已经可用。必须等 `/health` 和一条真实 `/search` 都成功，
  才能启动训练或评测。

### 多个 search server 共存

不同节点上的服务不会发生端口冲突；冲突只会来自多个 job 覆盖同一个 discovery
文件。每个新服务先选择一个稳定且唯一的 `search_server_id`，再派生地址文件：

```bash
search_server_id=<unique-search-service-id>
SEARCH_ADDR_FILE=/mnt/wsfuse/hhzhang01/supo-slime/search-servers/${search_server_id}.addr
```

启动 server 时显式传入该文件：

```bash
SEARCH_ADDR_FILE=/mnt/wsfuse/hhzhang01/supo-slime/search-servers/<unique-search-service-id>.addr \
  bash /slime-src/examples/supo_browsecomp/mast/search/run_search_server.sh
```

对应训练 config 也必须写同一路径：

```bash
SEARCH_ADDR_FILE=/mnt/wsfuse/hhzhang01/supo-slime/search-servers/<selected-search-service-id>.addr
```

`run_search_server.sh` 和 `submit_experiment.sh` 都不提供默认地址；缺失变量会直接失败。
不要在 config 中固定 `LOCAL_SEARCH_URL`。MAST restart 后 server IP 可能变化，而独立
地址文件可以由对应 server 更新为新 IP。

## 1. 提交前检查

从仓库根目录开始：

```bash
cd /home/hhzhang01/Long-Horizon-AgenticRL

local_archive=/home/hhzhang01/eag-wsf/hhzhang01/supo-slime/slime-code.tgz
test -s "${local_archive}"
tar tzf "${local_archive}" | rg \
  '^\./examples/supo_browsecomp/(search_server\.py|mast/search/run_search_server\.sh)$'

mast list-jobs \
  --cluster MastGenAICluster \
  --my \
  --status RUNNING \
  --detailed
```

不要只根据 job 名猜测用途。旧 search job 使用过通用的
`avocado_rev1_rl_debug_80m-*` 名称；必要时检查定义中是否真正启动 search server：

```bash
old_job=<full-MAST-job-name>
mast get-job-definition "${old_job}" | \
  rg 'run_search_server\.sh|SEARCH_ADDR|taskCount|restartPolicy'
```

如果用户要求替换旧服务，停止精确确认过的旧 job：

```bash
mast kill \
  --comment 'Replacing the existing BrowseComp search server.' \
  "${old_job}"

mast list-jobs \
  --cluster MastGenAICluster \
  --job-name "${old_job}" \
  --include-historical \
  --detailed
```

等待 `User Intent=KILLED`，并确认 task group 已进入 shutdown。这个停止流程只适用于
复用同一个地址文件；写不同地址文件的服务可以同时运行。

## 2. 构造并验证提交命令

每次提交都要填写下面所有变量。`search_server_id` 是服务身份，不能用 region 或 host
隐式代替；资源参数应按本次目标硬件独立填写：

```bash
mast_rl_cli=/data/users/hhzhang01/fbsource/genai/msl/rl/cli.sh
search_server_id=<unique-search-service-id>
search_job_base="supo_search_server_${search_server_id}"
search_addr_file="/mnt/wsfuse/hhzhang01/supo-slime/search-servers/${search_server_id}.addr"
search_tenant=<tenant-alias>
search_region=<region>
search_host=<host-type>
search_priority=<priority>
```

`--job_name` 会让最终 job 使用可识别的随机后缀名称；不要改回只影响 program
名称的 `--name`。选好上面一组变量后，构造命令：

```bash
search_mast_cmd=(
  "${mast_rl_cli}" mast
  --json
  "--tenant=${search_tenant}"
  "--region=${search_region}"
  "--job_priority=${search_priority}"
  --workspace=None
  --main_package=xlformers_pretrain1:latest
  program avocado.rev1.rl.debug_80m
  --roles=trainer_0
  "--job_name=${search_job_base}"
  --enable_ttls=True
  --retries=3
  --use_conda_docker=True
  --conda_docker_image=588845226011.dkr.ecr.us-east-2.amazonaws.com/msl_infra/slime:hhz-20260629a
  "--docker_custom_cmd=mkdir -p /slime-src && tar xzf /mnt/wsfuse/hhzhang01/supo-slime/slime-code.tgz -C /slime-src && SEARCH_ADDR_FILE=${search_addr_file} bash /slime-src/examples/supo_browsecomp/mast/search/run_search_server.sh"
  "--host=${search_host}"
  --wsf_src=ws://ws.ai.eag0genai/genai_fair_llm
)
```

先做一次 dry-run，并确认只有 1 个 task、8 个 GPU、启动命令正确：

```bash
"${search_mast_cmd[@]}" --dryrun | jq '{
  status,
  dryrun,
  task_count: .spec.hpc_job_definition.hpcTaskGroups[0].taskCount,
  gpu_per_task: .spec.hpc_job_definition.hpcTaskGroups[0].spec.resourceLimit.compute.gpu,
  command: .spec.hpc_job_definition.hpcTaskGroups[0].spec.command
}'
```

只有 `status=ok`、`dryrun=true`、`task_count=1`、`gpu_per_task=8`，且 command 同时
包含 `SEARCH_ADDR_FILE=${search_addr_file}` 和 `search/run_search_server.sh` 时才能
真实提交。

## 3. 真实提交只能执行一次

```bash
"${search_mast_cmd[@]}"
```

这条命令可能先长时间只打印 Conda 和 TorchX 初始化日志。此时：

1. 如果工具返回 session/cell ID，继续等待同一个进程；
2. 用 `ps` 确认原提交进程是否仍存在；
3. 查询 MAST，允许 job 延迟一两分钟才出现在列表中；
4. 无论暂时是否看到 job，都不要再次运行真实提交命令。

```bash
mast list-jobs \
  --cluster MastGenAICluster \
  --my \
  --prefix supo_search_server_critical \
  --include-historical \
  --detailed
```

记录完整 job 名：

```bash
search_job=<supo_search_server_critical-full-suffix>
```

如果误产生重复 job，只保留已确认的这一份，逐个精确停止其他 job；不要使用模糊匹配
批量 kill。

## 4. 等待容器和模型启动

首次启动会拉取较大的 slime 镜像。MAST 显示 `RUNNING` 时，容器内部仍可能处于镜像
拉取或模型加载阶段。持续检查同一个 job：

```bash
mast --output json get-status "${search_job}" | jq '{
  state: .data.state,
  restarts: .data.numRestarts,
  attempt: .data.latestAttempt.attemptIndex,
  attempt_state: .data.latestAttempt.state,
  tasks: [
    .data.latestAttempt.taskGroupExecutionAttempts[][]
    .taskExecutionAttempts[][] |
    {state, hostname, taskIp}
  ]
}'

mast get-logs \
  --file-path stdout \
  --regex '\[docker\]|SEARCH_ADDR|Starting search server|Worker [0-9]+: Ready|Loading corpus|Loaded [0-9]+ documents|Traceback|init error' \
  "${search_job}"
```

正常顺序包括镜像拉取、`SEARCH_ADDR=...`、corpus/embedding 加载和 8 个 worker ready。
GPU CDI 中关于缺少显示设备或图形库的 warning 通常与计算无关；以 job restart、
Python traceback、worker init error 和健康检查为准。

## 5. 验证地址和服务

根据本次 `search_server_id` 选择 devserver 和容器内路径：

```bash
search_server_id=<unique-search-service-id>
container_addr_file="/mnt/wsfuse/hhzhang01/supo-slime/search-servers/${search_server_id}.addr"
addr_file="/home/hhzhang01/eag-wsf/hhzhang01/supo-slime/search-servers/${search_server_id}.addr"
```

确认文件 mtime 晚于本次 job 创建时间、文件里的 IP 等于本次 task IP，然后检查
health、stats 和真实搜索：

```bash
stat -c '%y %s %n' "${addr_file}"
search_addr="$(tr -d ' \t\r\n' < "${addr_file}")"

curl --noproxy '*' -g -fsS \
  --connect-timeout 3 --max-time 10 \
  "http://${search_addr}/health" | jq .

curl --noproxy '*' -g -fsS \
  --connect-timeout 3 --max-time 10 \
  "http://${search_addr}/stats" | jq .

curl --noproxy '*' -g -fsS \
  --connect-timeout 3 --max-time 70 \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the capital of France?","k":3}' \
  "http://${search_addr}/search" | \
  jq '{result_count: (.results | length), took_ms}'
```

成功标准：

- `/health` 返回 HTTP 200，并显示 8 workers；
- `/stats` 显示 corpus、8 workers，队列没有持续堆积；
- `/search` 返回 3 条结果而不是仅仅端口可连接；
- 地址文件只由目标 full MAST job 写入，另一个 search server 的文件和 mtime 未变化；
- MAST job 为 `RUNNING` 且 restart 为 0。

满足全部条件后，才能提交或 resume 训练任务。

## 6. 抢占、重排和地址变化

MAST 可能因为更高优先级任务、容量回收、宿主机故障或维护而重排 search job。新的
execution attempt 可能保留同一个 full job name，但会获得不同的主机和 IPv6；
`search/run_search_server.sh` 会把新地址写回该 job 显式选择的 `SEARCH_ADDR_FILE`。

`CRITICAL` priority 会降低因普通容量竞争而被抢占的概率，但不能提供以下保证：

- job 一定运行在 tenant 自有物理容量上；
- job 永远不被抢占；
- retry 一定回到同一计算节点；
- IPv6 永远不变。

需要容量保证时，应使用 tenant reservation、FlexPool 或专属 entitlement；即便如此，
也仍要处理宿主机故障导致的 IP 变化。

当前 trainer 在启动时只读取一次 config 指定的 `SEARCH_ADDR_FILE`，并缓存
`LOCAL_SEARCH_URL`/HTTP client。如果 search job 在训练过程中换 IP，地址文件虽然更新，
已经运行的 trainer 也不会自动切换。发现大量
`BCPLUS search-server error ... ConnectError` 时：

1. 比较 trainer 启动日志中的 `LOCAL_SEARCH_URL` 和当前地址文件；
2. 检查 search job 的 attempt、task IP 和 `/health`；
3. 如果 trainer 使用旧地址，立即停止训练，避免 dynamic sampling 持续产生无效 rollout；
4. 从最新完整 checkpoint resume；
5. 在 trainer 支持安全的动态地址刷新前，不要假设它会自行恢复。

## 7. 最终交付格式

向用户报告：

1. 新 search job 的完整名称；
2. 实际 tenant、region、host、priority 和 GPU 数；
3. MAST state、attempt 和 restart 数；
4. 地址文件是否由本次 job 更新；
5. `/health`、`/stats` 和真实 `/search` 是否通过；
6. 所有旧 job 或误提交的重复 job 是否已经停止。
