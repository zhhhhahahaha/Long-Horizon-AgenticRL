# Context length: 分层限制笔记

BC+ rollout 里"context length"这个概念被四层参数控制，各管一段，很容易搞混。这份笔记把每一层的**语义、作用范围、当前值、修改后果**都列清楚。

---

## 四层参数总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│  L1. Model 架构上限                                                      │
│      max_position_embeddings = 262144 (256k)                            │
│      来源: config.json (Qwen3.5-4B)                                     │
│      超了: rope 位置编码失效，输出乱码                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    (SGLang server 拒绝 request >L2)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  L2. SGLang server context 上限                                          │
│      sglang.launch_server --context-length = 131072 (128k)              │
│      来源: launch_sglang_server.sh, SGLANG_CONTEXT_LENGTH env var       │
│      超了: sglang 返回 error, 拒绝 request                               │
│      影响: KV cache 预分配大小 (server 启动时就分配)                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    (Rollout 代码自己 check L3)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  L3. 单 sample 总 context 预算                                           │
│      --rollout-max-context-len = 32768 (debug) / 131072 (real)          │
│      来源: run_qwen3p5_4B_colocate.sh ROLLOUT_ARGS                               │
│      语义: 一个 sample 从头到尾 (prompt + 累计 response) 的上限          │
│      是谁强制: 每个 example 自己 (BC+ 是 compression trigger 用它)      │
│      超了: BC+ 会触发 compression 开新 sub-traj (在到达前 = 0.85 * L3)  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    (BC+ compression trigger 在 L3*0.85)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  L4. SGLang 单次调用 max_new_tokens                                      │
│      --rollout-max-response-len = 16384 (16k)                           │
│      来源: run_qwen3p5_4B_colocate.sh ROLLOUT_ARGS                               │
│      语义: 一次 sglang /generate 调用最多生成多少个 token                │
│      超了: sglang 返回 finish_reason="length" 截断这次生成               │
│      注意: 是"单次"上限，不是"累计"上限                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## 每层现在的值 + 为什么

| 层 | 参数 | 当前值 | Debug 阶段 | 正式训练建议 |
|---|---|---|---|---|
| L1 | Model `max_position_embeddings` | 262144 | 固定 (Qwen3.5) | 固定 |
| L2 | SGLang `--context-length` | 131072 | 128k | 128k (保持) |
| L3 | `--rollout-max-context-len` | 32768 | 32k (aggressive) | 65536 或 131072 |
| L4 | `--rollout-max-response-len` | 16384 | 16k | 32768 |

**必须满足的关系**：`L1 >= L2 >= L3` and `L4 <= L3`（单次生成不能超总预算）。

## 各参数详细解释

### L1: Model `max_position_embeddings` = 262144

- **在哪里**: `/genai/fsx-project/hhzhang01/models/Qwen3.5-4B/config.json`
- **意义**: 模型架构的物理上限。RoPE 位置编码只在这个范围内有效
- **改不改**: 不能改 (会 break model)
- **对我们的影响**: 反正 256k，我们用不完

### L2: SGLang server `--context-length` = 131072

- **在哪里**: `launch_sglang_server.sh` line 47-53, env var `SGLANG_CONTEXT_LENGTH`
- **传给 sglang**: `python -m sglang.launch_server --context-length ${SGLANG_CONTEXT_LENGTH}`
- **意义**: SGLang server 启动时告诉它"我最长会发多长的 request"。它据此**预分配 KV cache**
- **不设的话**: SGLang fallback 到 model config 里的 `max_position_embeddings` (256k)。KV cache 会分配 2x 我们需要的
- **改动后果**:
  - 调高 → 允许更长 request, 但 KV cache 预分配变大, 更少并发
  - 调低 → KV cache 省显存, 但超过这个长度的 request 会被 server 拒绝
- **debug vs 正式**: 都 128k, 不用变 (128k 已经比我们要用的 L3 大很多)

### L3: `--rollout-max-context-len` = 32768

- **在哪里**: `run_qwen3p5_4B_colocate.sh` ROLLOUT_ARGS
- **进入 args.namespace**: `args.rollout_max_context_len`
- **意义**: **单个 sample 的完整 context 预算 (prompt + 累计 response)**
- **谁强制**: slime 本身**不强制** (只用它做参数 default fill), 每个 rollout example 自己决定用不用
- **BC+ 怎么用**: `_run_one_sub_trajectory` 里判断
  ```python
  compress_threshold_tokens = 0.85 * args.rollout_max_context_len
  if len(prompt_ids) + len(response_token_ids) > compress_threshold_tokens:
      触发 compression
  ```
- **其他 example 怎么用**:
  - retool: 判 total_length >= max_context_len 就 truncate + clamp per-turn max_new_tokens
  - geo3k / coding_agent_rl: 类似
- **改动后果**:
  - 调高 → sample 可以跑更长, compression 触发更晚 (每个 sample 用更多资源)
  - 调低 → compression 更早触发 (更多 sub-traj), 每个 sub-traj 更短
- **debug vs 正式**:
  - Debug 32k: aggressive, 强制 compression 频繁触发, 能观察 sub-traj 拆分行为
  - 正式 65k/128k: 让 model 尽量走满上下文再压

### L4: `--rollout-max-response-len` = 16384

- **在哪里**: `run_qwen3p5_4B_colocate.sh` ROLLOUT_ARGS
- **进入 args.namespace**: `args.rollout_max_response_len`
- **传给 sglang**: slime 的 `GenerateState.sampling_params["max_new_tokens"] = args.rollout_max_response_len`
- **意义**: **每次 sglang `/generate` 调用**最多生成多少 token (即 sglang `max_new_tokens`)
- **注意**: 是**单次调用**上限, 不是**整个 sample**上限
  - 一个 sample 有多轮 turn, 每轮独立 sglang 调用, 各自有这个 budget
  - 累加起来一个 sample 的 response 可以 >> L4
- **为什么 debug 16k 就够**:
  - 一次 sglang call 最多吐 16k, 加上 8k obs, response 增长 24k
  - L3=32k trigger threshold=27k, 一次 turn 就会触发 compression
  - 如果 L4=32k, 一次 turn 有可能吐满 32k, 直接一次超预算然后才触发压缩 (浪费一 turn)
- **改动后果**:
  - 调高 → 单次 turn 可能吐很长 output, 但可能超 L3 才触发 compression (浪费)
  - 调低 → 每次 turn 更快结束, model 输出被截断更多 (finish_reason="length")

## Compression trigger 的完整流程

```python
# 在 _run_one_sub_trajectory 每一轮末尾, _run_action 之后:

total_context_len = len(prompt_ids) + len(response_token_ids)
if total_context_len > 0.85 * args.rollout_max_context_len:
    # 触发 compression:
    #   1. 追加 SUMMARY_PROMPT_SEARCH 作为 user turn
    #   2. sglang call 生成 summary
    #   3. 关闭当前 sub-traj
    #   4. generate() 会开新 sub-traj: prompt = 原 question + summary
```

数字例子 (L3=32768, threshold=0.85):
- compress_threshold_tokens = 27852
- prompt_ids = 1000 (system + user)
- 当 response_token_ids > 26852 时触发

## 三个隐含关系必须满足

1. **L1 >= L2**: model 架构决定 sglang 最多能设多长
   - 现在: 256k >= 128k ✅
2. **L2 >= L3**: sglang server 至少要允许一个 sample 用完总预算
   - 现在: 128k >= 32k ✅
   - 如果 L3 提到 256k, L2 也要提 (但会撞 L1)
3. **L4 <= L3 - prompt_len**: 单次生成不能超总预算
   - 现在: 16k <= 32k - 1k ✅
   - 严谨说应该动态 clamp (retool 的做法), 但我们靠 compression 兜底

## 修改任一层时的 checklist

- **改 L2** (SGLang server context): `SGLANG_CONTEXT_LENGTH` env var 或改 `launch_sglang_server.sh:48`. **必须重启 sglang server** (因为 KV cache 是启动时分配的)
- **改 L3** (sample 总预算): 改 `run_qwen3p5_4B_colocate.sh` 的 `--rollout-max-context-len`. **不用重启 sglang**, 下次 rollout 生效
- **改 L4** (sglang 单次): 改 `run_qwen3p5_4B_colocate.sh` 的 `--rollout-max-response-len`. **不用重启 sglang**
- **改 threshold**: `BCPLUS_COMPRESS_THRESH` env var (默认 0.85)

## 常见错误

- **只调 L3 不调 L4 (或反过来)**: 会失衡, L3 变大 L4 没变 → single turn 到 L4 就 stop 早, compression 从没触发
- **忘设 L2 让它 fallback 到 L1**: KV cache 预分配 2x 需要的
- **让 L4 >= L3**: 一次 sglang call 可以直接超 L3, compression 用不上
- **不设 L3 (`--rollout-max-context-len`)**: 我们 assert 强制要求了, 会在 rollout 启动时报错 (防误用)

## Metric 层面

Slime 记录的这几个 metric 分别对应:

| Metric | 是哪个长度 |
|---|---|
| `rollout/response_len/*` | 纯 model emit token 数 (`sum(loss_mask)`), obs 不算 |
| `multi_turn/wo_obs_response_length/*` | 同上 |
| `multi_turn/raw_response_length/*` | 含 obs 的 response 全长 (`len(loss_mask)`) |
| `raw_response_length_clip_ratio` | 达到 L4 (`rollout_max_response_len`) 的样本比例 |
| `bcplus/final_response_len_mean` (我们加的) | 最终 sub-traj 的 response 全长 |

**看 context 有多满**: 用 `multi_turn/raw_response_length/mean + prompt_len_mean` 或 `bcplus/final_response_len_mean`. 千万别用 `rollout/response_len/mean` (那只是 model emit 部分, obs 都被 mask 掉了).
