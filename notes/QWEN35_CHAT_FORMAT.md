# Qwen3.5 chat template & tool-call 格式 cheat sheet

给未来的自己（或者接手这块代码的人）看的参考。所有内容都是从 HF checkpoint 里的
`chat_template.jinja` 反推出来的（在 `/genai/fsx-project/hhzhang01/models/Qwen3.5-4B/`），
拿不准的话直接翻那个 template 文件（153 行）。

---

## TL;DR

- Tool call 是 **双层包裹**：`<tool_call><function=NAME>...</function></tool_call>`，**不是**裸的 `<function>`。
- Parameter value 前后**倾向于**有换行：`<parameter=name>\n{value}\n</parameter>`。是强 preference 但不是 hard requirement，**parser 必须 `.strip()`**。
- Tool response 用 `role: tool` 追加，template 自动包成 `<|im_start|>user\n<tool_response>...</tool_response><|im_end|>`。
- Thinking 默认开着：generation prompt 结尾是 `<|im_start|>assistant\n<think>\n`。
- **多轮 rollout re-render 时，老 assistant turn 的 `<think>...</think>` 会被静默剥掉。** 只有"最后一个非-tool 的 user 消息"之后的 assistant thinking 会保留。
- System 消息必须在位置 0，template 用 `raise_exception` 强制。
- 给 `apply_chat_template(..., tools=[...])` 传 tools，template 会自动在 system 前面拼一个 `<tools>` schema 块 + `<tool_call>` 格式说明。
- 采样 stop tag 用 `</tool_call>`（不是 `</function>`），保证外层 wrapper 写完再截断。

---

## `apply_chat_template` 渲染

### 不传 `tools=`（裸调）

```python
tokenizer.apply_chat_template(
    [{"role": "system", "content": "You are helpful."},
     {"role": "user",   "content": "Hi"}],
    tokenize=False,
    add_generation_prompt=True,
)
```

出来长这样：

```
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hi<|im_end|>
<|im_start|>assistant
<think>

```

### 传 `tools=[...]`

传 `tools=[SEARCH_SCHEMA, ...]` 时，template 会自动在 system 前面拼一个 tools 教学块。**结构**：

```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{...每个 tool 的 JSON schema...}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:
{...<tool_call> 格式示例...}
<IMPORTANT>
{...几条格式提醒...}
</IMPORTANT>

{...你自己的 system content 拼在这里...}<|im_end|>
<|im_start|>user
Hi<|im_end|>
<|im_start|>assistant
<think>

```

<details>
<summary>👆 完整 render 输出（点开看）</summary>

```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "search", "description": "...", "parameters": {...}}}
{"type": "function", "function": {"name": "open_page", ...}}
{"type": "function", "function": {"name": "finish", ...}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>

You are helpful.<|im_end|>
<|im_start|>user
Hi<|im_end|>
<|im_start|>assistant
<think>

```

</details>

注意几个点：
- 每个 tool 直接 `tool | tojson` 拼进 `<tools>` 块。用 OpenAI 的 `{"type":"function","function":{"name":..., "description":..., "parameters":{...}}}` 结构。
- 你的 system content 会被拼在 template 自动生成块**之后**（用 `\n\n` 分隔）。如果没传 system，template 还是会自己生成 tools 块。
- `<IMPORTANT>` 提醒是 template 自带的，不是你写的。

---

## Tool call 格式（assistant OUTPUT）

一次 tool call 长这样：

```
<tool_call>
<function=search>
<parameter=query>
who is elon musk
</parameter>
<parameter=topk>
5
</parameter>
</function>
</tool_call>
```

**规则**：
- 外层必须 `<tool_call>...</tool_call>`，内层 `<function=NAME>...</function>`。**别** emit 裸 `<function>`。
- 每个 parameter 块：`<parameter={name}>\n{value}\n</parameter>\n`（value 前后各一个 `\n`）。
- 一个 assistant turn 里多个 tool call 就多个 `<tool_call>` 块，之间用 `\n` 分隔。

**Parser 层面必须 `.strip()`**：模型不保证严格加换行（value 短的时候经常 inline），parse 出的 value 可能带首尾 `\n` 或者没有。SUPO 原正则 `<parameter=([^>]+)>(.*?)</parameter>` + DOTALL 两种都能 match，但需要手动 strip。

<details>
<summary>为什么模型会按换行格式生成？（prompt 里其实没明说）</summary>

Prompt 里其实**没有一句话**明确说"parameter value 前后必须换行"。`<IMPORTANT>` 只强调三件事：外层必须包 `<tool_call></tool_call>`、required 参数必须给、reasoning 只能在 tool call 前。

那模型是怎么"学"会换行格式的？四条路径叠加，从强到弱：

1. **Prompt 里的 few-shot 示例**（最主要）— 上面 tools 块里的 `<tool_call><function=example_function_name>...` 示例本身就是换行版，`value_1` 明明可以放在同一行但示例故意拆成三行。模型看到示例这么写就倾向这么写。
2. **Pretraining / SFT 分布** — Qwen3.5 训练时喂过大量 tool-use 数据，那些数据里就是换行格式。模型先验偏向换行。
3. **Template 对历史 assistant 的 re-render** — 如果 assistant turn 存成结构化 `{"tool_calls": [...]}`，template 每次 re-render 都会重新序列化成换行版，模型下一轮看到自己之前的 tool call 都是换行版 → 自我强化。**但如果存成裸 content 字符串（我们的做法），template as-is 保留，模型 emit 什么就是什么。**
4. **停止 tag（`</tool_call>`）间接约束** — 保证外层结构完整，但不管 value 换行。

**结论：模型 *倾向于* 换行版，但 *不保证*。** value 短的时候（比如 `<parameter=topk>5</parameter>`）经常会出现 inline 格式。所以 parser 必须两种都兼容。

</details>

<details>
<summary>Value 序列化规则（只在 template re-render 结构化 tool_calls 时触发）</summary>

**触发条件**：assistant 消息里带**结构化**的 `tool_calls` 字段，比如：

```python
{"role": "assistant", "content": "", "tool_calls": [
    {"function": {"name": "search", "arguments": {"query": "elon musk", "topk": 5}}}
]}
```

Template 拿到 arguments dict，跑下面这段 jinja 序列化成字符串（`chat_template.jinja:128-137`）：

```jinja
{%- if tool_call.arguments is defined %}
    {%- for args_name, args_value in tool_call.arguments|items %}
        {{- '<parameter=' + args_name + '>\n' }}
        {%- set args_value = args_value | tojson | safe
                             if args_value is mapping or (args_value is sequence and args_value is not string)
                             else args_value | string %}
        {{- args_value }}
        {{- '\n</parameter>\n' }}
    {%- endfor %}
{%- endif %}
```

**规则**：`dict` / `list` → `tojson`；其他（string、int、float、bool、None）→ Python `str()` 不带引号。

| Python value | 走哪支 | 序列化输出 |
|---|---|---|
| `"elon musk"` | `str()` | `elon musk` |
| `5` | `str()` | `5` |
| `True` | `str()` | `True`（首字母大写，**不是**JSON 的 `true`） |
| `None` | `str()` | `None`（**不是**JSON 的 `null`） |
| `3.14` | `str()` | `3.14` |
| `["a", "b"]` | `tojson` | `["a", "b"]` |
| `{"k": "v"}` | `tojson` | `{"k": "v"}` |

坑：
- `bool` / `None` 走 Python `str()` 不走 JSON，所以是 `True/False/None`。
- String 走 `str()` **不做** JSON escaping，双引号原样保留。

**我们的 rollout 不走这条路**：assistant 消息只塞裸 content 字符串（模型生成的原文），不带结构化 `tool_calls` 字段。所以这段序列化逻辑**用不到**——写在这里备忘，将来如果换成结构化路径要知道有这个行为。

</details>

---

## Tool response 格式（tool INPUT 到下一轮）

**推荐**：用 `tool` role，template 自动包装：

```python
chat.append({"role": "tool", "content": "results here"})
```

Template 输出：

```
<|im_start|>user
<tool_response>
results here
</tool_response><|im_end|>
```

**注意**：内部本质是 user turn，模型看到的外层是 `<|im_start|>user`。这是 Qwen 的 role alias 小把戏。

**连续 tool response 合并**：多个 `{"role": "tool", ...}` 挨着，只会产生**一个** `<|im_start|>user` 开头 + 多个 `<tool_response>` 块 + **一个** `<|im_end|>`。

<details>
<summary>Option B: 手动写 user turn（不推荐但有效）</summary>

```python
chat.append({"role": "user", "content": "<tool_response>\nresults\n</tool_response>"})
```

Template 检测到 `<tool_response>` wrapping 后当 tool role 处理（也影响 thinking 的 `last_query_index` 判定，见 thinking gotcha）。

除非有特别理由，直接用 `role: tool` 更清爽。

</details>

---

## Thinking 行为

Qwen3.5 是 thinking model。每次 assistant 生成的完整格式：

```
<think>
{reasoning}
</think>

{可见的回答 / tool_call}
```

`add_generation_prompt=True` 追加 `<|im_start|>assistant\n<think>\n`，模型默认从 think 块内部开始生成。禁用 thinking：`add_generation_prompt=True` + `enable_thinking=False`，追加带空 think 块的 prompt。

### ⚠️ Gotcha：老的 thinking 在 re-render 时被静默剥掉

Template 只保留"最后一个非-tool user 消息之后"的 assistant turn 的 `<think>...</think>` 块。之前的 assistant turn 在 re-render 时会**丢掉 thinking**。

**对我们的 search-loop 影响**：几乎没有，因为 rollout 中间不会插入真正的 user 消息（都是 tool response）。所有 assistant turn 的 index 都 > `last_query_index`（初始 user query 的 index），thinking 全保留。

**什么时候会咬到**：如果 rollout 中间真的注入一条 user 消息（比如"格式不对请重来"），`last_query_index` 会跳到那条消息，之前所有 assistant 的 thinking 瞬间从上下文消失。

<details>
<summary>Template 里的具体 jinja 逻辑</summary>

`chat_template.jinja` 第 78-90 行（找 `last_query_index`）：

```jinja
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- if ns.multi_step_tool and message.role == "user" %}
        {%- set content = render_content(message.content, false)|trim %}
        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
            {%- set ns.multi_step_tool = false %}
            {%- set ns.last_query_index = index %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
```

第 108-113 行（决定是否保留 thinking）：

```jinja
{%- if loop.index0 > ns.last_query_index %}
    {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
{%- else %}
    {{- '<|im_start|>' + message.role + '\n' + content }}   {# thinking 被剥了 #}
{%- endif %}
```

</details>

<details>
<summary>具体例子（一次 search → answer 和两次 search）</summary>

**例 1**：一次 search 一次 finish

```
[msg 0] system:    ...
[msg 1] user:      "who founded SpaceX?"       ← 最后一个非-tool user query
[msg 2] assistant: "<think>Let me search</think>\n\n<tool_call>...</tool_call>"
[msg 3] tool:      "results: Elon Musk..."     ← 被包成 <tool_response>
[msg 4] assistant: "<think>I have enough</think>\n\n<tool_call>...finish...</tool_call>"
```

`last_query_index` = 1。Re-render 时 msg 2、msg 4 都 > 1，thinking **全保留**。

**例 2**：两次 search 一次 finish

```
[msg 0] system
[msg 1] user       ← 最后一个非-tool user
[msg 2] assistant  "<think>plan A</think>...tool_call..."
[msg 3] tool
[msg 4] assistant  "<think>refine</think>...tool_call..."
[msg 5] tool
[msg 6] assistant  "<think>done</think>...finish..."
```

`last_query_index` 还是 1。Assistant 在 index 2, 4, 6 都 > 1，三段 thinking **全部保留**。没问题。

</details>

<details>
<summary>reasoning_content 字段（少用）</summary>

Thinking 也可以作为 assistant 消息的独立字段传，不用嵌 `<think>`：

```python
{"role": "assistant", "reasoning_content": "let me think...", "content": "final answer"}
```

Template 优先用 `reasoning_content`；没有则从 content 里 split `</think>`：

```jinja
{%- if message.reasoning_content is string %}
    {%- set reasoning_content = message.reasoning_content %}
{%- else %}
    {%- if '</think>' in content %}
        {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
        {%- set content = content.split('</think>')[-1].lstrip('\n') %}
    {%- endif %}
{%- endif %}
```

我们的 rollout 不用管这个，把 assistant content 里嵌 `<think>...</think>` 直接送就行。

</details>

---

## Message role 规则

| Role | 允许位置 | 备注 |
|---|---|---|
| `system` | 位置 0 only。`raise_exception('System message must be at the beginning.')`。 | 传了 `tools=` 时 content 会被 `\n\n` 拼在 tools 块后面。 |
| `user` | System 之后任意位置。**多轮 tool 检测**：被 `<tool_response>` 包裹的 user turn 不算"真正的 user query"（影响 thinking 剥不剥）。 | 至少要有一个非-tool-wrapped user 消息，否则 `raise_exception('No user query found in messages.')`。 |
| `assistant` | System 之后任意位置。支持和 `content` 并列的 `tool_calls` 字段。 | Thinking 保留基于 `last_query_index`。 |
| `tool` | 任意位置；连续 tool 消息会合并成一个 user turn，装多个 `<tool_response>` 块。 | 内部渲染成 `<|im_start|>user\n<tool_response>...</tool_response>`。 |

Vision content（image/video items）通过 multimodal `content: [{...}]` list 支持；我们纯文本 rollout 用不到。

---

## Rollout 代码 sanity check

如果我们每轮都用 `apply_chat_template(chat_so_far, tools=TOOLS, add_generation_prompt=True)` re-render，要验证：

1. `chat_so_far[0]` 是 `system` 且内容匹配 `tool_schemas.QWEN_SYSTEM_PROMPT`
2. `chat_so_far[1]` 是原始 user 问题（parquet 里的）
3. 每个后续 assistant turn 的 content 都以 `<think>` 开头，并且至少包含一个 `<tool_call>...</tool_call>` 块
4. Tool response 用 `{"role": "tool", "content": obs}` 追加
5. **绝对不要**手动往 content 里塞 `<|im_start|>` / `<|im_end|>` — template 自己管
6. `_extract_fn_call` 正则 match `<tool_call>\s*<function=...>...\s*</function>\s*</tool_call>`，parse 出的每个 arg value 都 `.strip()`
7. 采样 stop tag 是 `</tool_call>`（不是 `</function>`）

---

## 常见踩坑

- **emit 裸 `<function>` 的 tool call** — SUPO reference impl 这么写；Qwen3.5 不认。必须 `<tool_call>` 包住。
- **忘了 parameter value 前后的换行** — inline 版能 parse 但 off-distribution，template 序列化出的是换行版。
- **传 tool schema 忘了 OpenAI 的外层 `{"type":"function","function":{...}}` wrapper** — template 直接 `tool | tojson`，结构错了模型看不懂。
- **`add_generation_prompt=True` 之后又手动追加 `<|im_start|>assistant\n<think>\n`** — 会重复。
- **假设老的 thinking 会保留** — 一般会，但见 gotcha。
- **在 `</function>` 停生成** — 模型还没闭合外层 `<tool_call>`，下一次 parse 和 re-render 会错位。要在 `</tool_call>` 停。
- **Parse 出的 arg value 没 strip** — 前后 `\n` 会被带到 search server / open_page，可能影响下游行为。
