# SUPO — BrowseComp-Plus pipeline for slime

Port of the SUPO paper's ([arXiv 2510.11967](https://arxiv.org/abs/2510.11967))
BrowseComp-Plus training loop onto slime. Self-contained under this directory —
no runtime dependency on the SUPO / FoldAgent repo (everything we need is
either ported or vendored here).

This is a **pipeline-first** port:

- Rollout: single-session ReAct (`workflow="search"` in the SUPO reference
  impl), no branch / summary / process reward.
- Algorithm: slime's stock GRPO. FoldGRPO advantage estimation, branch
  scheduling, and `process_reward=[flat,scope]` are left for a follow-up.
- Judge: OpenAI-family model via Meta's internal MetaGen gateway (Llama API
  OpenAI-compat endpoint). Defaults to `gpt-5-4-genai-dss4` for both the
  primary judge and the near-miss fallback.
- Retrieval: `search_server.py` (vendored from the SUPO reference impl with a
  one-line patch to accept local embedding directories), run unchanged on a
  dedicated GPU node. Slime dials it over HTTP via `LOCAL_SEARCH_URL`.

## Files

- `search_server.py` — retrieval server (Qwen3-Embedding-8B over the
  BrowseComp-Plus corpus with pre-computed embeddings). Run once per cluster on
  a long-lived Slurm job.
- `local_search_client.py` — thin async HTTP client (`/search`, `/open`) for
  talking to the retrieval server from inside slime.
- `generate_with_bcplus.py` — the two callables wired into slime's custom
  generate + reward hooks:
  - `generate()` — multi-turn ReAct loop. Stops the sampler at `</function>`,
    parses the last `<function=...>` block, dispatches to `search`/`open_page`/
    `finish`, and appends the observation as `loss_mask=0` text before the next
    turn. Trainable tokens (model output) use `loss_mask=1`.
  - `reward_func()` — extracts the `<finish answer=...>` payload from the
    trajectory metadata and calls the MetaGen judge (with `em_score` /
    `relaxed_em` fast paths).
- `run_qwen3p5_4B.sh` — two-part launcher (login pod → in-container). Trains
  Qwen3.5-4B for 20 rollouts as a smoke test. Bump `HF_CKPT_HOST` /
  `scripts/models/qwen3.5-9B.sh` for the first-real-run 9B move. Auto-reads
  the search-server hostname from
  `/genai/fsx-project/hhzhang01/logs/search-server.hostname` if
  `LOCAL_SEARCH_URL` isn't already set.
- `launch_search_server.sh` — idempotent orchestrator that sbatch's a
  long-lived (7-day, 1 GPU) retrieval server, waits for `/health`, and writes
  the resolved `host:port` to the hostname file. Reuses an existing job if
  its remaining walltime is `>= MIN_HOURS_REMAINING` (default 48h); otherwise
  scancels + resubmits so a training job doesn't outlive its search server.
- `launch_all.sh` — one-button entry point: calls
  `launch_search_server.sh` with a `MIN_HOURS_REMAINING` that covers the
  requested training walltime + a buffer, then invokes `run_qwen3p5_4B.sh`,
  then runs `aws-cluster/wandb-sync.sh` after srun exits.

## Prerequisites

### 1. Search server (biggest external dep)

The retrieval corpus + embeddings need ~4 GB and the embedding model ~8 GB. One
80GB GPU is plenty for BC+ (corpus is ~830K docs).

**Preferred (scripted, idempotent)** — from the login pod:
```bash
bash /home/hhzhang01/slime/examples/supo_browsecomp/launch_search_server.sh
```
This sbatch's a `--time=7-00:00:00` `--gpus=1` job on partition `a100`, waits
until `/health` is 200, and writes `<host>:<port>` to
`/genai/fsx-project/hhzhang01/logs/search-server.hostname`. Subsequent runs
of the same script reuse the existing job if it has `>= MIN_HOURS_REMAINING`
hours left (default 48). Set `MIN_HOURS_REMAINING=99999` to force a fresh
server every time.

**Manual fallback** — if you want to bypass the wrapper:
```bash
sbatch --nodes=1 --gpus=1 --time=7-00:00:00 --partition=a100 \
  --job-name=supo-search-server \
  --output=/genai/fsx-project/hhzhang01/logs/search-server.log \
  --wrap='
    ENROOT_TEMP_PATH=/dev/shm ENROOT_MOUNT_HOME=false \
    enroot start \
      --mount /home/hhzhang01/slime:/slime \
      --mount /genai/fsx-project/hhzhang01:/genai_hh \
      /home/hhzhang01/slimerl+slime+latest.sqsh \
      bash -c "
        cd /slime/examples/supo_browsecomp
        python search_server.py \
          --model /genai_hh/models/Qwen3-Embedding-8B \
          --corpus /genai_hh/datasets/browsecomp-plus-corpus \
          --corpus-embedding-dataset /genai_hh/datasets/browsecomp-plus-embeds \
          --host 0.0.0.0 --port 8000
      "
  '
```

In either case, wait for the log to print `Loaded ~830000 documents`,
`Worker 0: Ready`, and `Application startup complete.` — then verify:
```bash
curl -X POST http://<hostname>:8000/health
curl -X POST http://<hostname>:8000/search \
  -H 'content-type: application/json' -d '{"query":"Elon Musk","k":3}'
```

### 2. MetaGen judge (Llama API OpenAI-compatible endpoint)

Compute nodes cannot reach `api.openai.com`. Judge calls therefore go through
the internal Llama API OpenAI-compat endpoint (`https://api.llama.com/compat/v1/`)
against MetaGen-hosted OpenAI models. The example uses the standard `openai`
Python SDK — only the `base_url` + `api_key` change.

Setup (one-time):

1. Create a Llama API key (`LLM|...`) at
   <https://www.internalfb.com/metagen/tools/llm-api-keys> → "Create API Key".
2. Give that key entitlement to the judge model in Model Explorer
   (<https://www.internalfb.com/metagen/tools/aimodels/explorer>) → find
   `gpt-5-4-genai-dss4` (used for both primary and fallback) → "Access
   Control" tab → add your `mg-api-...` key. OpenAI models are typically MSL
   Self-Service; non-MSL teams file a 3P access request via
   <https://fburl.com/metagen-3p-model-access-post>.
3. Verify from the login pod:
   ```bash
   LLAMA_API_KEY='LLM|...' MODEL_ID='gpt-5-4-genai-dss4' \
     python3 /home/hhzhang01/metagen_probe.py
   ```
   Expect `OK -> pipeline ok` on the `compat/v1 (chat.completions)` line;
   the other three targets (experimental passthrough and /v1 responses) fail
   from the login pod because they need corp VPN or a different key surface.
4. Export `LLAMA_API_KEY="LLM|..."` in the shell you launch training from.

Override the judge models by setting `BCPLUS_JUDGE_MODEL` /
`BCPLUS_JUDGE_FALLBACK_MODEL` / `BCPLUS_JUDGE_BASE_URL` if you want to swap
in a different MetaGen model or point at a local vLLM judge instead.

### 3. Model checkpoints on Lustre

- `Qwen/Qwen3.5-4B` HF snapshot at `/genai/fsx-project/hhzhang01/models/Qwen3.5-4B/`
- The corresponding `torch_dist` mcore checkpoint at
  `/genai/fsx-project/hhzhang01/models/Qwen3.5-4B_torch_dist/`. Run once with
  `tools/convert_hf_to_torch_dist.py` inside the slime container:
  ```bash
  PYTHONPATH=/root/Megatron-LM python /slime/tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /genai_hh/models/Qwen3.5-4B \
    --save /genai_hh/models/Qwen3.5-4B_torch_dist
  ```
  where `MODEL_ARGS` comes from `source /slime/scripts/models/qwen3.5-4B.sh`.

### 4. BC+ parquet on Lustre

Already staged: `/genai/fsx-project/hhzhang01/datasets/BC+/{bc_train,bc_test}.parquet`
(680 train + 150 test rows).

## Run the smoke test

**One-button** — after Steps 1–4 in Prerequisites are done:
```bash
export LLAMA_API_KEY="LLM|..."
bash /home/hhzhang01/slime/examples/supo_browsecomp/launch_all.sh
```
This ensures the search server is up (with enough runway), submits the
training srun, and syncs wandb offline runs after srun exits. Tune
`TRAIN_WALLTIME_HOURS` (default 24) and `SEARCH_BUFFER_HOURS` (default 4).

**Manual** — if you're already sure the search server is healthy:
```bash
export LOCAL_SEARCH_URL="http://<search-node>:8000"
export LLAMA_API_KEY="LLM|..."
bash /home/hhzhang01/slime/examples/supo_browsecomp/run_qwen3p5_4B.sh
# After srun exits:
bash /home/hhzhang01/slime/aws-cluster/wandb-sync.sh "${RUN_NAME}"
```

## What to expect from a healthy smoke run

- SGLang and actor colocated on 8 GPUs (`--colocate`); actor tp=2.
- wandb (offline) records non-zero `reward/mean` within the first few
  iterations — most rollouts should call `search` at least once and then
  `finish` with an answer that occasionally matches the gold.
- Search-server access log shows one burst of `/search` requests per rollout
  batch, then a quiet period during the PPO step.
- Judge calls fire once per finished trajectory. Check the search-server node
  can also reach `api.llama.com` (it should — that's the whole point of the
  internal Llama API endpoint).

## Not (yet) ported from the SUPO paper

- Branch / summary / multi-session tool (`workflow="search_branch"`).
- FoldGRPO advantage estimator.
- `process_reward=[flat,scope]` shaping and `scope_judge` sub-agent.
- Multi-question `<q1>...<q2>...` prompts appear in the eval set — the judge
  handles them but the rollout loop does not do the "you must answer all"
  reminder yet.

## Scaling from smoke to real runs

- **9B move**: `source scripts/models/qwen3.5-9B.sh` instead of `qwen3.5-4B.sh`;
  bump `--tensor-model-parallel-size` to 4 and `--max-tokens-per-gpu` to
  ~24576 depending on how OOM-y the actor gets.
- **Longer rollouts**: raise `BCPLUS_MAX_TURNS` from 5 → 20 → 100 (paper uses
  100 for the branch variant, less for pure ReAct). Also raise
  `--rollout-max-response-len` past 8k to accommodate.
- **More concurrency to the search server**: `BCPLUS_SEARCH_CONCURRENCY`
  gates the async semaphore; the vendored server has 20k-slot queues so it
  handles 100s of concurrent requests fine.

## Layout / conventions in the code

- `sample.prompt` comes straight from the parquet `prompt` column (a list of
  chat messages already tool-decorated). `--apply-chat-template` renders it to
  a single string before rollout.
- `sample.label` is the parquet `answer` column; `sample.metadata` is the
  `extra_info` dict (`query`, `answer`, `evidence_docs`, ...).
- Per-rollout stats (search count, finished, turns used) are stashed on
  `sample.metadata["_bcplus"]` so `reward_func` and slime's default wandb
  logger can pick them up.
