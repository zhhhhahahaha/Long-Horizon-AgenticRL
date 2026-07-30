# SUPO — BrowseComp-Plus pipeline for slime

Port of the SUPO paper's ([arXiv 2510.11967](https://arxiv.org/abs/2510.11967))
BrowseComp-Plus training loop onto slime. Self-contained under this directory —
no runtime dependency on the SUPO / FoldAgent repo (everything we need is
either ported or vendored here).

This is a **pipeline-first** port:

- Rollout: ReAct search with automatic SUPO session folding when a sub-
  trajectory approaches its context budget. Branching and process reward are
  not implemented.
- Algorithm: rollout-id-deduped FoldGRPO normalization across compression
  siblings, plus token-level advantage control for the generated summary turn.
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
- `generate_with_bcplus.py` — rollout, reward, post-processing, and W&B hooks:
  - `generate()` — multi-turn ReAct loop. Stops the sampler at `</function>`,
    parses the last `<function=...>` block, dispatches to `search`/`open_page`/
    `finish`, and appends the observation as `loss_mask=0` text before the next
    turn. Trainable tokens (model output) use `loss_mask=1`.
  - `reward_func()` — extracts the `<finish answer=...>` payload from the
    trajectory metadata and calls the MetaGen judge (with `em_score` /
    `relaxed_em` fast paths).
- `summary_advantage.py` — expands each rollout's normalized GRPO signal over
  response tokens and assigns a fixed negative advantage only to malformed
  compression-generation turns (thinking plus summary output).
- `aws/run_qwen3p5_4B_colocate.sh` — the live launcher for the full 8-node run
  (64 GPU, colocate: sglang shares the training GPUs in a single srun).
  Two-part (login pod → in-container). Auto-ensures the search server (see
  below), then submits training. Training QOS defaults to
  `a100_genai_interns_high`; the search server runs on `SEARCH_QOS` (default
  `a100_dev`). `aws/run_qwen3p5_9B_colocate.sh` is the 9B sibling. For a quick
  1-node dump smoke, see `debug_scripts/run_qwen3p5_4B_1node_dumpsmoke.sh`.
- `aws/search/launch_search_server.sh` — idempotent orchestrator that sbatch's a
  long-lived (7-day, 1 GPU) retrieval server on `a100_dev`, waits for
  `/health`, and writes the resolved `host:port` to the hostname file. Reuses
  an existing job if its remaining walltime is `>= MIN_HOURS_REMAINING`
  (default 48h); otherwise scancels + resubmits so a training job doesn't
  outlive its search server. `aws/run_qwen3p5_4B_colocate.sh` calls this
  automatically (you can also run it standalone).

## Operations and evaluation

- [`aws/README.md`](aws/README.md) indexes the AWS/Slurm launchers: the 8-node
  colocate trainers, the search-server orchestrator, and the one-shot HF→mcore
  conversion.
- [`mast/README.md`](mast/README.md) indexes the current MAST training, search,
  checkpoint-evaluation, checkpoint-slimming, and W&B workflows.
- [`mast/eval/README.md`](mast/eval/README.md) is the operational runbook for
  evaluating complete or still-running MAST training runs, reusing base,
  extending a batch with new checkpoints, monitoring jobs, and collecting
  reports locally.
- [`eval/README.md`](eval/README.md) defines the scheduler-independent metric,
  artifact, summary-fallback, and checkpoint-load contracts.
- [`eval/legacy/README.md`](eval/legacy/README.md) documents the retained older
  Slurm and research-analysis pipeline. It is not the official metric path.

## Search and open-page variants

The canonical rollout keeps the original SUPO behavior by default: `search`
lets the model choose `topk` (default 10, capped at 20), and `open_page` returns
at most 4096 words. Two independent environment variables select experimental
variants without forking the rollout implementation:

| Variable | Unset (baseline) | Example override |
|---|---|---|
| `BCPLUS_FIXED_SEARCH_TOPK` | Model-controlled topk | `5` removes topk from the tool schema and forces top-5 |
| `BCPLUS_DOC_WORDS_FULL` | `4096` | `10000` raises the open-page full-text cap |

For the combined fixed-top5/open-10000 variant on AWS:

```bash
BCPLUS_FIXED_SEARCH_TOPK=5 \
BCPLUS_DOC_WORDS_FULL=10000 \
bash examples/supo_browsecomp/aws/run_qwen3p5_4B_colocate.sh
```

Either variable can be set alone. The 4B AWS launcher includes the effective
combination in its default `RUN_NAME`, so checkpoints, dumps, logs, Ray
coordination files, and W&B runs stay separate. An explicit `RUN_NAME` still
overrides that generated name.

## Training trajectory artifacts

The AWS launcher writes a per-DP-rank parquet dump by default. It contains the
token IDs, loss masks, rollout log probabilities, rewards, and advantages used
by training. For complete research trajectories in the same raw format as
evaluation's `eval_0.pt`, also set `BCPLUS_RAW_ROLLOUT_DIR`:

```bash
RUN_NAME=supo-bcplus-fixedtopk5-open10000w-run \
BCPLUS_DUMP_DIR="" \
BCPLUS_RAW_ROLLOUT_DIR=/genai/fsx-project/hhzhang01/raw-rollouts/supo-bcplus-fixedtopk5-open10000w-run \
bash examples/supo_browsecomp/aws/run_qwen3p5_4B_colocate.sh
```

This writes one `rollout_<iteration>.pt` file per training iteration. Each file
contains full `Sample` records: decoded prompt and response, gold label, reward,
tool/compression metadata, tokens, and loss masks. The sibling chains can be
analyzed with the same helpers used by
[`eval/eval_pipeline.py`](eval/eval_pipeline.py), especially
`analyze_samples`. Keep the parquet dump enabled when logprob or advantage
auditing is required. Set `BCPLUS_DUMP_DIR=""` as above when the native `.pt`
trajectory is sufficient and the extra parquet is not needed.

## Prerequisites

### 1. Search server (biggest external dep)

The retrieval corpus + embeddings need ~4 GB and the embedding model ~8 GB. One
80GB GPU is plenty for BC+ (corpus is ~830K docs).

**Preferred (scripted, idempotent)** — from the login pod:
```bash
bash /home/hhzhang01/slime/examples/supo_browsecomp/aws/search/launch_search_server.sh
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

## Run training (8-node colocate)

**One command** — after the Prerequisites are done:
```bash
export LLAMA_API_KEY="LLM|..."
bash /home/hhzhang01/slime/examples/supo_browsecomp/aws/run_qwen3p5_4B_colocate.sh
```
This auto-ensures the search server is up (with enough runway, on `a100_dev`),
submits the 8-node training srun (QOS `a100_genai_interns_high`), and syncs
wandb offline runs while/after srun runs. Tune `TRAIN_WALLTIME` and
`SEARCH_BUFFER_HOURS` (default 4); `SEARCH_QOS` (default `a100_dev`) sets the
search server's queue.

### MAST W&B live sync

For config-defined trainer jobs, keep the runner fixed and create one shell
config per experiment under `examples/supo_browsecomp/mast/configs/`. The
submission entry point always performs a MAST dry-run first and checks both the
trainer `taskCount` and `ROLE_ASSIGNMENT_MAP` before submitting:

```bash
# Validate only: must report taskCount=16 and trainer_0=128.
examples/supo_browsecomp/mast/submit_experiment.sh --dry-run \
  examples/supo_browsecomp/mast/configs/4b_16node_dynamic_n16.sh

# Validate again, submit, then start the W&B watcher.
examples/supo_browsecomp/mast/submit_experiment.sh \
  examples/supo_browsecomp/mast/configs/4b_16node_dynamic_n16.sh
```

The example preserves the existing 4B run's `BC_N_SAMPLES=16`. With dynamic
sampling enabled, its first pool is 64 prompt groups, or 1024 trajectories;
`BCPLUS_SEARCH_CONCURRENCY=512` bounds active client requests rather than the
number of generated trajectories. Refresh the configured `MAST_CODE_ARCHIVE`
before submission whenever trainer code changes.

MAST writes W&B transaction logs to each node's local `/tmp` and publishes an
immutable tar snapshot to `supo-slime/wandb-snapshots/<job>/` every 60 seconds.
It does not write active `.wandb` files directly to OILFS. From the devserver,
wrap the existing JSON-mode MAST command to submit the job and start its W&B
watcher atomically from the user's point of view:

```bash
cd /home/hhzhang01/Long-Horizon-AgenticRL
examples/supo_browsecomp/mast/submit_with_wandb.sh -- \
  /home/hhzhang01/local/fbsource/genai/msl/rl/cli.sh mast --json \
  ...the existing MAST arguments...
```

The wrapper passes the submission command through unchanged, parses
`.job.job_name` from its structured response, and starts the watcher in tmux.
It saves the response and watcher log under
`~/.local/state/mast-wandb/<job>/`. If a devserver reboot removes the tmux
session, restore only the watcher without submitting another training job:

```bash
examples/supo_browsecomp/mast/submit_with_wandb.sh \
  watch-only avocado_rev1_rl_debug_80m-xxxxxxxx
```

The wrapper refuses commands without `--json`. If submission succeeds but the
watcher fails to start, it reports the already-submitted job and exits with
status 3; use `watch-only` to recover rather than rerunning the submission.

The watcher extracts only completed snapshots to devserver-local cache, then
uploads to `https://meta.wandb.io` every five minutes with `wandb sync
--append`. It survives MAST preemption/rescheduling and performs a final sync
after MAST reaches `COMPLETE`, `FAILED`, or `DEAD`. For an on-demand upload,
replace `watch` with `once`. The Meta W&B key remains in `~/.wandb-key` on the
devserver and is never copied into MAST or OILFS. Set
`MAST_WANDB_SNAPSHOT_INTERVAL_SEC` on the trainer to change the 60-second
snapshot interval.

Before pre-creating a remote run, the watcher validates the offline log's run
ID, project, group, and entity against `MAST_WANDB_PROJECT`, the MAST job name,
and `MAST_WANDB_ENTITY`. When overriding the trainer's `BC_WANDB_PROJECT`, set
`MAST_WANDB_PROJECT` to the same value; a mismatch fails before upload instead
of creating a run in the wrong project.

At shutdown, the head task also copies Ray text logs to OILFS. This cleanup is
limited to 120 seconds so a large log directory cannot keep GPUs allocated after
training has finished. Set `MAST_RAY_LOG_COPY_TIMEOUT_SEC` to change that limit,
or `MAST_PERSIST_RAY_LOGS=0` to skip Ray log persistence. The final W&B snapshot
is published before this cleanup starts.

**Point at an existing search server** — set `LOCAL_SEARCH_URL` to skip the
ensure step entirely:
```bash
export LOCAL_SEARCH_URL="http://<search-node>:8000"
export LLAMA_API_KEY="LLM|..."
bash /home/hhzhang01/slime/examples/supo_browsecomp/aws/run_qwen3p5_4B_colocate.sh
```

**1-node dump smoke** — quick end-to-end check (dump on, train_old off):
```bash
export LLAMA_API_KEY="LLM|..."
bash /home/hhzhang01/slime/examples/supo_browsecomp/debug_scripts/run_qwen3p5_4B_1node_dumpsmoke.sh
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

- Search branching / model-selected compression (`workflow="search_branch"`).
- `process_reward=[flat,scope]` shaping and `scope_judge` sub-agent.
- Multi-question `<q1>...<q2>...` prompts appear in the eval set — the judge
  handles them but the rollout loop does not do the "you must answer all"
  reminder yet.

## Scaling from smoke to real runs

- **9B move**: use `aws/run_qwen3p5_9B_colocate.sh`. Its candidate starting topology
  is Megatron TP=4/CP=2, SGLang TP=4, and `--max-tokens-per-gpu=32768`; these
  knobs remain experimental because one oversized sub-traj can exceed the
  packing target. For MAST,
  set `BC_MODEL_SIZE=9B`; the trainer selects the matching model spec,
  checkpoints, and 9B memory defaults as one coherent configuration.
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
