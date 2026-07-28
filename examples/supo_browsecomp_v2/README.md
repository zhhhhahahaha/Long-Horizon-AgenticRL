# supo_browsecomp_v2

Development fork of [`examples/supo_browsecomp`](../supo_browsecomp/). The
original is left **byte-for-byte untouched** — v2 exists to iterate on the
retrieval-tool configuration without disturbing the baseline, so the two can run
side by side for comparison.

For full setup (search server, data staging, judge/`LLAMA_API_KEY`, checkpoints,
wandb, the reward/compression/dump design), read
[`../supo_browsecomp/README.md`](../supo_browsecomp/README.md). This file only
documents what v2 changes.

## What changed vs. `supo_browsecomp`

Two edits to the `search` / `open_page` tool configuration, both aimed at the
context-budget and evidence-truncation problems observed in baseline rollouts:

1. **`search` returns a fixed top-5; `topk` is no longer model-settable.**
   - Baseline exposed a `topk` parameter (default 10, cap 20) the model could
     pass. RL pushed it upward, so each `search` observation ballooned and ate
     the context window (a single trajectory reached ~48k tokens with only ~2k
     model-generated — the rest was tool observation).
   - v2 removes `topk` from the tool schema entirely and hardcodes top-5,
     matching the official BrowseComp-Plus eval setup and removing the incentive
     to inflate `topk`.
   - All other search-side logic is unchanged: still fetches `k=50` upstream,
     still applies the visited-dedup `0.25` increment and the per-doc snippet
     truncation (`doc_words_snippet=512`). A stray `topk` arg from the model is
     silently ignored.

2. **`open_page` full-text cap raised 4096 → 10000 words.**
   - BrowseComp-Plus docs average ~5179 words, so the baseline 4096-word cap
     dropped the back half of long documents — "retrieved the right doc but
     answered wrong." 10k words fits the large majority of docs whole.

Concretely, the diffs live in:
- `tool_schemas.py` — `SEARCH_SCHEMA` drops the `topk` property; description says
  "top 5".
- `generate_with_bcplus.py` — `BCPLUS_CONFIGS`: `search_topk_default`/
  `search_topk_cap` replaced by a single `search_topk: 5`; `doc_words_full`
  `4096 → 10000`. The `search` branch of `_run_action` no longer parses `topk`.

No data/parquet change is required: tool schemas are injected at rollout time via
`apply_chat_template(sample.prompt, tools=TOOLS)`, and the parquet carries only
the generic system blurb + the question (no tool-parameter text). See
`../supo_browsecomp/scripts/migrate_parquet_to_qwen_tool_format.py`.

## Files in this folder

Copied from `supo_browsecomp` (relative imports must resolve inside this package):
- `generate_with_bcplus.py` — the 3 edits above.
- `tool_schemas.py` — the `SEARCH_SCHEMA` edit.
- `local_search_client.py` — **verbatim copy, no changes.** Present only because
  `generate_with_bcplus.py` does `from .local_search_client import
  AsyncSearchClient`; it has no topk logic.
- `run_qwen3p5_4B_colocate.sh` — copy with the rollout/reward hook module paths
  and the in-container self-relaunch pointed at `supo_browsecomp_v2`. The
  search-server launch line still points at `supo_browsecomp` (shared server).

**Reused from `supo_browsecomp` by absolute path (not copied, not modified):**
`search_server.py`, `launch_search_server.sh`, the BC+ corpus/embeddings/data
parquets, the judge config, `scripts/`, and the 9B scripts.

## Launching

Same workflow as the baseline — start the (shared) search server, then run the
v2 script from inside tmux on the login pod:

```bash
# 1. shared search server (reused from supo_browsecomp; skip if already up)
bash examples/supo_browsecomp/launch_search_server.sh

# 2. v2 training (auto-discovers the server, submits the 8-node srun)
bash examples/supo_browsecomp_v2/run_qwen3p5_4B_colocate.sh
```

`RUN_NAME` defaults to `supo-bcplus-v2-qwen3p5-4b-<date>` so v2 runs are
distinguishable from baseline in wandb/logs. All other env-var overrides
(`BC_*`, `BCPLUS_*`) behave exactly as documented in the baseline README.

## Note on the larger `open_page`

A full 10k-word `open_page` observation is ~13-15k tokens. This is safely
absorbed by the existing projected-size compression/rollback logic in
`generate()` (`generate_with_bcplus.py`, the per-turn `projected >
compress_threshold_tokens` check): if committing the observation would exceed the
budget, the turn is rolled back and compression fires — no crash. Expected
consequence: trajectories that open several long documents may hit compression a
bit earlier than under the baseline. This is intended behavior, not a regression.
