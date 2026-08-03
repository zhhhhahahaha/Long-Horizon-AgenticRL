# BC+ training-data filter sweep

This directory evaluates the complete 680-row BC+ training set at three model
points and finds questions that are correct in every one of eight deterministic
samples at every point:

- Qwen3.5-4B base model
- `supo_4b_n8_8n_40iter_dump_groupfix-mj1d0qw1` checkpoint 4
- the same run at checkpoint 9

Each point is one MAST job on one `grandteton_80g_roce` host (8x H100 80GB).
SGLang runs eight TP1 engines with a request-side target of 36 concurrent
rollouts per engine (288 globally across the eight engines).
The retrieval server remains the independent job in tenant
`rhea_assistant_interns`; eval jobs read its current address from
`/mnt/wsfuse/hhzhang01/supo-slime/search-server.addr`.

## Frozen protocol

| Setting | Value |
|---|---:|
| Training questions | 680 |
| Samples per question and point | 8 |
| Rollout seeds | 42-49 |
| Temperature | 1.0 |
| Max response / context | 32768 / 65536 |
| SGLang engines / TP | 8 / 1 |
| Per-engine SGLang concurrency | 36 |
| Search / judge concurrency per job | 64 / 16 |
| MAST tenant | `rhea_assistant_lens` |
| MAST region / host | `eag` / `grandteton_80g_roce` |
| MAST priority | `CRITICAL` |

Correctness is strict `score == 1`. A filter candidate must therefore have 24
successful rollouts: 8/8 at base, 8/8 at checkpoint 4, and 8/8 at checkpoint 9.

## Submit

Use a unique batch id. The controller freezes the config and code archive in
OILFS, validates a MAST dry-run, checks the search server, and records each
real submission immediately.

```bash
BATCH=bcplus-train-filter-supo4b-ckpt4-9-20260730

python3 examples/supo_browsecomp/mast/train_data_filter/filter_sweep.py \
  dry-run --batch-id "${BATCH}"

python3 examples/supo_browsecomp/mast/train_data_filter/filter_sweep.py \
  submit --batch-id "${BATCH}"
```

Do not rerun a real MAST command manually. Resume through the controller, which
will skip every point already present in `state.json`.

## Inspect and aggregate

```bash
python3 examples/supo_browsecomp/mast/train_data_filter/filter_sweep.py \
  status --batch-id "${BATCH}"

python3 examples/supo_browsecomp/mast/train_data_filter/filter_sweep.py \
  finalize --batch-id "${BATCH}"
```

Results live under:

```text
/data/users/hhzhang01/wsfuse_mnt/hhzhang01/supo-slime/train-data-filter/<batch>/
  config.json
  state.json
  code.json
  dry_run.json
  points/{base,iter04,iter09}/
  filter_candidates.jsonl
  filter_candidate_query_ids.txt
  filter_summary.json
  filter_summary.md
```

The raw rollout dumps remain under each point for audit. `filter_candidates.jsonl`
contains the query text and the exact per-point success counts; use
`filter_candidate_query_ids.txt` as the key list for a later physical Parquet
rewrite after reviewing the candidates.

The source Parquet is never rewritten by this workflow. Its SHA-256 is frozen
in `config.json`; submission and aggregation fail if the source bytes change.
