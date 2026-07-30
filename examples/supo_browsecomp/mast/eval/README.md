# BC+ checkpoint evaluation

This pipeline evaluates every completed Megatron checkpoint from one or more
MAST training runs. It runs a shared base point once, validates every
checkpoint load, monitors the search service, and writes one report per run.

The two files in this directory have distinct roles:

- `eval_sweep.py` is the devserver controller. It discovers checkpoints,
  freezes batch state, archives code, submits jobs, monitors them, and builds
  reports.
- `run_eval.sh` is the immutable single-point container entrypoint executed on
  one MAST host. It starts slime, writes the raw dump, and calls the common
  analyzer in `examples/supo_browsecomp/eval/eval_pipeline.py`.

The end-to-end completion chain is:

```text
checkpoint discovery -> frozen sweep_config.json -> code archive
  -> one MAST job per point -> raw dump + eval.log
  -> point validation + _SUCCESS -> per-run report -> local report copy
```

## Current runtime profile

| Setting | 4B | 9B |
|---|---:|---:|
| MAST host | 1 x `zionex_80g` | 1 x `zionex_80g` |
| GPUs | 8 | 8 |
| Megatron TP / CP | 4 / 2 | 4 / 2 |
| SGLang engines | 8 x TP1 | 8 x TP1 |
| SGLang memory fraction | 0.8 | 0.8 |
| SGLang server concurrency per engine | 36 | 32 |
| SGLang router request timeout | 5400 seconds | 5400 seconds |
| NCCL timeout debug dump | disabled | disabled |
| Max response / context | 32768 / 65536 | 32768 / 65536 |
| Questions x samples | 150 x 4 | 150 x 4 |
| Search / judge concurrency | 64 / 16 | 64 / 16 |

These defaults are frozen into the submitted code archive. Search concurrency,
judge concurrency, question count, sample count, and seed are controller
options. The remaining runtime defaults are currently owned by `run_eval.sh`.
The model-specific SGLang concurrency leaves headroom above the observed
long-context running set without restoring the effectively unbounded historical
load. It can be overridden explicitly with
`BCPLUS_SGLANG_SERVER_CONCURRENCY` for a controlled experiment.

Eval jobs set `TORCH_NCCL_DUMP_ON_TIMEOUT=0`. Eight independent TP1 engines on
one host are all NCCL rank 0; leaving the timeout dump enabled lets them race
for the shared `/tmp/nccl_trace_rank0.pipe` and can abort one engine before
rollout starts. This disables only the detailed NCCL timeout dump. Normal NCCL
errors, watchdog behavior, SGLang request timeouts, and eval results are
unchanged.

The controller also recognizes the pre-reorganization archive path
`mast/run_eval.sh` when extending an older batch. It never rewrites that frozen
archive, so old and newly appended points keep the batch's original runtime
profile.

## Start a new run

Choose a unique batch id so the sweep can be resumed without rediscovery:

```bash
RUN=<mast-training-run-name>
BATCH=bcplus-4b-eval-$(date +%Y%m%d-%H%M%S)

python examples/supo_browsecomp/mast/eval/eval_sweep.py orchestrate \
  --batch-id "$BATCH" \
  --run "$RUN"
```

Repeat `--run` to evaluate multiple runs in one batch. Each run still receives
an independent report; the base point is shared.

The controller infers `4B` or `9B` from standard run names. It can also be set
explicitly, which is recommended for a new 9B batch:

```bash
python examples/supo_browsecomp/mast/eval/eval_sweep.py orchestrate \
  --batch-id "$BATCH" \
  --model-size 9B \
  --run "$RUN"
```

A batch cannot mix model sizes because its base result is shared. A 9B run uses
the Qwen3.5-9B HF and torch-distributed checkpoints and evaluates a separate 9B
base; it never reuses a 4B base.

## Resume training and keep the original report

A resumed MAST job normally has a new MAST job name but retains the original
`BC_RUN_NAME`. In that case it writes new checkpoints into the original
checkpoint root, and evaluation should extend the original eval batch rather
than start a new one:

```bash
OLD_BATCH=<batch-that-produced-the-existing-report>

python examples/supo_browsecomp/mast/eval/eval_sweep.py orchestrate \
  --batch-id "$OLD_BATCH" \
  --extend-checkpoints
```

Do not pass `--run` or `--reuse-base-from` when extending an existing batch.
Its frozen `sweep_config.json` already owns the run names, base source,
evaluation settings, and code archive. Existing `_SUCCESS` points remain in
place, only newly discovered complete checkpoints are submitted, and the
report is regenerated at the same cloud and local report paths.

Check `sweep_config.json` before extending. The controller rediscovers every
run configured in the batch. A single-run batch therefore extends only the
resumed run. For a multi-run batch, use the same command only when new complete
checkpoints from all configured runs may be appended. The controller currently
has no per-run extension selector; use a separate batch when one run must be
isolated rather than manually editing the frozen configuration.

## Extend an unfinished training run

The first invocation freezes every checkpoint that is complete at that time.
After training writes more checkpoints, explicitly extend the same batch:

```bash
python examples/supo_browsecomp/mast/eval/eval_sweep.py orchestrate \
  --batch-id "$BATCH" \
  --extend-checkpoints
```

The controller rediscovers all configured runs, appends only new complete
steps, and atomically updates `sweep_config.json`. Existing `_SUCCESS` outputs
and MAST job records are preserved, so only new steps are submitted. Reports
are regenerated over the combined old and new results. The original code
archive, base result, seeds, and evaluation settings remain unchanged.

Run this after the target `iter_*/.metadata` appears. A tracker that points
beyond the latest complete metadata is rejected. A lagging tracker is allowed
because every eval passes `--ckpt-step` explicitly, but the lag is logged.
Running `orchestrate --batch-id "$BATCH"` without `--extend-checkpoints` only
resumes the already frozen set and never discovers additional steps.

## Reuse an existing base

For later runs using the same evaluation protocol, point to the earlier eval
batch that actually contains the completed `base` directory:

```bash
python examples/supo_browsecomp/mast/eval/eval_sweep.py orchestrate \
  --batch-id "$BATCH" \
  --reuse-base-from bcplus-4b-eval-20260727-v2 \
  --run "$RUN"
```

No base MAST job is submitted. The source batch is frozen in
`sweep_config.json`. Before accepting it, the controller validates its artifact
counts, deterministic seeds, and release-checkpoint evidence. After the new
checkpoint jobs finish, it also requires the dataset hash, judge model, and
complete sampling/rollout configuration to match the reused base. A protocol
change therefore fails explicitly instead of producing an invalid comparison.

The controller discovers every `iter_*/.metadata` directory under
`/data/users/hhzhang01/wsfuse_mnt/hhzhang01/supo-slime/checkpoints/<run>`.
The tracker `latest_checkpointed_iteration.txt` must exist and cannot point
beyond the latest complete checkpoint. The discovered list is frozen in
`<batch>/sweep_config.json`, so a
resumed sweep cannot silently gain or lose points. Only the explicit extension
command can append points; it never removes a previously recorded step.

## What the controller does

1. Archives the current repository and records its SHA-256.
2. Evaluates base once, or validates and reuses the explicitly selected base.
3. Checks search-server health once, then submits every checkpoint job without
   a smoke gate, waves, or an active-job cap.
4. Monitors all submitted MAST jobs and search-server stats until completion.
5. For each point, records raw rollout data, strict `score == 1` metrics,
   question-level results, summary behavior, checkpoint metadata SHA-256, and
   the exact load-log evidence.
6. Builds one Markdown/CSV/JSON/SVG report set per run and copies those small
   artifacts to the local report directory.

Defaults are 150 questions, deterministic seeds 42-45, search concurrency 64,
and judge concurrency 16. Override them with `--expected-questions`, `--eval-n`,
`--eval-seed`, `--search-concurrency`, or `--judge-concurrency` when needed.

Each checkpoint uses one 8-GPU `zionex_80g` host. Megatron loads the distributed
checkpoint with TP4/CP2 and is offloaded during rollout. For both 4B and 9B,
SGLang uses eight TP1 engines, one per GPU, with
`--sglang-mem-fraction-static 0.8`.

## Weights-only intermediate checkpoints

Intermediate checkpoints may omit optimizer, scheduler, and RNG state. The
eval runner always supplies `--no-load-optim --no-load-rng`, and checkpoint
discovery requires only the run tracker plus each `iter_*/.metadata`. A slim
checkpoint must retain its original `iter_XXXXXXX` directory name and saved
iteration so that `--load <run> --ckpt-step N` and load-log verification still
select the requested model weights.

Use the self-contained `mast/checkpoint_slim/` workflow to rewrite an existing
completed checkpoint through Megatron's native load/save APIs. The tool stages
a new torch_dist checkpoint, compares a distributed hash of all model
parameters and buffers, reloads it at the original iteration, and only then
replaces the full checkpoint. Conversion manifests preserve the old and new
metadata hashes for provenance. Do not delete individual `.distcp` files:
model and optimizer records are normally mixed in the same storage files.

For the first checkpoint in a cleanup batch,
`mast/checkpoint_slim/run_checkpoint_slim_canary_eval.sh` exposes only the
staged directory through a temporary checkpoint-root symlink, runs the
unchanged eval runner, and promotes the checkpoint only after eval succeeds.
The trap removes the temporary symlink on both success and failure, and never
removes a non-symlink path. See `mast/checkpoint_slim/README.md` for the full
planning, MAST execution, canary, promotion, verification, and recovery
procedure.

Weights-only checkpoints are suitable for eval but not exact training resume.
The slimming tool automatically protects the numerically latest complete
checkpoint and requires the run tracker to agree with it; that checkpoint
keeps its optimizer and scheduler state regardless of its iteration number.
New training runs may enable `--slim-intermediate-checkpoints` (mapped from
`BC_SLIM_INTERMEDIATE_CHECKPOINTS=1` by the BC+ MAST runner) to maintain this
layout during training while always retaining the latest full resume point.

## Resume and inspect

The batch configuration and submitted MAST jobs are persistent:

```bash
python examples/supo_browsecomp/mast/eval/eval_sweep.py status --batch-id "$BATCH"

python examples/supo_browsecomp/mast/eval/eval_sweep.py orchestrate \
  --batch-id "$BATCH"
```

Only one controller command may access a batch at a time. A second controller
fails immediately on `.controller.lock`, preventing stale in-memory state from
overwriting newer job records. The lock is released automatically when the
controller exits or is interrupted.

To regenerate reports without rerunning evaluation:

```bash
python examples/supo_browsecomp/mast/eval/eval_sweep.py report --batch-id "$BATCH"
```

`report` uses all completed point directories currently present, so it can
produce an interim report for a partially evaluated single-run batch. It
requires a completed base and at least one completed trained checkpoint. A
normal `orchestrate` invocation generates the final report automatically only
after every point frozen in the batch has `_SUCCESS`.

Cloud results are stored under
`/data/users/hhzhang01/wsfuse_mnt/hhzhang01/supo-slime/evals/<batch>`.
Report artifacts are also copied to
`/home/hhzhang01/bcplus-eval-reports/<batch>` by default. Raw `.pt` rollout
dumps remain only in cloud storage.

The cloud layout is:

```text
<batch>/
  sweep_config.json
  sweep_state.json
  code.json
  base/
  runs/<run>/iterNN/
  runs/<run>/report.md
  runs/<run>/checkpoint_metrics.csv
  runs/<run>/question_changes.csv
  runs/<run>/metrics.json
  runs/<run>/*.svg
```

Each completed point contains `manifest.json`, `point_metrics.json`,
`rollouts.jsonl`, `questions.jsonl`, `eval.log`, the raw dump, and `_SUCCESS`.
See [`../../eval/README.md`](../../eval/README.md) for the artifact and metric
contract.

## Failure handling

- Rerunning `orchestrate --batch-id ...` resumes the frozen point set and does
  not resubmit points with `_SUCCESS` or an existing MAST job record.
- Use `--extend-checkpoints` only to append newly completed training steps.
- Use `retry-point --batch-id ... --key RUN/iterNN` only after the recorded MAST
  job is `FAILED` or `DEAD` and the output directory has no raw dump.
- Any raw dump without `_SUCCESS` requires manual audit: automatic MAST retries
  and `retry-point` intentionally refuse to overwrite it. Preserve it for
  diagnosis, then move it aside before an explicit retry.
- Always verify the point manifest's `load_verification.actual_step`; similar
  metrics across checkpoints are not evidence that loading was correct.

## Report metrics

- `pass@1`: rollout accuracy, using only `score == 1` as correct.
- `pass@N`: fraction of questions with at least one correct result among N seeds.
- `N/N`: fraction of questions answered correctly by every seed.
- Curves: `accuracy_curve.svg` for pass@1 and `pass_at_n_curve.svg` for
  pass@N (pass@4 with the default four seeds).
- Behavior: finish rate, turns, searches, opens, sub-trajectories, response
  tokens, bad tool calls, compression rate, and search-server errors.
- Summary behavior: extracted/fallback/empty rates and extracted-content token
  length statistics. Fallback and empty summaries are excluded from lengths.
- Reliability: judge failures, actual loaded iteration, checkpoint metadata
  hash, code hash, dataset hash, MAST job, and deterministic sampling seeds.

The report's "best observed checkpoint" is selected on the test set and should
not be treated as an unbiased model-selection estimate.
