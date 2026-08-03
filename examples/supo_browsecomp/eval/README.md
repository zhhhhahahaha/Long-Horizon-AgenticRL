# BrowseComp evaluation artifacts

This directory owns scheduler-independent evaluation semantics. It turns a
raw slime eval dump into validated point artifacts, then combines completed
points into one report for a training run.

MAST submission, checkpoint discovery, monitoring, and retries live under
[`../mast/eval/`](../mast/eval/README.md). Historical Slurm and research
scripts live under [`legacy/`](legacy/README.md) and are not part of the
current report pipeline.

## Active implementation

`eval_pipeline.py` has two commands:

- `point` validates and summarizes one base or checkpoint eval.
- `report` validates protocol compatibility and builds a per-run report from
  all completed points currently present on disk.

The supported optional [`analysis/summary_retention.py`](analysis/README.md)
pipeline stages failed rollouts whose earlier tool observations plausibly
contained the answer, semantically judges whether the final handover retained
it, and produces deterministic checkpoint-level retention metrics. It remains
separate from accuracy because it is conditional, judge-based failure analysis.

The MAST point runner invokes `point` automatically after slime writes
`rollout_data/eval_0.pt`. The MAST controller invokes `report` after all jobs
in the configured batch complete. These commands can also be run directly for
artifact recovery; see the MAST eval runbook for the supported entry points.

## Evaluation invariants

The active analyzer enforces the following rules:

1. Every parent rollout has exactly one final compression sibling and a
   contiguous sibling chain.
2. The score and final answer come from that final sibling, not the maximum
   score observed anywhere in the chain.
3. A rollout is correct only when `score == 1.0`.
4. Every question has exactly the configured number of deterministic samples.
5. Judge failures are counted separately rather than silently treated as an
   ordinary model error.
6. A compression attempt is classified as `extracted`, `fallback`, or `empty`
   using the metadata written by `generate_with_bcplus.py`.
7. A missing or empty `<summary>...</summary>` body uses the rollout fallback
   when salvageable text exists. It is `empty` only when no usable text can be
   recovered.
8. Summary length statistics include only `extracted` summary content. Fallback
   and empty summaries are excluded.
9. A trained checkpoint point is accepted only when its log explicitly
   confirms the requested `at iteration N` load. Base requires release-load
   evidence.

## Point artifacts

Each successfully analyzed point contains:

| Artifact | Purpose |
|---|---|
| `rollout_data/eval_0.pt` | Raw slime dump with trajectories and metadata |
| `eval.log` | Complete runner log and checkpoint-load evidence |
| `manifest.json` | Model, checkpoint, hashes, sampling protocol, and MAST job |
| `point_metrics.json` | Aggregate accuracy, reliability, behavior, and summary metrics |
| `rollouts.jsonl` | One normalized record per parent rollout |
| `questions.jsonl` | Per-question success counts across deterministic seeds |
| `_SUCCESS` | Written last, after every validation above succeeds |

`_SUCCESS` is the completion contract used by the MAST controller. A raw dump
without `_SUCCESS` is incomplete and must not be included in a report.

## Per-run report artifacts

The `report` command produces:

| Artifact | Purpose |
|---|---|
| `report.md` | Human-readable checkpoint report |
| `checkpoint_metrics.csv` | One metrics row per base/checkpoint point |
| `question_changes.csv` | Per-question success-count changes from base |
| `metrics.json` | Machine-readable report data and provenance |
| `accuracy_curve.svg` | pass@1 across checkpoints |
| `pass_at_n_curve.svg` | pass@N across checkpoints |

`pass@1` is rollout accuracy. `pass@N` is the fraction of questions with at
least one correct result among N samples. `N/N` is the fraction answered
correctly by every sample.

Before combining points, the report verifies that base and trained checkpoints
use the same model family, dataset hash, judge model, and complete sampling
configuration.

## Tests

The CPU contract tests cover strict scoring, sibling reconstruction,
checkpoint-load verification, report curves, sweep state, base reuse,
checkpoint slimming, and the summary-retention candidate/judge/report contract:

```bash
pytest -q tests/test_bcplus_eval_pipeline.py tests/test_bcplus_summary_retention.py
```

## Legacy code

The scripts in `legacy/` were retained because their deeper trajectory
analyses may be useful again. They are intentionally isolated: several use a
historical `score >= 0.5` definition, maximum-sibling scoring, old output
layouts, and Slurm-specific paths. Do not use their summary tables as official
metrics for the current MAST pipeline without porting the relevant analysis to
the contracts above.
