# Legacy evaluation tools

These files are preserved from the earlier Slurm-based BrowseComp evaluation
workflow. They are useful references for deeper failure, grounding, retrieval,
and compression investigations, but they are not called by the current MAST
checkpoint sweep.

## Layout

- `slurm/` contains the old one-checkpoint launcher, sweep launcher, and
  in-container report launcher.
- `analysis/` contains the Stage-A report builder, trajectory diagnostics,
  staging utilities, and Stage-B agent workflows.

## Historical workflow

The original flow was:

1. `slurm/eval_all_checkpoints.sh` submitted one Slurm job per point.
2. `slurm/run_qwen3p5_4B_eval.sh` reused the colocated 4B training launcher and
   wrote `rollout_data/eval_0.pt`.
3. `slurm/run_report.sh` ran `analysis/build_eval_report.py` inside enroot.
4. The remaining scripts consumed the Stage-A outputs for focused research
   analyses and Stage-B agent review.

The scripts still contain historical `/genai/fsx-project/...`, enroot, Slurm,
and 4B-specific assumptions. Their paths and comments have been updated after
the move, but their evaluation behavior has deliberately not been rewritten.

## Compatibility warning

This code does not implement the current official metric contract:

- `analysis/build_eval_report.py` and several downstream scripts use
  `score >= 0.5` as correct instead of strict `score == 1.0`.
- Some analyses take the maximum score across compression siblings instead of
  the final sibling's score.
- The expected directory layout predates shared/reused base artifacts.
- Judge-failure and summary-fallback semantics predate the current metadata
  contract.

Use the active [`../eval_pipeline.py`](../eval_pipeline.py) for official point
metrics and reports. When one of these research analyses is needed again, port
it to `rollouts.jsonl`, `questions.jsonl`, or the current raw dump contract and
place the supported version in a new `eval/analysis/` directory.

The summary-retention study has now been ported. Use the supported
[`../analysis/deepdive.py`](../analysis/README.md) run-level pipeline (which
invokes `summary_retention.py`) instead of
`analysis/stage_summary_retention.py` and
`analysis/stage_b_summret_workflow.js`.
