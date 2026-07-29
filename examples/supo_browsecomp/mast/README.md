# BrowseComp on MAST

This directory contains MAST-specific launchers and operational workflows. The
model rollout and reward implementation remains scheduler-independent in the
parent directory.

## Workflows

| Workflow | Entry point | Documentation |
|---|---|---|
| Training | `submit_experiment.sh` | [`SUBMIT_NEW_TRAINING_JOB.md`](SUBMIT_NEW_TRAINING_JOB.md) |
| Checkpoint evaluation | `eval/eval_sweep.py` | [`eval/README.md`](eval/README.md) |
| Search service | `run_search_server.sh` | [`SEARCH_SERVER.md`](SEARCH_SERVER.md) |
| Checkpoint slimming | `checkpoint_slim/checkpoint_slim.py` | [`checkpoint_slim/README.md`](checkpoint_slim/README.md) |
| W&B synchronization | `submit_with_wandb.sh`, `wandb_sync.sh` | Training runbook and script headers |

## Directory ownership

- `configs/` contains immutable per-training-run resource and experiment
  settings.
- Training launchers remain at the MAST root because they are shared by all
  current training configs.
- `eval/` owns checkpoint discovery, eval submission, monitoring, resumption,
  and report synchronization.
- `checkpoint_slim/` owns weights-only checkpoint conversion and canary
  validation.

Do not put metric definitions or raw-dump parsing in this directory. Those
belong in [`../eval/eval_pipeline.py`](../eval/eval_pipeline.py), so another
scheduler can reuse the same correctness and report contract.
