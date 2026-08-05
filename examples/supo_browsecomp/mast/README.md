# BrowseComp on MAST

This directory contains MAST-specific launchers and operational workflows. The
model rollout and reward implementation remains scheduler-independent in the
parent directory.

## Workflows

| Workflow | Entry point | Documentation |
|---|---|---|
| Training | `submit_experiment.sh` | [`SUBMIT_NEW_TRAINING_JOB.md`](SUBMIT_NEW_TRAINING_JOB.md) |
| Checkpoint evaluation | `eval/eval_sweep.py` | [`eval/README.md`](eval/README.md) |
| Search service | `search/run_search_server.sh` | [`search/README.md`](search/README.md) |
| Checkpoint slimming | `checkpoint_slim/checkpoint_slim.py` | [`checkpoint_slim/README.md`](checkpoint_slim/README.md) |
| W&B online/offline logging | `wandb/submit_with_wandb.sh`, `wandb/wandb_sync.sh` | [`wandb/README.md`](wandb/README.md) |
| W&B online smoke | `wandb/smoke/submit_wandb_online_smoke.sh` | [`wandb/README.md`](wandb/README.md) |

## Directory ownership

- `configs/` contains immutable per-training-run resource and experiment
  settings.
- Training launchers remain at the MAST root because they are shared by all
  current training configs.
- `eval/` owns checkpoint discovery, eval submission, monitoring, resumption,
  and report synchronization.
- `search/` owns the MAST search-service launcher and operational runbook.
- `checkpoint_slim/` owns weights-only checkpoint conversion and canary
  validation.
- `wandb/` owns online/offline transport, recovery, health checks, and the
  isolated MAST connectivity smoke.

Do not put metric definitions or raw-dump parsing in this directory. Those
belong in [`../eval/eval_pipeline.py`](../eval/eval_pipeline.py), so another
scheduler can reuse the same correctness and report contract.

## Training data

New training runs default to
`/mnt/wsfuse/hhzhang01/supo-data/BC+/bc_train_exclude_stable91_20260730.parquet`:
589 questions after removing the 91 questions that scored 8/8 at base, checkpoint
4, and checkpoint 9. The original 680-question `bc_train.parquet` is unchanged.

Set `BC_TRAIN_DATA` to an absolute container path in an experiment config to
override the default. A resumed run must keep the dataset it originally used;
for an older run trained on all 680 questions, set:

```bash
BC_TRAIN_DATA=/mnt/wsfuse/hhzhang01/supo-data/BC+/bc_train.parquet
```

See [`SUBMIT_NEW_TRAINING_JOB.md`](SUBMIT_NEW_TRAINING_JOB.md) for the full
submission and resume procedure.

## Tool protocol

Training supports two tool-protocol modes. Select one complete preset in the
immutable experiment config; the submitter forwards the variables and the
trainer logs the effective protocol at startup.

### Model-controlled mode (SUPO baseline)

```bash
unset BCPLUS_FIXED_SEARCH_TOPK
unset BCPLUS_DOC_WORDS_FULL
```

- The `search` tool schema exposes the optional `topk` argument to the model.
- Missing or invalid `topk` uses 10; model-provided values are clamped to 1-20.
- `open_page` returns at most the first 4096 words.

Use `unset`, rather than assigning an empty string, so inherited environment
values cannot accidentally select another protocol.

### Fixed-budget mode

```bash
BCPLUS_FIXED_SEARCH_TOPK=5
BCPLUS_DOC_WORDS_FULL=10000
```

- The `topk` argument is removed from the `search` tool schema and execution
  always uses a weighted result budget of 5, ignoring stray model arguments.
- A new result costs 1 budget unit and a previously visited result costs 0.25.
- `open_page` returns at most the first 10000 words.

The two environment variables are independently configurable in code, but the
combinations above are the supported experiment presets. Setting only one
creates a different mixed protocol and should be done only as an explicitly
named ablation. Do not change protocol when resuming an existing logical run.

Checkpoint evaluation uses separate JSON presets under `eval/configs/` because
it is a different immutable workflow. Pass one to `eval_sweep.py` with the
`--eval-config` option; the effective settings are frozen into the batch's
`sweep_config.json`.

## Experimental W&B online smoke

`wandb/smoke/submit_wandb_online_smoke.sh` verifies that a MAST compute container can reach
and authenticate to `https://meta-3.wandb.io` through the fwdproxy configured by
MAST TTLS. It calls slime's real secondary tracking path, logs three metrics,
finishes the run, and tears down the W&B service. It does not use the offline
snapshot or devserver sync workflow.

The worktree must be clean because the submitter builds its code archive from
`HEAD`. Run a dry-run first, then submit once:

```bash
WANDB_KEY_FILE="${HOME}/.wandb-key" \
  bash examples/supo_browsecomp/mast/wandb/smoke/submit_wandb_online_smoke.sh --dry-run

WANDB_KEY_FILE="${HOME}/.wandb-key" \
  bash examples/supo_browsecomp/mast/wandb/smoke/submit_wandb_online_smoke.sh
```

The key is staged in a mode-`0600` OILFS file and expanded only inside the
compute container. It is not included in the MAST command, dry-run JSON, or W&B
run config. A successful job writes its non-secret result to the corresponding
devserver path under:

```text
/data/users/hhzhang01/wsfuse_mnt/hhzhang01/supo-slime/wandb-online-smoke-results/
```

The formal training path can enable online logging while retaining OILFS
recovery snapshots. See [`wandb/README.md`](wandb/README.md) for mode selection,
secret handling, failure behavior, and recovery commands.
