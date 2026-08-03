# BrowseComp on AWS (Slurm + enroot)

This directory contains the AWS-cluster launchers for the SUPO / BrowseComp-Plus
pipeline. Everything here is scheduler-specific: two-part (login pod →
in-container) scripts that submit `srun`/`sbatch` jobs and run enroot on
`/genai/fsx-project/...`. The model rollout, reward, and eval-metric
implementation stays scheduler-independent in the parent directory. The MAST
counterparts live under [`../mast/`](../mast/README.md).

## Workflows

| Workflow | Entry point |
|---|---|
| Training (8-node colocate) | `run_qwen3p5_4B_colocate.sh` (canonical), `run_qwen3p5_9B_colocate.sh` |
| Search service | `search/launch_search_server.sh` |
| HF → mcore conversion (one-shot) | `convert_qwen3p5_9B.sh` |

See [`../README.md`](../README.md) for prerequisites (search server, MetaGen
judge, checkpoints, parquet) and the full run walkthrough.

## Checkpoint storage

Run `/genai` filesystem checks and all Slurm submissions from the cluster login
shell reached through `/home/hhzhang01/.tmux-forclaude` (session `v2_dev`). The
Codex execution container does not expose the cluster's `/genai` mounts, so a
missing path there does not mean the path is absent on the AWS cluster.

The 4B launcher saves new checkpoints under the large intern FSx filesystem by
default:

```text
/genai/fsx-llm/interns/hhzhang01/checkpoints/${RUN_NAME}
```

It mounts `/genai/fsx-llm/interns/hhzhang01` as `/genai_llm` inside enroot and
passes the container-visible path to slime. Override `BCPLUS_CHECKPOINT_ROOT`
only when a run needs another location. In particular, resuming an older run
that was saved on the project FSx requires both its original `RUN_NAME` and:

```bash
BCPLUS_CHECKPOINT_ROOT=/genai/fsx-project/hhzhang01/checkpoints
```

Logs, rollout dumps, and coordination files remain under
`/genai/fsx-project/hhzhang01`; this setting changes checkpoint storage only.

New 4B runs enable rolling checkpoint slimming by default. After each
successful save, the newest checkpoint remains full so training can resume
with optimizer, scheduler, and RNG state; the preceding checkpoint is
atomically replaced by a weights-only copy. Set
`BC_SLIM_INTERMEDIATE_CHECKPOINTS=0` only when every historical checkpoint
must remain resumable. Changing this setting does not affect an already
running launcher process.

## Rootfs rules for single-node and multi-node runs

| Run mode | Required behavior |
|---|---|
| Persistent single-node debug | Set `DEV_ALLOCATION_JOB_ID` and `NUM_NODES=1`. Use the shared imported rootfs directly; do not stage it to `/dev/shm`. |
| Normal multi-node job | Leave `DEV_ALLOCATION_JOB_ID` unset. Each node stages the rootfs once under `/dev/shm/enroot-${USER}-${SLURM_JOB_ID}`. |
| Persistent multi-node allocation | Use staged mode. The first step stages once on every node; later steps in the same allocation reuse the same path because `SLURM_JOB_ID` and the node list are unchanged. |

For persistent multi-node runs, never start all nodes directly from the shared
rootfs. Confirm the first step logs `copying rootfs` followed by `rootfs staged`.
Later steps should skip the copy. If the launcher logs `using shared enroot
rootfs for dev allocation` with `NUM_NODES` greater than 1, stop the run because
the wrong rootfs mode was selected.

Staging uses `rsync` and excludes only imported host/Slurm runtime residue:
`tmp_host/` and `var/spool/slurmd/pmix.*`. These paths are recreated when the
container starts and are not image contents. A `.slime-stage-complete` marker
is written only after a successful copy, so a retry completes a partial stage
instead of treating the directory itself as proof of success. Confirm the log
prints `rootfs staged` before relying on the local copy.

Reuse lasts only for the lifetime of the same persistent allocation. Before
releasing a persistent multi-node allocation, remove its staging directory on
every node, then cancel the holder job. A new allocation has a new
`SLURM_JOB_ID` and stages again.

## Persistent one-node development allocation

The 4B topology uses all eight GPUs on one node (`TP=4`, `CP=2`). For repeated
smoke runs, keep one node allocated and attach each run as a Slurm job step.
This development path uses the shared, already-imported enroot rootfs directly,
so retries do not pull or copy the image. Normal multi-node launches still stage
the rootfs under `/dev/shm` to avoid concurrent NFS lock races.

From a persistent login-node tmux session, create a 24-hour holder job:

```bash
sbatch --parsable \
  --nodes=1 --gpus-per-node=8 --ntasks-per-node=1 --exclusive \
  --cpus-per-task=64 --mem=0 \
  --account=genai_interns --qos=a100_genai_interns_high \
  --time=24:00:00 --job-name=supo-dev-node \
  --output=/genai/fsx-project/hhzhang01/logs/supo-dev-node-%j.log \
  --wrap='while true; do sleep 3600; done'
```

After the holder reaches `RUNNING`, pass its numeric job ID to the launcher:

```bash
DEV_ALLOCATION_JOB_ID=<job-id> \
NUM_NODES=1 \
BC_NUM_ROLLOUT=1 \
bash examples/supo_browsecomp/aws/run_qwen3p5_4B_colocate.sh
```

The launcher uses `srun --jobid=<job-id> --overlap` instead of requesting a new
allocation. Run only one training step at a time. When development is finished,
confirm no step is active and release the holder with `scancel <job-id>`.

## Directory ownership

- Training launchers live at this root because both 4B and 9B share the same
  colocate machinery (per-node `/dev/shm` rootfs staging, ray head, weight sync,
  wandb-sync poll). `run_qwen3p5_4B_colocate.sh` is the live canonical script;
  `run_qwen3p5_9B_colocate.sh` is its 9B sibling.
- `search/` owns the long-lived retrieval-server orchestrator. It is idempotent:
  reuses a running `supo-search-server` job with enough walltime left, otherwise
  scancels + resubmits. Both colocate scripts call it automatically.
- `convert_qwen3p5_9B.sh` is a one-shot preflight that turns the Qwen3.5-9B HF
  checkpoint into the Megatron `torch_dist` format. It writes to FSx
  (`/genai/fsx-project/.../models/Qwen3.5-9B_torch_dist`). MAST consumes the
  same artifact from its own path and does not run this script.

The 1-node debug wrappers (`../debug_scripts/`) are kept out of git
(personal tooling); they re-use these launchers' in-container half via
`SLIME_INNER=1`.

Do not put rollout, reward, or metric logic here. That belongs in the parent
directory's shared Python (`generate_with_bcplus.py`, `dynamic_sampling.py`,
`summary_advantage.py`, `eval/eval_pipeline.py`) so MAST can reuse it.
